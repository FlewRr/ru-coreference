import torch
from torch.utils.data import DataLoader
from utils import coref_loss, get_gold_antecedents
from dataset import collate_fn, RuCoCoDataset
from model import SpanBert
from clusters import get_predicted_clusters

def get_gold_clusters(mentions, mention_to_cluster):
    """
    mentions: List[Tuple[int, int]]
    mention_to_cluster: List[int]
    """
    cluster_map = {}
    for mention, cluster_id in zip(mentions, mention_to_cluster):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(mention)
    return list(cluster_map.values())


def extract_coref_links(clusters):
    """
    clusters: List[List[Tuple[int, int]]]
    Возвращает set пар (a, b) для всех coreferent пар (упорядоченных)
    """
    links = set()
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                links.add((cluster[i], cluster[j]))
    return links


def coref_metrics(pred_clusters, gold_clusters):
    pred_links = extract_coref_links(pred_clusters)
    gold_links = extract_coref_links(gold_clusters)

    tp = len(pred_links & gold_links)
    fp = len(pred_links - gold_links)
    fn = len(gold_links - pred_links)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1



if __name__ == "__main__":
    dataset = RuCoCoDataset(data_dir="RuCoCo")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    model = SpanBert()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(3):
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mention_to_cluster_batch = batch['mention_to_cluster']

            batch_span_starts = []
            batch_span_ends = []

            for mentions in batch['mentions']:
                starts = [s for s, e in mentions]
                ends = [e for s, e in mentions]
                batch_span_starts.append(starts)
                batch_span_ends.append(ends)

            mention_scores_batch, antecedent_scores_batch, filtered_span_starts_batch, filtered_span_ends_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_starts=batch_span_starts,
                span_ends=batch_span_ends,
            )

            losses = []
            batch_size = len(mention_scores_batch)
            for b in range(batch_size):
                mention_scores = mention_scores_batch[b]  # (num_filtered_spans,)
                antecedent_scores = antecedent_scores_batch[b]  # (num_filtered_spans, num_filtered_spans)
                filtered_span_starts = filtered_span_starts_batch[b]
                filtered_span_ends = filtered_span_ends_batch[b]
                mention_to_cluster = mention_to_cluster_batch[b]

                # Формируем gold_antecedents только для отфильтрованных спанов
                # Нужно сопоставить индексы filtered_span с их кластерами
                filtered_indices = list(range(len(filtered_span_starts)))  # индексы отфильтрованных спанов
                gold_antecedents = get_gold_antecedents(filtered_indices, mention_to_cluster)
                gold_antecedents = torch.tensor(gold_antecedents, dtype=torch.long, device=device).unsqueeze(0)

                loss = coref_loss(
                    mention_scores.unsqueeze(0),
                    antecedent_scores.unsqueeze(0),
                    gold_antecedents
                )
                losses.append(loss)

            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Для метрик используем именно отфильтрованные спаны
                for b in range(batch_size):
                    span_starts = filtered_span_starts_batch[b]
                    span_ends = filtered_span_ends_batch[b]
                    pairwise_scores = antecedent_scores_batch[b]

                    pred_clusters = get_predicted_clusters(span_starts, span_ends, pairwise_scores, threshold=0.5)
                    gold_clusters = get_gold_clusters(batch['mentions'][b], batch['mention_to_cluster'][b])

                    p, r, f1 = coref_metrics(pred_clusters, gold_clusters)
                    print(f"[Train Metrics] Batch {b} Epoch {epoch}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
