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



from tqdm import tqdm

if __name__ == "__main__":
    dataset = RuCoCoDataset(data_dir="RuCoCo")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    model = SpanBert()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(3):
        epoch_loss = 0.0
        total_batches = 0

        epoch_p = 0.0
        epoch_r = 0.0
        epoch_f1 = 0.0
        total_samples = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=True)
        for batch in loop:
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
                mention_scores = mention_scores_batch[b]
                antecedent_scores = antecedent_scores_batch[b]
                filtered_span_starts = filtered_span_starts_batch[b]
                filtered_span_ends = filtered_span_ends_batch[b]
                mention_to_cluster = mention_to_cluster_batch[b]

                filtered_indices = list(range(len(filtered_span_starts)))
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

            epoch_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                for b in range(batch_size):
                    span_starts = filtered_span_starts_batch[b]
                    span_ends = filtered_span_ends_batch[b]
                    pairwise_scores = antecedent_scores_batch[b]

                    pred_clusters = get_predicted_clusters(span_starts, span_ends, pairwise_scores, threshold=0.5)
                    gold_clusters = get_gold_clusters(batch['mentions'][b], batch['mention_to_cluster'][b])

                    p, r, f1 = coref_metrics(pred_clusters, gold_clusters)
                    epoch_p += p
                    epoch_r += r
                    epoch_f1 += f1
                    total_samples += 1

            loop.set_postfix(loss=epoch_loss / total_batches,
                             precision=epoch_p / total_samples if total_samples else 0,
                             recall=epoch_r / total_samples if total_samples else 0,
                             f1=epoch_f1 / total_samples if total_samples else 0)

        print(f"\nEpoch {epoch} Mean Loss: {epoch_loss / total_batches:.4f}")
        print(f"Epoch {epoch} Mean Precision: {epoch_p / total_samples:.3f}, Recall: {epoch_r / total_samples:.3f}, F1: {epoch_f1 / total_samples:.3f}")
