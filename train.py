import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import coref_loss, get_gold_antecedents
from dataset import collate_fn, RuCoCoDataset
from model import SpanBert
from clusters import get_predicted_clusters
from argparse import ArgumentParser

def get_gold_clusters(mentions, mention_to_cluster):
    cluster_map = {}
    for mention, cluster_id in zip(mentions, mention_to_cluster):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(mention)
    return list(cluster_map.values())

def extract_coref_links(clusters):
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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path) as f:
        config = json.load(f)

    lr = config.get("lr", 3e-5)
    epochs = config.get("epochs", 3)
    train_batch_size = config.get("train_batch_size", 2)
    eval_batch_size = config.get("eval_batch_size", 2)
    save_every = config.get("save_every", 1)
    save_path = config.get("save_path", "checkpoints")
    data_dir = config.get("data_dir", "RuCoCo")

    os.makedirs(save_path, exist_ok=True)

    dataset = RuCoCoDataset(data_dir=data_dir)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False)

    model = SpanBert()

    for param in model.bert.parameters():
        param.requires_grad = False
    # TO DELETE
    # optimizer = torch.optim.Adam([
    #     {"params": model.bert.parameters(), "lr": 2e-5},
    #     {"params": model.mention_scorer.parameters(), "lr": 2e-4},
    #     {"params": model.pairwise_scorer.parameters(), "lr": 2e-4}
    # ])
    ## TO DELETE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True)
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mention_to_cluster_batch = batch['mention_to_cluster']

            batch_span_starts, batch_span_ends = [], []
            for mentions in batch['mentions']:
                starts = [s for s, _ in mentions]
                ends = [e for _, e in mentions]
                batch_span_starts.append(starts)
                batch_span_ends.append(ends)

            mention_scores_batch, antecedent_scores_batch, filtered_span_starts_batch, filtered_span_ends_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_starts=batch_span_starts,
                span_ends=batch_span_ends,
                top_k=30
            )

            losses = []
            batch_size = len(mention_scores_batch)
            for b in range(batch_size):
                mention_scores = mention_scores_batch[b]
                antecedent_scores = antecedent_scores_batch[b]
                mention_to_cluster = mention_to_cluster_batch[b]
                filtered_indices = list(range(len(mention_scores)))
                # gold_antecedents = get_gold_antecedents(filtered_indices, mention_to_cluster)
                gold_antecedents = get_gold_antecedents(
                    topk_indices=filtered_indices,
                    mention_to_cluster=mention_to_cluster,
                    all_mentions=batch['mentions'][b],
                    input_ids=input_ids[b],
                    tokenizer=dataset.tokenizer  # или где он у тебя
                )
                gold_antecedents = torch.tensor(gold_antecedents, dtype=torch.long, device=device).unsqueeze(0)

                loss = coref_loss(
                    antecedent_scores.unsqueeze(0),
                    gold_antecedents
                )
                losses.append(loss)

            loss = torch.stack(losses).mean()
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Avg Train Loss = {total_train_loss / len(train_loader):.4f}")

        # Валидация
        model.eval()
        val_losses, all_precisions, all_recalls, all_f1s = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                mention_to_cluster_batch = batch['mention_to_cluster']

                batch_span_starts, batch_span_ends = [], []
                for mentions in batch['mentions']:
                    starts = [s for s, _ in mentions]
                    ends = [e for _, e in mentions]
                    batch_span_starts.append(starts)
                    batch_span_ends.append(ends)

                mention_scores_batch, antecedent_scores_batch, filtered_span_starts_batch, filtered_span_ends_batch = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    span_starts=batch_span_starts,
                    span_ends=batch_span_ends,
                    top_k=30
                )

                for b in range(len(mention_scores_batch)):
                    mention_scores = mention_scores_batch[b]
                    antecedent_scores = antecedent_scores_batch[b]
                    span_starts = filtered_span_starts_batch[b]
                    span_ends = filtered_span_ends_batch[b]
                    mention_to_cluster = mention_to_cluster_batch[b]
                    filtered_indices = list(range(len(mention_scores)))
                    # gold_antecedents = get_gold_antecedents(filtered_indices, mention_to_cluster)
                    gold_antecedents = get_gold_antecedents(
                        topk_indices=filtered_indices,
                        mention_to_cluster=mention_to_cluster,
                        all_mentions=batch['mentions'][b],
                        input_ids=input_ids[b],
                        tokenizer=dataset.tokenizer  # или где он у тебя
                    )
                    gold_antecedents = torch.tensor(gold_antecedents, dtype=torch.long, device=device).unsqueeze(0)

                    loss = coref_loss(
                        antecedent_scores.unsqueeze(0),
                        gold_antecedents
                    )
                    val_losses.append(loss.item())

                    pred_clusters = get_predicted_clusters(span_starts, span_ends, antecedent_scores, threshold=0.5)
                    pred_clusters = [c for c in pred_clusters if len(c) > 1]
                    gold_clusters = get_gold_clusters(batch['mentions'][b], batch['mention_to_cluster'][b])
                    p, r, f1 = coref_metrics(pred_clusters, gold_clusters)
                    all_precisions.append(p)
                    all_recalls.append(r)
                    all_f1s.append(f1)

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_p = sum(all_precisions) / len(all_precisions)
        avg_r = sum(all_recalls) / len(all_recalls)
        avg_f1 = sum(all_f1s) / len(all_f1s)
        print(f"[Validation Epoch {epoch}] Loss: {avg_val_loss:.4f}, P: {avg_p:.3f}, R: {avg_r:.3f}, F1: {avg_f1:.3f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
