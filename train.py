import os
import json
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from utils import coref_loss, get_gold_antecedents, visualize_scores_save
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
    threshold = config.get("threshold", 0.8)
    top_k = config.get("top_k", 50)
    val_top_k = config.get("val_top_k", 15)
    loss_weight = config.get("mention_loss_weight", 0.5)
    reg_weight = config.get("reg_weight", 0.01)

    os.makedirs(save_path, exist_ok=True)

    dataset = RuCoCoDataset(data_dir=data_dir)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False)

    model = SpanBert()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.bert.named_parameters() if p.requires_grad], "lr": 1e-5},
        {"params": model.mention_scorer.parameters(), "lr": 1e-4},
        {"params": model.pairwise_scorer.parameters(), "lr": 1e-4},
    ], weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True)

        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mention_to_cluster_batch = batch['mention_to_cluster']

            batch_span_starts = [[s for s, _ in m] for m in batch['mentions']]
            batch_span_ends = [[e for _, e in m] for m in batch['mentions']]

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_starts=batch_span_starts,
                span_ends=batch_span_ends,
                top_k=top_k
            )

            (
                mention_scores_batch,
                antecedent_scores_batch,
                filtered_span_starts_batch,
                filtered_span_ends_batch,
                antecedent_masks_batch
            ) = out

            losses = []
            for b in range(len(mention_scores_batch)):
                if mention_scores_batch[b].numel() == 0:
                    continue

                mention_scores = mention_scores_batch[b]
                antecedent_scores = antecedent_scores_batch[b]
                mention_to_cluster = mention_to_cluster_batch[b]

                mention_spans = batch['mentions'][b]
                span_to_label = {(s, e): int(cid != -1) for (s, e), cid in zip(mention_spans, mention_to_cluster)}

                filtered_spans = list(zip(filtered_span_starts_batch[b], filtered_span_ends_batch[b]))
                mention_labels = torch.tensor(
                    [span_to_label.get(span, 0) for span in filtered_spans],
                    dtype=torch.float, device=device
                )

                span_to_cluster = {(s, e): cid for (s, e), cid in zip(mention_spans, mention_to_cluster)}
                filtered_clusters = [span_to_cluster.get(span, -1) for span in filtered_spans]

                gold_antecedents = get_gold_antecedents(list(range(len(filtered_clusters))), filtered_clusters)
                gold_antecedents = torch.tensor(gold_antecedents, dtype=torch.long, device=device).unsqueeze(0)

                mention_loss = F.binary_cross_entropy_with_logits(mention_scores, mention_labels)
                reg_loss = reg_weight * torch.norm(mention_scores, p=2)
                loss = coref_loss(antecedent_scores.unsqueeze(0), gold_antecedents, antecedent_masks_batch[b].unsqueeze(0))
                total_loss = loss + loss_weight * mention_loss + reg_loss
                losses.append(total_loss)

            if losses:
                loss = torch.stack(losses).mean()
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

                batch_span_starts = [[s for s, _ in m] for m in batch['mentions']]
                batch_span_ends = [[e for _, e in m] for m in batch['mentions']]

                (
                    mention_scores_batch,
                    antecedent_scores_batch,
                    filtered_span_starts_batch,
                    filtered_span_ends_batch,
                    antecedent_masks_batch
                ) = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    span_starts=batch_span_starts,
                    span_ends=batch_span_ends,
                    top_k=val_top_k
                )

                for b in range(len(mention_scores_batch)):
                    if mention_scores_batch[b].numel() == 0:
                        continue

                    mention_scores = mention_scores_batch[b]
                    antecedent_scores = antecedent_scores_batch[b]
                    span_starts = filtered_span_starts_batch[b]
                    span_ends = filtered_span_ends_batch[b]
                    mention_to_cluster = mention_to_cluster_batch[b]

                    mention_spans = batch['mentions'][b]
                    span_to_label = {(s, e): int(cid != -1) for (s, e), cid in zip(mention_spans, mention_to_cluster)}
                    filtered_spans = list(zip(span_starts, span_ends))

                    mention_labels = torch.tensor(
                        [span_to_label.get(span, 0) for span in filtered_spans],
                        dtype=torch.float, device=device
                    )

                    span_to_cluster = {(s, e): cid for (s, e), cid in zip(mention_spans, mention_to_cluster)}
                    filtered_clusters = [span_to_cluster.get(span, -1) for span in filtered_spans]

                    gold_antecedents = get_gold_antecedents(list(range(len(filtered_clusters))), filtered_clusters)
                    gold_antecedents = torch.tensor(gold_antecedents, dtype=torch.long, device=device).unsqueeze(0)

                    mention_loss = F.binary_cross_entropy_with_logits(mention_scores, mention_labels)
                    loss = coref_loss(antecedent_scores.unsqueeze(0), gold_antecedents, antecedent_masks_batch[b].unsqueeze(0))
                    reg_loss = reg_weight * torch.norm(mention_scores, p=2)
                    total_loss = loss + loss_weight * mention_loss + reg_loss

                    val_losses.append(total_loss)

                    pred_clusters = get_predicted_clusters(
                        filtered_span_starts_batch[b], filtered_span_ends_batch[b], antecedent_scores,
                        mention_scores=mention_scores, mention_threshold=threshold, antecedent_score_threshold=0.75
                    )
                    gold_clusters = get_gold_clusters(filtered_spans, filtered_clusters)

                    p, r, f1 = coref_metrics(pred_clusters, gold_clusters)
                    all_precisions.append(p)
                    all_recalls.append(r)
                    all_f1s.append(f1)

                    if b % 10 == 0:
                        visualize_scores_save(mention_scores, antecedent_scores, epoch=epoch + 1, batch_idx=b + 1)

        print(f"[Validation Epoch {epoch}] Loss: {sum(val_losses) / len(val_losses):.4f}, "
              f"P: {sum(all_precisions)/len(all_precisions):.3f}, "
              f"R: {sum(all_recalls)/len(all_recalls):.3f}, "
              f"F1: {sum(all_f1s)/len(all_f1s):.3f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
