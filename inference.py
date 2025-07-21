from typing import Any
import argparse

import torch
from transformers import AutoTokenizer

from model import SpanBert
from clusters import get_predicted_clusters


def get_spans(offset_mapping: list[tuple[int, int]], max_span_width: int = 10) -> list[tuple[int, int]]:
    spans = []
    for start in range(len(offset_mapping)):
        for end in range(start, min(start + max_span_width, len(offset_mapping))):
            if offset_mapping[start] is None or offset_mapping[end] is None:
                continue
            spans.append((start, end))
    return spans


def reconstruct_text_spans(span_indices: list[tuple[int, int]], offset_mapping: list[tuple[int, int]], original_text: str) -> list[str]:
    span_texts = []
    for start_idx, end_idx in span_indices:
        start_char = offset_mapping[start_idx][0]
        end_char = offset_mapping[end_idx][1]
        span_texts.append(original_text[start_char:end_char])
    return span_texts


def run_inference(
        text: str| list[str],
        model: SpanBert,
        tokenizer: AutoTokenizer,
        device: torch.device | str,
        threshold: float = 0.3,
        top_k: int = 30) -> list[Any]:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    offset_mapping = encoded['offset_mapping'][0].tolist()

    offset_mapping = [(s, e) if s != e else None for s, e in offset_mapping]
    valid_token_indices = [i for i, val in enumerate(offset_mapping) if val is not None]
    offset_mapping = [offset_mapping[i] for i in valid_token_indices]

    spans = get_spans(offset_mapping)
    span_starts = [s for s, e in spans]
    span_ends = [e for s, e in spans]

    model.eval()
    with torch.no_grad():
        mention_scores_batch, antecedent_scores_batch, span_starts_out, span_ends_out, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_starts=[span_starts],
            span_ends=[span_ends],
            top_k=top_k
        )

    mention_scores = mention_scores_batch[0]
    antecedent_scores = antecedent_scores_batch[0]
    starts = span_starts_out[0]
    ends = span_ends_out[0]

    if len(mention_scores) == 0:
        return []

    top_k = min(top_k, len(mention_scores))
    top_indices = torch.topk(mention_scores, top_k).indices.tolist()

    mention_scores = mention_scores[top_indices]
    filtered_starts = [starts[i] for i in top_indices]
    filtered_ends = [ends[i] for i in top_indices]
    filtered_antecedent_scores = antecedent_scores[top_indices][:, top_indices]

    pred_clusters = get_predicted_clusters(
        filtered_starts,
        filtered_ends,
        filtered_antecedent_scores,
        mention_scores=mention_scores,
        threshold=threshold
    )

    result_clusters = []
    for cluster in pred_clusters:
        cluster_texts = reconstruct_text_spans(cluster, offset_mapping, text)
        result_clusters.append(cluster_texts)

    print(f"\nTop mention scores: {[round(s.item(), 2) for s in mention_scores]}")
    print(f"Predicted clusters (token spans): {pred_clusters}")

    return result_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint (.pt)')
    parser.add_argument('--text_path', type=str, required=True, help='Path to input .txt file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for coreference linking')
    parser.add_argument('--top_k', type=int, default=100, help='Top-K mentions to consider')
    args = parser.parse_args()

    with open(args.text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = SpanBert()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

    clusters = run_inference(text, model, tokenizer, device, threshold=args.threshold, top_k=args.top_k)

    print("\n=== Coreference Clusters ===")
    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {cluster}")


if __name__ == "__main__":
    main()
