import argparse
import torch
from transformers import AutoTokenizer
from model import SpanBert
from clusters import get_predicted_clusters

def get_spans(offset_mapping, max_span_width=10):
    spans = []
    for start in range(len(offset_mapping)):
        for end in range(start, min(start + max_span_width, len(offset_mapping))):
            if offset_mapping[start] is None or offset_mapping[end] is None:
                continue
            spans.append((start, end))
    return spans

def reconstruct_text_spans(span_indices, offset_mapping, original_text):
    span_texts = []
    for start_idx, end_idx in span_indices:
        start_char = offset_mapping[start_idx][0]
        end_char = offset_mapping[end_idx][1]
        span_texts.append(original_text[start_char:end_char])
    return span_texts

def run_inference(text, model, tokenizer, device, threshold=0.5):
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

    offset_mapping = [
        (start, end) if start != end else None
        for start, end in offset_mapping
    ]

    spans = get_spans(offset_mapping)
    span_starts = [s for s, e in spans]
    span_ends = [e for s, e in spans]

    model.eval()
    with torch.no_grad():
        mention_scores, antecedent_scores, span_starts_out, span_ends_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_starts=[span_starts],
            span_ends=[span_ends],
        )

    pred_clusters = get_predicted_clusters(
        span_starts_out[0],
        span_ends_out[0],
        antecedent_scores[0],
        threshold=threshold
    )

    # Reconstruct span texts from clusters
    result_clusters = []
    for cluster in pred_clusters:
        cluster_texts = reconstruct_text_spans(cluster, offset_mapping, text)
        result_clusters.append(cluster_texts)

    return result_clusters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint (.pt)')
    parser.add_argument('--text_path', type=str, required=True, help='Path to input .txt file')
    args = parser.parse_args()

    # Load text
    with open(args.text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = SpanBert()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint if provided
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

    clusters = run_inference(text, model, tokenizer, device)

    print("\n=== Coreference Clusters ===")
    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {cluster}")

if __name__ == "__main__":
    main()
