import argparse
import torch
from transformers import AutoTokenizer
from model import SpanBert
from clusters import get_predicted_clusters

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, tokenizer, max_length=512):
        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        self.offset_mapping = self.encoding['offset_mapping'].squeeze(0).tolist()

    def _char_to_token_span(self, start_char, end_char):
        for idx, (start, end) in enumerate(self.offset_mapping):
            if start <= start_char < end and end_char <= end:
                return idx, idx
            elif start_char >= start and end_char <= end:
                return idx, idx
            elif start <= start_char < end_char <= end:
                return idx, idx
        return None, None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        input_ids = self.encoding['input_ids'].squeeze(0)
        attention_mask = self.encoding['attention_mask'].squeeze(0)

        # Dummy mentions for demonstration purposes — in practice you may run mention detection
        # Here we assume every 2-token window might be a mention (you can use your training logic)
        mention_spans = []
        for i in range(len(input_ids) - 1):
            mention_spans.append((i, i + 1))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mentions': mention_spans,
            'text': self.text
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'mentions': [item['mentions'] for item in batch],
        'text': [item['text'] for item in batch],
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    with open(args.input_txt, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    dataset = InferenceDataset(text, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    model = SpanBert()
    if args.checkpoint_path:
        print(f"Loading model from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    else:
        print("No checkpoint provided, using randomly initialized model.")

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mentions = batch['mentions'][0]
            span_starts = [s for s, e in mentions]
            span_ends = [e for s, e in mentions]

            mention_scores_batch, antecedent_scores_batch, filtered_starts, filtered_ends = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_starts=[span_starts],
                span_ends=[span_ends],
            )

            span_starts = filtered_starts[0]
            span_ends = filtered_ends[0]
            pairwise_scores = antecedent_scores_batch[0]

            clusters = get_predicted_clusters(span_starts, span_ends, pairwise_scores, threshold=0.5)

            print("\nPredicted coreference clusters:")
            for i, cluster in enumerate(clusters):
                print(f"Cluster {i + 1}: {[text[token_start:token_end+1] for token_start, token_end in cluster]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", type=str, required=True, help="Path to .txt file with input text")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint (.pt)")
    args = parser.parse_args()
    main(args)
