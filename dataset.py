import os
import json
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RuCoCoDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            tokenizer_name: str = "DeepPavlov/rubert-base-cased",
            max_length: int = 512):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length

    def _char_to_token_span(
            self,
            offset_mapping: list[tuple[int, int]],
            start_char: int,
            end_char: int) -> tuple[int, int]:
        token_start = None
        token_end = None

        for i, (start, end) in enumerate(offset_mapping):
            if start == end == 0:
                continue # скип спешл токенов

            if token_start is None and start <= start_char < end:
                token_start = i

            if start < end_char <= end:
                token_end = i

        # если token end не был найден, но token_start был - ищем ближайший правый конец
        if token_start is not None and token_end is None:
            for i in reversed(range(len(offset_mapping))):
                s, e = offset_mapping[i]
                if s == e == 0:
                    continue
                if s >= end_char:
                    continue
                token_end = i
                break

        return token_start, token_end

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        filepath = self.files[idx]
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text = data["text"]
        entities = data["entities"]

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        offset_mapping = encoding.offset_mapping.squeeze(0).tolist()
        mentions = []
        mention_to_cluster = []

        for cluster_idx, cluster in enumerate(entities):
            for start_char, end_char in cluster:
                token_start, token_end = self._char_to_token_span(offset_mapping, start_char, end_char)

                if token_start is not None and token_end is not None:
                    mentions.append((token_start, token_end))
                    mention_to_cluster.append(cluster_idx)

        # антеценденты для каждого упоминания
        max_antecedents = 50
        antecedent_indices = []
        antecedent_labels = []
        gold_antecedents = []

        for i, cluster_id in enumerate(mention_to_cluster):
            start_idx = max(0, i - max_antecedents)
            candidates = list(range(start_idx, i))
            labels = [1 if mention_to_cluster[c] == cluster_id else 0 for c in candidates]

            padding = max_antecedents - len(candidates)
            candidates = [-1] * padding + candidates
            labels = [0] * padding + labels

            antecedent_indices.append(candidates)
            antecedent_labels.append(labels)

            try:
                gold_index_in_candidates = labels.index(1)
                gold_antecedents.append(candidates[gold_index_in_candidates])
            except ValueError:
                gold_antecedents.append(-1)

        return {
            'input_ids': encoding.input_ids.squeeze(0),
            'attention_mask': encoding.attention_mask.squeeze(0),
            'mentions': mentions,
            'mention_to_cluster': mention_to_cluster,
            'antecedent_indices': torch.tensor(antecedent_indices, dtype=torch.long),
            'antecedent_labels': torch.tensor(antecedent_labels, dtype=torch.float),
            'gold_antecedents': torch.tensor(gold_antecedents, dtype=torch.long),
            'text': text
        }

def collate_fn(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)

    all_mentions = [item['mentions'] for item in batch]
    all_clusters = [item['mention_to_cluster'] for item in batch]

    antecedent_indices = pad_sequence([item['antecedent_indices'] for item in batch], batch_first=True, padding_value=-1)
    antecedent_labels = pad_sequence([item['antecedent_labels'] for item in batch], batch_first=True, padding_value=0)

    gold_antecedents = pad_sequence([item['gold_antecedents'] for item in batch], batch_first=True, padding_value=-1)

    texts = [item['text'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'mentions': all_mentions,
        'mention_to_cluster': all_clusters,
        'antecedent_indices': antecedent_indices,
        'antecedent_labels': antecedent_labels,
        'gold_antecedents': gold_antecedents,
        'text': texts
    }


if __name__ == "__main__":
    dataset = RuCoCoDataset(data_dir="RuCoCo")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

    for batch in dataloader:
        print("input_ids shape:", batch['input_ids'].shape)
        print("attention_mask shape:", batch['attention_mask'].shape)
        print("mentions (batch):", batch['mentions'])
        print("antecedent indices shape:", batch['antecedent_indices'].shape)
        print("antecedent labels shape:", batch['antecedent_labels'].shape)
        print("text:", batch['text'][0][:100])
        break