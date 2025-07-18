import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpanRepresentation(nn.Module):
    def __init__(self, hidden_size: int, max_span_width: int = 30, width_embedding_dim: int = 30):
        super().__init__()
        self.ff_start = nn.Linear(hidden_size, hidden_size)
        self.ff_end = nn.Linear(hidden_size, hidden_size)
        self.width_embedding = nn.Embedding(max_span_width + 1, width_embedding_dim)
        self.output_layer = nn.Linear(hidden_size * 2 + width_embedding_dim, hidden_size)

    def forward(self, sequence_output: torch.FloatTensor, span_starts: torch.LongTensor, span_ends: torch.LongTensor) -> torch.FloatTensor:
        batch_size = sequence_output.size(0)
        hidden_size = sequence_output.size(-1)

        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, span_starts.size(1))  # (batch, num_spans)
        start_embed = self.ff_start(sequence_output[batch_idx, span_starts])  # (batch, num_spans, hidden)
        end_embed = self.ff_end(sequence_output[batch_idx, span_ends])        # (batch, num_spans, hidden)

        span_widths = torch.clamp(span_ends - span_starts, min=0, max=self.width_embedding.num_embeddings - 1)
        width_embed = self.width_embedding(span_widths)  # (batch, num_spans, width_dim)

        span_embedding = torch.cat([start_embed, end_embed, width_embed], dim=-1)
        span_repr = self.output_layer(span_embedding)  # (batch, num_spans, hidden_size)
        return span_repr

class PairwiseScorer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, span_repr, antecedents_repr):
        # span_repr: (k, h), antecedents_repr: (k, h)
        k = span_repr.size(0)
        pairs = []
        for i in range(k):
            for j in range(i):
                pair = torch.cat([
                    span_repr[i],                     # span i
                    antecedents_repr[j],             # antecedent j
                    span_repr[i] - antecedents_repr[j]  # diff
                ])
                pairs.append(pair)
        if not pairs:
            return torch.zeros((k, k), device=span_repr.device)
        scores = self.ffn(torch.stack(pairs))
        pairwise_scores = torch.zeros((k, k), device=span_repr.device)
        idx = 0
        for i in range(k):
            for j in range(i):
                pairwise_scores[i, j] = scores[idx]
                idx += 1
        return pairwise_scores

class MentionScorer(nn.Module):
    def __init__(self, hidden_size: int, max_span_width: int = 10):
        super().__init__()
        self.span_repr = SpanRepresentation(hidden_size, max_span_width)
        self.mention_scorer = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output, span_starts, span_ends):
        span_emb = self.span_repr(sequence_output, span_starts, span_ends)
        scores = self.mention_scorer(span_emb).squeeze(-1)
        return scores

class SpanBert(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", max_span_width=10, top_k: int=10):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.mention_scorer = MentionScorer(hidden_size, max_span_width)
        self.pairwise_scorer = PairwiseScorer(hidden_size)
        self.max_span_width = max_span_width
        self.top_k = top_k

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            span_starts: list[list[int]],
            span_ends: list[list[int]],
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device

        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, T, H)

        all_mention_scores = []
        all_pairwise_scores = []
        filtered_span_starts_batch = []
        filtered_span_ends_batch = []

        for b in range(batch_size):
            starts = span_starts[b]
            ends = span_ends[b]

            if len(starts) == 0:
                starts = [0]
                ends = [0]

            # пример фильтрации (тут можно применить top-K или другие критерии)
            # пока просто берем все спаны без фильтрации
            filtered_starts = starts
            filtered_ends = ends

            filtered_span_starts_batch.append(filtered_starts)
            filtered_span_ends_batch.append(filtered_ends)

            span_starts_tensor = torch.tensor(filtered_starts, device=device).unsqueeze(0)  # (1, num_spans)
            span_ends_tensor = torch.tensor(filtered_ends, device=device).unsqueeze(0)  # (1, num_spans)

            mention_scores = self.mention_scorer(sequence_output[b:b + 1], span_starts_tensor,
                                                 span_ends_tensor).squeeze(0)  # (num_spans,)

            span_repr = self.mention_scorer.span_repr(sequence_output[b:b + 1], span_starts_tensor,
                                                      span_ends_tensor)[0]  # (num_spans, hidden)

            n = span_repr.size(0)
            pairwise_scores = torch.zeros((n, n), device=device)
            for i in range(n):
                for j in range(i):
                    span_i = span_repr[i].unsqueeze(0) if span_repr[i].dim() == 0 else span_repr[i]
                    span_j = span_repr[j].unsqueeze(0) if span_repr[j].dim() == 0 else span_repr[j]
                    pairwise_scores[i, j] = self.pairwise_scorer(span_i, span_j)

            all_mention_scores.append(mention_scores)
            all_pairwise_scores.append(pairwise_scores)

        return all_mention_scores, all_pairwise_scores, filtered_span_starts_batch, filtered_span_ends_batch
