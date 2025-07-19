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
    def __init__(self, input_dim: int, hidden_dim: int = 150):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, span1: torch.Tensor, span2: torch.Tensor):
        features = torch.cat([
            span1,
            span2,
            span1 - span2,
            span1 * span2
        ], dim=-1)
        score = self.ff(features)
        return score.squeeze(-1)


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

    def forward(self, input_ids, attention_mask, span_starts, span_ends, top_k=None):
        batch_size = input_ids.size(0)
        device = input_ids.device
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        all_mention_scores = []
        all_antecedent_scores = []
        filtered_span_starts_batch = []
        filtered_span_ends_batch = []

        for b in range(batch_size):
            starts = span_starts[b]
            ends = span_ends[b]

            if len(starts) == 0:
                all_mention_scores.append(torch.tensor([]).to(device))
                all_antecedent_scores.append(torch.tensor([]).to(device))
                filtered_span_starts_batch.append([])
                filtered_span_ends_batch.append([])
                continue

            span_starts_tensor = torch.tensor(starts, device=device).unsqueeze(0)
            span_ends_tensor = torch.tensor(ends, device=device).unsqueeze(0)

            mention_scores = self.mention_scorer(sequence_output[b:b + 1], span_starts_tensor,
                                                 span_ends_tensor).squeeze(0)
            span_repr = self.mention_scorer.span_repr(sequence_output[b:b + 1], span_starts_tensor,
                                                      span_ends_tensor).squeeze(0)

            if top_k is not None and top_k < len(mention_scores):
                top_k_ = min(top_k, len(mention_scores))
                top_indices = torch.topk(mention_scores, k=top_k_).indices
                span_repr = span_repr[top_indices]
                mention_scores = mention_scores[top_indices]
                filtered_starts = [starts[i] for i in top_indices.cpu().tolist()]
                filtered_ends = [ends[i] for i in top_indices.cpu().tolist()]
            else:
                filtered_starts = starts
                filtered_ends = ends

            n = span_repr.size(0)
            if n == 0:
                antecedent_scores = torch.tensor([]).to(device)
            else:
                span1 = span_repr.unsqueeze(1).expand(-1, n, -1)
                span2 = span_repr.unsqueeze(0).expand(n, -1, -1)
                pairwise_features = torch.cat([span1, span2, span1 - span2, span1 * span2], dim=-1)
                antecedent_scores = self.pairwise_scorer.ff(pairwise_features).squeeze(-1)
                antecedent_scores = torch.tril(antecedent_scores, diagonal=-1)

            all_mention_scores.append(mention_scores)
            all_antecedent_scores.append(antecedent_scores)
            filtered_span_starts_batch.append(filtered_starts)
            filtered_span_ends_batch.append(filtered_ends)

        return all_mention_scores, all_antecedent_scores, filtered_span_starts_batch, filtered_span_ends_batch
