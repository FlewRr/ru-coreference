from transformers import AutoModel

import torch
from torch import nn
from torch.nn import functional as F

from utils import filter_overlapping_spans


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

        batch_idx = torch.arange(batch_size, device=sequence_output.device).unsqueeze(1).expand(-1, span_starts.size(1))
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

    def forward(self, span1: torch.Tensor, span2: torch.Tensor) -> torch.Tensor:
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

    def forward(
            self,
            sequence_output: torch.Tensor,
            span_starts: torch.Tensor,
            span_ends: torch.Tensor) -> torch.Tensor:
        span_emb = self.span_repr(sequence_output, span_starts, span_ends)
        scores = self.mention_scorer(span_emb).squeeze(-1)
        return scores


class SpanBert(nn.Module):
    def __init__(
            self,
            bert_backbone_name: str = "DeepPavlov/rubert-base-cased",
            max_span_width: int = 10,
            top_k: int = 50):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_backbone_name)
        hidden_size = self.bert.config.hidden_size
        self.mention_scorer = MentionScorer(hidden_size, max_span_width)
        self.pairwise_scorer = PairwiseScorer(hidden_size)
        self.max_span_width = max_span_width
        self.top_k = top_k

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None,
            span_starts: torch.Tensor,
            span_ends: torch.Tensor,
            top_k: int | None = None
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        all_mention_scores = []
        all_antecedent_scores = []
        filtered_span_starts_batch = []
        filtered_span_ends_batch = []
        all_antecedent_masks = []

        for b in range(batch_size):
            starts = span_starts[b]
            ends = span_ends[b]

            if len(starts) == 0:
                all_mention_scores.append(torch.tensor([]).to(device))
                all_antecedent_scores.append(torch.tensor([]).to(device))
                filtered_span_starts_batch.append([])
                filtered_span_ends_batch.append([])
                all_antecedent_masks.append(torch.tensor([]).to(device))
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

            spans = list(zip(filtered_starts, filtered_ends))
            mention_scores_list = mention_scores.cpu().tolist()
            filtered_spans_nms = filter_overlapping_spans(spans, mention_scores_list)
            indices_to_keep = [spans.index(span) for span in filtered_spans_nms]

            span_repr = span_repr[indices_to_keep]
            mention_scores = mention_scores[indices_to_keep]
            filtered_starts = [filtered_starts[i] for i in indices_to_keep]
            filtered_ends = [filtered_ends[i] for i in indices_to_keep]

            n = span_repr.size(0)
            antecedent_scores = torch.full((n, n), float("-inf"), device=device)
            antecedent_mask = torch.zeros((n, n), dtype=torch.bool, device=device)

            for i in range(n):
                if i == 0:
                    continue
                valid_antecedents = span_repr[:i]
                repeated = span_repr[i].unsqueeze(0).expand(i, -1)
                scores = self.pairwise_scorer(repeated, valid_antecedents)
                antecedent_scores[i, :i] = scores
                antecedent_mask[i, :i] = True

            all_mention_scores.append(mention_scores)
            all_antecedent_scores.append(antecedent_scores)
            filtered_span_starts_batch.append(filtered_starts)
            filtered_span_ends_batch.append(filtered_ends)
            all_antecedent_masks.append(antecedent_mask)

        return all_mention_scores, all_antecedent_scores, filtered_span_starts_batch, filtered_span_ends_batch, all_antecedent_masks
