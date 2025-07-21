# def coref_loss(antecedent_scores, gold_antecedents):
#
#     batch_size, num_mentions, max_antecedents = antecedent_scores.shape
#     null_scores = torch.zeros(batch_size, num_mentions, 1, device=antecedent_scores.device)
#     scores_with_null = torch.cat([null_scores, antecedent_scores], dim=-1)  # (B, M, max_antecedents+1)
#
#     gold_antecedents_clamped = gold_antecedents.clone()
#     gold_antecedents_clamped[gold_antecedents_clamped == -1] = 0
#
#     log_probs = F.log_softmax(scores_with_null, dim=-1)  # (B, M, max_antecedents+1)
#     gold_log_probs = torch.gather(log_probs, 2, gold_antecedents_clamped.unsqueeze(-1)).squeeze(-1)  # (B, M)
#     loss = -gold_log_probs.mean()
#
#     return loss

import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def coref_loss(antecedent_scores, gold_antecedents, antecedent_mask=None):
    """
    antecedent_scores: (B, M, K)
    gold_antecedents: (B, M)
    antecedent_mask: (B, M, K), optional — True where valid
    """
    B, M, K = antecedent_scores.shape

    null_logits = torch.zeros(B, M, 1, device=antecedent_scores.device, dtype=antecedent_scores.dtype)
    scores_with_null = torch.cat([null_logits, antecedent_scores], dim=-1)  # (B, M, K+1)

    if antecedent_mask is not None:
        null_mask = torch.ones(B, M, 1, dtype=torch.bool, device=antecedent_scores.device)
        mask_with_null = torch.cat([null_mask, antecedent_mask], dim=-1)  # (B, M, K+1)
        scores_with_null = scores_with_null.masked_fill(~mask_with_null, float('-inf'))

    gold_antecedents_clamped = gold_antecedents.clone()
    gold_antecedents_clamped[gold_antecedents_clamped == -1] = 0  # null antecedent

    log_probs = F.log_softmax(scores_with_null, dim=-1)
    gold_log_probs = torch.gather(log_probs, 2, gold_antecedents_clamped.unsqueeze(-1)).squeeze(-1)

    return -gold_log_probs.mean()


def get_gold_antecedents(topk_indices, mention_to_cluster):
    """
    topk_indices: List[int] — индексы top-K меншионов
    mention_to_cluster: List[int] — соответствие всех меншионов кластерам
    Возвращает List[int] — индекс антецедента из topk_indices для каждого меншиона, или -1 если нет
    """
    gold_antecedents = []

    for i, idx_i in enumerate(topk_indices):
        cluster_i = mention_to_cluster[idx_i]

        if cluster_i == -1:
            gold_antecedents.append(-1)
            continue

        found_antecedent = -1
        for j in range(i):
            idx_j = topk_indices[j]
            if mention_to_cluster[idx_j] == cluster_i:
                found_antecedent = j
        gold_antecedents.append(found_antecedent)

    return gold_antecedents


def filter_overlapping_spans(spans, scores):
    """
    spans: list of (start, end)
    scores: torch.Tensor или list с оценками упоминаний

    Возвращает список отфильтрованных span, убирая пересекающиеся с меньшим score.
    """
    # Сортируем спаны по убыванию mention_scores
    sorted_indices = sorted(range(len(spans)), key=lambda i: scores[i], reverse=True)
    selected = []
    occupied = set()

    for idx in sorted_indices:
        start, end = spans[idx]
        # Проверяем пересечение с уже выбранными
        overlaps = False
        for s, e in selected:
            # Если есть пересечение, отбрасываем
            if not (end < s or start > e):
                overlaps = True
                break
        if not overlaps:
            selected.append((start, end))
    return selected


def visualize_scores_save(mention_scores, pairwise_scores, epoch=None, batch_idx=None, save_dir='plots'):
    import os
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib.colors")
    os.makedirs(save_dir, exist_ok=True)

    mention_scores = mention_scores.detach().cpu().numpy()
    pairwise_scores = pairwise_scores.detach().cpu().numpy()
    pairwise_scores = np.nan_to_num(pairwise_scores, nan=0.0, posinf=np.max(pairwise_scores), neginf=np.min(pairwise_scores))

    if np.isnan(pairwise_scores).any():
        print("NAN DETECTED in pairwise_scores!")
        raise ValueError("NaN in pairwise_scores")



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(mention_scores, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Mention Scores Histogram (Epoch {epoch}, Batch {batch_idx})')
    plt.xlabel('Mention Score')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.heatmap(pairwise_scores, cmap='coolwarm', center=0)
    plt.title(f'Pairwise Scores Heatmap (Epoch {epoch}, Batch {batch_idx})')
    plt.xlabel('Antecedent Index')
    plt.ylabel('Mention Index')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch{epoch}_batch{batch_idx}.png')
    plt.close()











# для каждого mention [i] - берем правильный антецедент gold_ante
# если gold_ante == -1, значит его для данного mention нет - скипаем
# берём скоры кандидатов-антецедентов antecedent_scores[b, i]
# конкатим слева скаляр 0.0
# смещаем индекс антецедента
# возвращаем (-1) * усредненный лог софтмакс для всех кандидатов в scores


# FAQ:
# Mention - ссылка на сущность в тексте (фраза, слово, группа)
# Антецедент - предыдущий mention к которому текущий mention ссылается
# Задача - связать все mention которые относятся к одному объекту в один кореферентный кластер
# Вроде вот так я задачу интерпретировал хз