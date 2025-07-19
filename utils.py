import torch
import torch.nn.functional as F

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
import torch.nn.functional as F
from matplotlib import pyplot as plt
import seaborn as sns

def coref_loss(antecedent_scores, gold_antecedents, antecedent_mask=None):
    """
    Вычисляет loss для кореференции с учетом фиктивного null-антецедента.

    Args:
        antecedent_scores: torch.Tensor, shape (B, M, K)
            Логиты модели для каждого меншиона и его K возможных антецедентов (без null).
        gold_antecedents: torch.LongTensor, shape (B, M)
            Индексы правильных антецедентов для каждого меншиона в диапазоне [0..K],
            где 0 — фиктивный null-антецедент.
        antecedent_mask: torch.BoolTensor или None, shape (B, M, K)
            Маска допустимых антецедентов (True — valid), если есть padding.

    Returns:
        loss: scalar tensor
    """

    B, M, K = antecedent_scores.shape

    # Добавляем фиктивный null-антецедент с логитом 0
    null_logits = torch.zeros(B, M, 1, device=antecedent_scores.device, dtype=antecedent_scores.dtype)
    scores_with_null = torch.cat([null_logits, antecedent_scores], dim=-1)  # (B, M, K+1)

    if antecedent_mask is not None:
        # Маска для всех K антецедентов (без null)
        # Добавим True для null-антецедента (его нет в маске, он всегда валиден)
        null_mask = torch.ones(B, M, 1, device=antecedent_mask.device, dtype=torch.bool)
        mask_with_null = torch.cat([null_mask, antecedent_mask], dim=-1)  # (B, M, K+1)

        # Чтобы softmax не учитывал паддинги, заливаем их -inf
        scores_with_null = scores_with_null.masked_fill(~mask_with_null, float('-inf'))

    # Если в gold антецедентах есть -1 (нет правильного антецедента), заменим на 0 (null)
    gold_antecedents_clamped = gold_antecedents.clone()
    gold_antecedents_clamped[gold_antecedents_clamped == -1] = 0

    # Лог-пробабильности
    log_probs = F.log_softmax(scores_with_null, dim=-1)  # (B, M, K+1)

    # Собираем лог-пробы для правильных антецедентов
    gold_log_probs = torch.gather(log_probs, 2, gold_antecedents_clamped.unsqueeze(-1)).squeeze(-1)  # (B, M)

    # Усредняем loss по всем меншионам и батчу
    loss = -gold_log_probs.mean()

    return loss

def get_gold_antecedents(topk_indices, mention_to_cluster):
    """
    topk_indices: List[int] — индексы топ-K меншионов
    mention_to_cluster: List[int] — соответствие всех меншионов кластерам
    Возвращает List[int] — индекс антецедента из topk_indices для каждого меншиона, или -1 если нет
    """
    gold_antecedents = []

    for i, idx_i in enumerate(topk_indices):
        cluster_i = mention_to_cluster[idx_i]
        found = False
        for j in range(i - 1, -1, -1):
            idx_j = topk_indices[j]
            if mention_to_cluster[idx_j] == cluster_i:
                gold_antecedents.append(j)
                found = True
                break
        if not found:
            gold_antecedents.append(-1)

    return gold_antecedents


def visualize_scores_save(mention_scores, pairwise_scores, epoch=None, batch_idx=None, save_dir='plots'):
    import os
    os.makedirs(save_dir, exist_ok=True)

    mention_scores = mention_scores.detach().cpu().numpy()
    pairwise_scores = pairwise_scores.detach().cpu().numpy()
    if torch.isnan(pairwise_scores).any():
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