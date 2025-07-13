import torch
import torch.nn.functional as F


# вроде рабочий лосс но с for лупом внутри
def coref_loss(mention_scores, antecedent_scores, gold_antecedents):
    loss = 0
    batch_size = mention_scores.size(0)

    for b in range(batch_size):
        for i, gold_ante in enumerate(gold_antecedents[b]):
            scores = torch.cat([torch.tensor([0.0], device=antecedent_scores.device), antecedent_scores[b, i]])
            gold_index = 0 if gold_ante == -1 else gold_ante + 1
            loss += -F.log_softmax(scores, dim=0)[gold_index]
    return loss / batch_size

# матричная версия, но ее надо перепроверить и протестировать
# def coref_loss(mention_scores, antecedent_scores, gold_antecedents):
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