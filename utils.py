import torch
import torch.nn.functional as F

def coref_loss(mention_scores, antecedent_scores, gold_antecedents):
    """
    mention_scores: (batch, num_mentions)
    antecedent_scores: (batch, num_mentions, max_antecedents)
    gold_antecedents: (batch, num_mentions) индексы правильных антецедентов (или -1 для null)

    Возвращает средний loss по батчу.
    """
    loss = 0
    batch_size = mention_scores.size(0)

    for b in range(batch_size):
        for i, gold_ante in enumerate(gold_antecedents[b]):
            if gold_ante == -1:
                # Отсутствие антецедента — считаем score null
                continue
            # Считаем softmax по кандидатом
            scores = torch.cat([torch.tensor([0.0]), antecedent_scores[b, i]])  # null + кандидаты
            gold_index = gold_ante + 1  # смещение +1 для null
            loss += -F.log_softmax(scores, dim=0)[gold_index]
    return loss / batch_size

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