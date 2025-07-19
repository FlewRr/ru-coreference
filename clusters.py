import torch

def get_predicted_clusters(span_starts, span_ends, antecedent_scores, mention_scores=None, threshold=0.0):
    """
    Строит предсказанные кластеры на основе span-антецедентных скор и optionally mention_scores.

    Args:
        span_starts: List[int]
        span_ends: List[int]
        antecedent_scores: Tensor (M, K)
        mention_scores: Optional Tensor (M,)
        threshold: float — порог уверенности для использования меншиона

    Returns:
        List[List[Tuple[int, int]]]: кластеры, как списки спанов
    """
    clusters = []
    mention_to_cluster = {}
    mention_spans = list(zip(span_starts, span_ends))

    M, K = antecedent_scores.shape

    for i in range(M):
        # optionally skip invalid mentions
        if mention_scores is not None and mention_scores[i].item() < threshold:
            continue

        # ищем максимальный скор среди антецедентов
        scores_i = antecedent_scores[i][:i]
        if len(scores_i) == 0:
            continue

        top_ante_idx = torch.argmax(scores_i).item()
        antecedent_idx = top_ante_idx

        if mention_scores is not None and mention_scores[antecedent_idx].item() < threshold:
            # антецедент слишком "неуверенный"
            clusters.append([mention_spans[i]])
            continue

        span_i = mention_spans[i]
        span_j = mention_spans[antecedent_idx]

        # ищем, где находится span_j
        if span_j in mention_to_cluster:
            cluster = mention_to_cluster[span_j]
            cluster.append(span_i)
            mention_to_cluster[span_i] = cluster
        else:
            # новый кластер
            cluster = [span_j, span_i]
            clusters.append(cluster)
            mention_to_cluster[span_j] = cluster
            mention_to_cluster[span_i] = cluster

    return clusters
