import torch

def get_predicted_clusters(
    span_starts, span_ends, antecedent_scores,
    mention_scores=None, mention_threshold=0.0,
    antecedent_score_threshold: float = 0.0
):
    """
    Построение предсказанных кластеров coreference на основе
    антецедентных скор и (опционально) mention_score'ов.

    Args:
        span_starts: List[int]
        span_ends: List[int]
        antecedent_scores: Tensor of shape (M, M) with -inf for invalid positions
        mention_scores: Optional Tensor of shape (M,)
        mention_threshold: float, threshold to filter weak mentions
        antecedent_score_threshold: float, filter weak coref links

    Returns:
        clusters: List of lists of (start, end) tuples
    """
    clusters = []
    mention_to_cluster = {}

    mention_spans = list(zip(span_starts, span_ends))
    M = len(mention_spans)

    for i in range(M):
        span_i = mention_spans[i]

        # Отфильтровываем слабые mentions
        if mention_scores is not None and mention_scores[i].item() < mention_threshold:
            continue

        scores_i = antecedent_scores[i][:i]  # только для j < i
        if scores_i.numel() == 0:
            continue

        # Находим наилучший антецедент
        top_score, top_idx = scores_i.max(0)

        # Если наилучший score всё ещё слабый — не связываем
        if top_score.item() < antecedent_score_threshold:
            continue

        span_j = mention_spans[top_idx.item()]

        cluster_i = mention_to_cluster.get(span_i)
        cluster_j = mention_to_cluster.get(span_j)

        if cluster_i is not None and cluster_j is not None:
            if cluster_i is not cluster_j:
                # Объединяем два разных кластера
                merged = cluster_i + cluster_j
                clusters.remove(cluster_i)
                clusters.remove(cluster_j)
                clusters.append(merged)
                for span in merged:
                    mention_to_cluster[span] = merged
        elif cluster_i is not None:
            cluster_i.append(span_j)
            mention_to_cluster[span_j] = cluster_i
        elif cluster_j is not None:
            cluster_j.append(span_i)
            mention_to_cluster[span_i] = cluster_j
        else:
            # Оба спана не в кластерах — создаём новый
            new_cluster = [span_j, span_i]
            clusters.append(new_cluster)
            mention_to_cluster[span_i] = new_cluster
            mention_to_cluster[span_j] = new_cluster

    return clusters
