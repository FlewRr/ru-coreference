def get_predicted_clusters(span_starts, span_ends, pairwise_scores, threshold=0.5):
    """
    Построение кластеров кореференции по выходам модели.

    Args:
        span_starts: List[int] — начала спанов
        span_ends: List[int] — концы спанов
        pairwise_scores: torch.Tensor[num_spans, num_spans] — score[i, j] насколько j — антецедент i
        threshold: float — минимальный score для связывания

    Returns:
        clusters: List[List[Tuple[int, int]]] — список кластеров (список токеновых спанов)
    """
    num_spans = len(span_starts)
    antecedents = [-1] * num_spans

    # шаг 1: выбираем лучшего антецедента для каждого спана
    for i in range(num_spans):
        candidates = [(j, pairwise_scores[i, j].item()) for j in range(i)]
        if candidates:
            best_j, best_score = max(candidates, key=lambda x: x[1])
            if best_score > threshold:
                antecedents[i] = best_j

    # шаг 2: собираем кластеры
    clusters = []
    span_to_cluster = {}

    for i in range(num_spans):
        if antecedents[i] == -1:
            # создаем новый кластер
            cluster_idx = len(clusters)
            clusters.append([(span_starts[i], span_ends[i])])
            span_to_cluster[i] = cluster_idx
        else:
            ante = antecedents[i]
            if ante in span_to_cluster:
                cluster_idx = span_to_cluster[ante]
                clusters[cluster_idx].append((span_starts[i], span_ends[i]))
                span_to_cluster[i] = cluster_idx
            else:
                # антецедент ещё не добавлен — создаем кластер с ним и текущим
                cluster_idx = len(clusters)
                clusters.append([
                    (span_starts[ante], span_ends[ante]),
                    (span_starts[i], span_ends[i])
                ])
                span_to_cluster[ante] = cluster_idx
                span_to_cluster[i] = cluster_idx

    return clusters
