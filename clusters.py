import torch


def get_predicted_clusters(
    span_starts: list[int], 
    span_ends: list[int],
    antecedent_scores: torch.Tensor,
    mention_scores: torch.Tensor | None = None,
    threshold: float = 0.0
) -> list[list[int, int]]:
    clusters = []
    mention_to_cluster = {}
    mention_spans = list(zip(span_starts, span_ends))
    M = len(mention_spans)

    for i in range(M):
        if mention_scores is not None and mention_scores[i].item() < threshold:
            continue

        span_i = mention_spans[i]
        valid_antecedents = list(range(i))  # только предыдущие

        if not valid_antecedents:
            clusters.append([span_i])
            mention_to_cluster[span_i] = clusters[-1]
            continue

        scores_i = antecedent_scores[i][:i]

        null_score = torch.tensor([0.0], device=scores_i.device)  # можно и -inf, но 0 удобнее
        all_scores = torch.cat([null_score, scores_i])  # [null, span_0, ..., span_i-1]
        best_idx = torch.argmax(all_scores).item()

        if best_idx == 0:
            # выбран null — создаем одиночный кластер
            clusters.append([span_i])
            mention_to_cluster[span_i] = clusters[-1]
            continue

        antecedent_idx = valid_antecedents[best_idx - 1]
        span_j = mention_spans[antecedent_idx]

        cluster_i = mention_to_cluster.get(span_i)
        cluster_j = mention_to_cluster.get(span_j)

        if cluster_i and cluster_j:
            if cluster_i is not cluster_j:
                cluster_i.extend(cluster_j)
                clusters.remove(cluster_j)
                for span in cluster_j:
                    mention_to_cluster[span] = cluster_i
        
        elif cluster_i:
            cluster_i.append(span_j)
            mention_to_cluster[span_j] = cluster_i
        
        elif cluster_j:
            cluster_j.append(span_i)
            mention_to_cluster[span_i] = cluster_j
        else:
            new_cluster = [span_j, span_i]
            clusters.append(new_cluster)
            mention_to_cluster[span_i] = new_cluster
            mention_to_cluster[span_j] = new_cluster

    return clusters
