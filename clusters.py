import torch

def get_predicted_clusters(span_starts, span_ends, antecedent_scores, mention_scores=None, threshold=0.0):
    clusters = []
    mention_to_cluster = {}
    mention_spans = list(zip(span_starts, span_ends))
    M = antecedent_scores.shape[0]

    for i in range(M):
        if mention_scores is not None and mention_scores[i].item() < threshold:
            continue

        scores_i = antecedent_scores[i][:i]
        if len(scores_i) == 0:
            continue

        top_ante_idx = torch.argmax(scores_i).item()
        antecedent_idx = top_ante_idx

        if mention_scores is not None and mention_scores[antecedent_idx].item() < threshold:
            clusters.append([mention_spans[i]])
            continue

        span_i = mention_spans[i]
        span_j = mention_spans[antecedent_idx]

        cluster_i = mention_to_cluster.get(span_i)
        cluster_j = mention_to_cluster.get(span_j)

        if cluster_i is not None and cluster_j is not None:
            if cluster_i is not cluster_j:
                # Объединить два разных кластера
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
            # оба спана еще не в кластерах — создаем новый
            new_cluster = [span_j, span_i]
            clusters.append(new_cluster)
            mention_to_cluster[span_i] = new_cluster
            mention_to_cluster[span_j] = new_cluster

    return clusters
