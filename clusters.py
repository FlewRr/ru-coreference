import torch

def get_predicted_clusters(span_starts, span_ends, antecedent_scores, mention_scores=None, threshold=0.0):
    clusters = []
    mention_to_cluster = {}
    mention_spans = list(zip(span_starts, span_ends))
    M = len(mention_spans)

    for i in range(M):
        if mention_scores is not None and mention_scores[i].item() < threshold:
            continue

        span_i = mention_spans[i]
        scores_i = antecedent_scores[i][:i]  # valid antecedents

        if scores_i.numel() == 0:
            continue  # нет антецедентов — ничего не делаем

        top_idx = torch.argmax(scores_i).item()
        span_j = mention_spans[top_idx]

        cluster_i = mention_to_cluster.get(span_i)
        cluster_j = mention_to_cluster.get(span_j)

        if cluster_i and cluster_j:
            if cluster_i is not cluster_j:
                # объединяем
                cluster_i.extend(cluster_j)
                clusters.remove(cluster_j)
                for m in cluster_j:
                    mention_to_cluster[m] = cluster_i
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
