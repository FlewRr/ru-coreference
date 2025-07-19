def get_predicted_clusters(span_starts, span_ends, pairwise_scores, threshold=0.5):
    import networkx as nx

    num_spans = len(span_starts)
    G = nx.Graph()

    spans = [(span_starts[i], span_ends[i]) for i in range(num_spans)]

    for i in range(num_spans):
        G.add_node(i)

        for j in range(i):  # только предыдущие, как антецеденты
            score = pairwise_scores[i, j].item()
            if score > threshold:
                G.add_edge(i, j)

    clusters = []
    for component in nx.connected_components(G):
        cluster = [spans[i] for i in component]
        clusters.append(sorted(cluster, key=lambda x: x[0]))  # сортировка по позиции
    return clusters
