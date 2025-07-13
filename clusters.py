# Для будущего инференса (весь код сгенерен гпт, нужно будет перепроверить и поправить)

def get_clusters(span_starts, span_ends, pairwise_scores, threshold=0.5):
    """
    span_starts, span_ends: [num_spans] — индексы токенов спанов
    pairwise_scores: [num_spans, num_spans] — score[i, j] - насколько span_j - антецедент span_i
    threshold: float — минимальный score для связывания

    Возвращает список кластеров, каждый — список (start, end) спанов.
    """
    num_spans = len(span_starts)
    antecedents = [-1] * num_spans  # индекс антецедента для каждого спана, -1 — нет

    for i in range(num_spans):
        # смотрим на антецедентов j < i
        candidates = [(j, pairwise_scores[i, j].item()) for j in range(i)]
        if not candidates:
            continue
        # выбираем лучшего антецедента
        best_j, best_score = max(candidates, key=lambda x: x[1])
        if best_score > threshold:
            antecedents[i] = best_j

    # Строим кластеры — объединяем связанные цепочки
    clusters = []
    cluster_map = {}  # span idx -> cluster idx

    for i in range(num_spans):
        if antecedents[i] == -1:
            # новый кластер
            cluster_idx = len(clusters)
            clusters.append([(span_starts[i], span_ends[i])])
            cluster_map[i] = cluster_idx
        else:
            # добавляем в кластер антецедента
            antecedent_cluster = cluster_map.get(antecedents[i])
            if antecedent_cluster is not None:
                clusters[antecedent_cluster].append((span_starts[i], span_ends[i]))
                cluster_map[i] = antecedent_cluster
            else:
                # если антецедент еще не в кластере, создаем
                cluster_idx = len(clusters)
                clusters.append(
                    [(span_starts[antecedents[i]], span_ends[antecedents[i]]), (span_starts[i], span_ends[i])])
                cluster_map[antecedents[i]] = cluster_idx
                cluster_map[i] = cluster_idx

    return clusters


def spans_to_char_spans(span_starts, span_ends, offsets):
    char_spans = []
    for start, end in zip(span_starts, span_ends):
        # offsets — список (start_char, end_char) для каждого токена
        char_start = offsets[start][0].item()
        char_end = offsets[end][1].item() - 1  # чтобы включить последний символ
        char_spans.append((char_start, char_end))
    return char_spans

def print_clusters(text: str, clusters: List[List[Tuple[int,int]]]):
    """
    Печатает кластеры на основе текста и списка спанов.
    text: исходный текст (строка)
    clusters: список кластеров, где каждый — список (start, end) индексов символов (байтов) или токенов

    Предполагается, что индексы — по символам (если по токенам — надо сделать маппинг)
    """
    print("Кореферентные кластеры:")
    for i, cluster in enumerate(clusters):
        print(f"Кластер {i + 1}:")
        for start, end in cluster:
            # Для демонстрации возьмём кусок текста (если индексы — по символам)
            # Можно подстроить под индексы токенов и использовать offset mapping
            span_text = text[start:end+1]
            print(f"  '{span_text}' [{start}, {end}]")
        print()
