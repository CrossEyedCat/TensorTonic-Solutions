def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    topK_rel = len(set(recommended[:k]) & set(relevant))
    return [topK_rel/k, topK_rel/len(relevant)]