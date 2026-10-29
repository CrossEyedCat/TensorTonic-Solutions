import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    X = np.array(X, dtype = float)
    labels = np.array(labels, dtype = float)
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    same = labels[:, None] == labels[None, :]
    own_dist = np.where(same, D, 0)
    a = own_dist.sum(axis=1) / (same.sum(axis=1) - 1)
    clusters = np.unique(labels)
    M = np.stack([D[:, labels == c].mean(axis=1) for c in clusters], axis=1)
    own_cluster = labels[:, None] == clusters[None, :]
    b = np.where(own_cluster, np.inf, M).min(axis=1)
    s = (b - a) / np.maximum(a, b)
    return float(s.mean())