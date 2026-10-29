import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    vals, counts = np.unique(y, return_counts=True)
    p = counts / np.sum(counts)
    return -np.sum(p * np.log2(p))