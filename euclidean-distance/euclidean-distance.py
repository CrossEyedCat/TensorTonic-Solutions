import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return np.sqrt(np.sum((x-y)**2))