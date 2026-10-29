import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return np.sum(np.abs(x-y))