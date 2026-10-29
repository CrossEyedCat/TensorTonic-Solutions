import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype = float)
    p = y_pred[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(p)))