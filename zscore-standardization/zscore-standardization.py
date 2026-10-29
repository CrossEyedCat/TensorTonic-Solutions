import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X = np.array(X, dtype = float)
    mu = np.mean(X, axis=axis, keepdims=True)
    sd = np.std(X, axis=axis, keepdims=True)
    return (X - mu) / np.maximum(sd, eps)