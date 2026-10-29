import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X = np.array(X, dtype = float)
    mn = np.min(X, axis=axis, keepdims=True)
    mx = np.max(X, axis=axis, keepdims=True)
    rng = np.maximum(mx - mn, eps)
    return (X - mn) / rng