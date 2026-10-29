import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.array(x, dtype = float)
    m = np.max(x, axis=-1, keepdims=True)
    m_sum = np.sum(np.exp(x-m), axis=-1, keepdims=True)
    p = np.exp(x-m)/m_sum
    return p