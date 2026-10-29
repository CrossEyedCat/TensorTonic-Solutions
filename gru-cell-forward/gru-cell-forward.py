import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    Wz = np.array(params["Wz"], dtype=float)
    Wr = np.array(params["Wr"], dtype=float)
    Wh = np.array(params["Wh"], dtype=float)
    
    Uz = np.array(params["Uz"], dtype=float)
    Ur = np.array(params["Ur"], dtype=float)
    Uh = np.array(params["Uh"], dtype=float)
    
    bz = np.array(params["bz"], dtype=float)
    br = np.array(params["br"], dtype=float)
    bh = np.array(params["bh"], dtype=float)
    z = sigmoid(x@Wz + h_prev@Uz + bz)
    r = sigmoid(x@Wr + h_prev@Ur + br)
    h_ = np.tanh(x @ Wh + (r * h_prev) @ Uh + bh)
    h = (1-z) * h_prev + z * h_
    return h