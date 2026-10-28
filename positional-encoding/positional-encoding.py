import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    out = np.zeros((seq_len, d_model))
    for i in range(seq_len):
        for l in range(d_model):
            if l%2==0:
                out[i][l]=np.sin(i/(base**(2*(l // 2)/d_model)))
            else:
                out[i][l]=np.cos(i/(base**(2*(l // 2)/d_model)))
    return out