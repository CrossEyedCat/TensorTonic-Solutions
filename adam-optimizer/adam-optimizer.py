import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    param = np.array(param, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad**2
    m_div = m/(1-beta1**t)
    v_div = v/(1-beta2**t)    
    param=param - lr * m_div/(np.sqrt(v_div) + eps)
    return (param, m, v)