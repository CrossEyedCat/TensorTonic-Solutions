import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.array(x, dtype=float)

    if rng is not None:
        r = rng.random(x.shape)
    else:
        r = np.random.random(x.shape)

    if p >= 1.0:
        pattern = np.zeros_like(x)
    else:
        keep = r >= p
        pattern = keep / (1 - p)

    output = x * pattern
    return output, pattern