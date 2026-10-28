import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is not None:
        L = max_len
    else:
        L = max((len(s) for s in seqs), default=0)

    out = np.full((len(seqs), L), pad_value)

    for i, s in enumerate(seqs):
        s = s[:L]
        out[i, :len(s)] = s

    return out
    