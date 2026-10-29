def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    TP = sum(t == p for t, p in zip(y_true, y_pred))
    FN = sum(t != p for t, p in zip(y_true, y_pred))
    FP = FN
    return 2*TP/(2*TP+ FP + FN)