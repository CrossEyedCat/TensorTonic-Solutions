import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    def safe_div(num, den):
        return num / den if den != 0 else 0.0

    def class_counts(c):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        return tp, fp, fn

    def prf(tp, fp, fn):
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f = safe_div(2 * p * r, p + r)
        return p, r, f

    accuracy = float((y_true == y_pred).mean())
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        precision, recall, f1 = prf(*class_counts(pos_label))

    elif average == "micro":
        counts = [class_counts(c) for c in classes]
        tp = sum(c[0] for c in counts)
        fp = sum(c[1] for c in counts)
        fn = sum(c[2] for c in counts)
        precision, recall, f1 = prf(tp, fp, fn)

    elif average == "macro":
        scores = [prf(*class_counts(c)) for c in classes]
        precision, recall, f1 = (float(np.mean(s)) for s in zip(*scores))

    elif average == "weighted":
        scores = [prf(*class_counts(c)) for c in classes]
        support = np.array([(y_true == c).sum() for c in classes], dtype=float)
        w = support / support.sum()
        precision, recall, f1 = (float(np.dot(w, s)) for s in zip(*scores))

    else:
        raise ValueError(f"Unknown average: {average}")

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }