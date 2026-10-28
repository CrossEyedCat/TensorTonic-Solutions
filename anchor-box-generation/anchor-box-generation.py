import numpy as np
def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    acc =[]
    stride = image_size/feature_size
    for i in range(feature_size):
        for j in range(feature_size):
            for s in scales:
                for rat in aspect_ratios:
                    cx = (j + 0.5)* stride
                    cy = (i + 0.5)* stride
                    w = s * np.sqrt(rat)
                    h = s / np.sqrt(rat)
                    acc.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return acc