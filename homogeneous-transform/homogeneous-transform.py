import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    
    T = np.array(T, dtype = float)
    points = np.array(points, dtype = float)
    if points.ndim == 1:
        ph =  np.array([points[0], points[1], points[2], 1]).T;
        p_new = (T @ ph).T
        
        return np.array([p_new[0], p_new[1], p_new[2]])
    else:
        acc = []
        for p in points:
            ph =  np.array([p[0], p[1], p[2], 1]).T;
            p_new = (T @ ph).T
            acc.append([p_new[0], p_new[1], p_new[2]])
        return np.array(acc) 