import numpy as np

def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 127).astype(np.uint8)
    b = (b > 127).astype(np.uint8)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / max(1, float(union))
