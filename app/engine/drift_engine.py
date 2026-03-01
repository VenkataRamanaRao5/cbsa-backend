import numpy as np


def l2_norm(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(np.linalg.norm(vector_a - vector_b))


def normalize_to_unit(value: float) -> float:
    if value <= 0:
        return 0.0
    return float(value / (1.0 + value))
