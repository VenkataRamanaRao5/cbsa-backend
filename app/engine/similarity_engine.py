import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0.0:
        return 0.0
    value = float(np.dot(vector_a, vector_b) / denominator)
    return max(0.0, min(1.0, value))


def mahalanobis_distance(sample: np.ndarray, prototype: np.ndarray, variance_vector: np.ndarray) -> float:
    safe_variance = np.maximum(variance_vector, 1e-6)
    delta = sample - prototype
    value = float(np.sqrt(np.sum((delta * delta) / safe_variance)))
    return max(0.0, value)


def normalize_mahalanobis(distance: float) -> float:
    if distance <= 0.0:
        return 0.0
    return float(distance / (1.0 + distance))


def composite_similarity(cosine: float, normalized_mahalanobis: float, stability_score: float) -> float:
    score = 0.5 * cosine + 0.3 * (1.0 - normalized_mahalanobis) + 0.2 * stability_score
    return max(0.0, min(1.0, float(score)))
