"""app.prototype.similarity_engine — Re-export."""
from app.engine.similarity_engine import (  # noqa: F401
    composite_similarity, cosine_similarity, mahalanobis_distance,
    compute_anomaly_indicator, compute_prototype_confidence,
    compute_prototype_support_strength,
)
__all__ = [
    "composite_similarity", "cosine_similarity", "mahalanobis_distance",
    "compute_anomaly_indicator", "compute_prototype_confidence",
    "compute_prototype_support_strength",
]
