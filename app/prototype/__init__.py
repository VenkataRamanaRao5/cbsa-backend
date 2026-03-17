"""app.prototype — Layer-2 prototype matching, quarantine, and lifecycle."""
from app.prototype.prototype_engine import compute_prototype_metrics   # noqa: F401
from app.prototype.similarity_engine import (                          # noqa: F401
    composite_similarity, cosine_similarity,
    mahalanobis_distance, compute_anomaly_indicator,
    compute_prototype_confidence, compute_prototype_support_strength,
)
from app.prototype.quarantine_manager import quarantine_manager        # noqa: F401
from app.prototype.prototype_models import Prototype, PrototypeMetrics # noqa: F401

__all__ = [
    "compute_prototype_metrics",
    "composite_similarity", "cosine_similarity",
    "mahalanobis_distance", "compute_anomaly_indicator",
    "compute_prototype_confidence", "compute_prototype_support_strength",
    "quarantine_manager",
    "Prototype", "PrototypeMetrics",
]
