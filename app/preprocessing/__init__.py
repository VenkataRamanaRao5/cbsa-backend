"""app.preprocessing — Layer-2 preprocessing: buffer management, drift, stability."""
from app.preprocessing.preprocessing import process_event          # noqa: F401
from app.preprocessing.buffer_manager import (                     # noqa: F401
    update_session_buffer, PreUpdateSnapshot,
)
from app.preprocessing.drift_engine import (                       # noqa: F401
    compute_short_drift, compute_long_drift,
    compute_stability_score, compute_behavioural_consistency,
)
from app.preprocessing.preprocessed_behaviour import PreprocessedBehaviour  # noqa: F401

__all__ = [
    "process_event",
    "update_session_buffer", "PreUpdateSnapshot",
    "compute_short_drift", "compute_long_drift",
    "compute_stability_score", "compute_behavioural_consistency",
    "PreprocessedBehaviour",
]
