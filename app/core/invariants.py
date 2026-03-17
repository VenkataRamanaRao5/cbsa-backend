"""
app.core.invariants — Re-export from canonical location app.engine.invariants.

New code should import from app.core.invariants.
Old imports from app.engine.invariants continue to work.
"""
from app.engine.invariants import (  # noqa: F401
    VECTOR_DIM,
    InvariantError,
    check_vector,
    check_variance_vector,
    check_scalar_01,
    check_scalar_nonneg,
    check_preprocessed_behaviour,
    check_prototype_metrics,
    check_trust_result,
)

__all__ = [
    "VECTOR_DIM", "InvariantError",
    "check_vector", "check_variance_vector",
    "check_scalar_01", "check_scalar_nonneg",
    "check_preprocessed_behaviour", "check_prototype_metrics",
    "check_trust_result",
]
