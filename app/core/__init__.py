"""
app.core — Configuration, constants, and runtime invariants.

Re-exports:
  from app.core.config     import settings, configure_logging
  from app.core.invariants import check_*, InvariantError
  from app.core.constants  import (layer weights, thresholds, vector dimensions)
"""
from app.core.config import settings, configure_logging
from app.core.invariants import (
    check_vector,
    check_variance_vector,
    check_scalar_01,
    check_scalar_nonneg,
    check_preprocessed_behaviour,
    check_prototype_metrics,
    check_trust_result,
    InvariantError,
)
from app.core.constants import (
    VECTOR_DIM,
    THETA_SAFE,
    THETA_RISK,
    ACCEPT_THRESHOLD,
)

__all__ = [
    "settings", "configure_logging",
    "check_vector", "check_variance_vector",
    "check_scalar_01", "check_scalar_nonneg",
    "check_preprocessed_behaviour", "check_prototype_metrics",
    "check_trust_result", "InvariantError",
    "VECTOR_DIM", "THETA_SAFE", "THETA_RISK", "ACCEPT_THRESHOLD",
]
