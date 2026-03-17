"""
app.core.constants — System-wide numeric constants.

Single source of truth for thresholds and configuration values that are
referenced by multiple layers. Import from here rather than duplicating
magic numbers across files.
"""

import math

# ── Vector dimensions ─────────────────────────────────────────────────────────

VECTOR_DIM: int = 48
SQRT_D: float = math.sqrt(VECTOR_DIM)   # ≈ 6.928

# ── Trust engine thresholds ───────────────────────────────────────────────────

THETA_SAFE: float = 0.65    # trust > this → SAFE
THETA_RISK: float = 0.40    # trust < this → RISK

# ── Decision threshold (for binary accept/reject evaluation) ──────────────────

ACCEPT_THRESHOLD: float = 0.65   # trust_score > this → ACCEPT

# ── Trust EMA parameters ──────────────────────────────────────────────────────

ALPHA_MAX: float = 0.85
ALPHA_MIN: float = 0.30

# ── Raw signal weights (sum = 1.0) ────────────────────────────────────────────

W_SIMILARITY: float = 0.45
W_STABILITY:  float = 0.25
W_DRIFT:      float = 0.30

# ── Composite drift weights (sum = 1.0) ───────────────────────────────────────

W_SHORT_DRIFT: float = 0.60
W_LONG_DRIFT:  float = 0.40

# ── Prototype matching thresholds ─────────────────────────────────────────────

THRESHOLD_UPDATE: float = 0.75   # similarity ≥ this → update prototype
THRESHOLD_CREATE: float = 0.50   # similarity < this → quarantine

# ── Quarantine protocol ───────────────────────────────────────────────────────

N_MIN_OBSERVATIONS:    int   = 3
T_MIN_SPAN_SECONDS:    float = 30.0
CONSISTENCY_THRESHOLD: float = 0.72
T_EXPIRE_SECONDS:      float = 600.0

# ── Session management ────────────────────────────────────────────────────────

SESSION_TTL_SECONDS: float = 600.0
MAX_SESSION_EVENTS:  int   = 10_000   # cap on event_history per session

# ── GAT escalation ────────────────────────────────────────────────────────────

ANOMALY_ESCALATION_THRESHOLD: float = 0.40
N_UNCERTAIN_ESCALATION:       int   = 3
T_RECHECK_SECONDS:            float = 30.0
