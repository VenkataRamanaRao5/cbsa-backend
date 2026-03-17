"""
PreprocessedBehaviour — Rich Layer-2 Output Model

All fields are bounded and mathematically defined.
No decisions are made here — only behavioural state representation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PreprocessedBehaviour:
    """
    Behavioural state extracted from a single incoming event.

    Fields
    ------
    window_vector : np.ndarray  (D=48)
        Mean of the post-update short window. Used as the representative
        vector for prototype matching. This is the ONLY field that uses
        post-update statistics — it represents the current behavioral state.

    short_drift : float  [0, 1)
        Short-term drift: deviation of current vector from pre-update local
        window mean.
            d_short(t) = 1 - exp(-||v_t - mu_window^{t-1}||_2 / (sqrt(D)*sigma))
        Captures sudden within-session behavioral changes. Leakage-free.

    long_drift : float  [0, 1)
        Long-term drift: deviation of local window mean from global session mean.
            d_long(t) = 1 - exp(-||mu_window^{t-1} - mu_global^{t-1}||_2 / (sqrt(D)*sigma))
        Captures gradual behavioral drift across the session. Leakage-free.

    stability_score : float  (0, 1]
        Exponential variance-ratio stability measure:
            S(t) = exp(-(1/D) * sum_i [Var_short,i / max(Var_global,i, eps)])
        When short variance near 0: S approaches 1 (stable behavior).
        When short variance >> long variance: S approaches 0 (erratic behavior).
        Bounded by construction via the exponential — no clipping required.

    variance_vector : np.ndarray  (D=48)
        Per-dimension running variance of the current session (Welford's sigma^2).
        Used for Mahalanobis distance computation in prototype matching.

    behavioural_consistency : float  [0, 1]
        Mean cosine similarity of recent window vectors to their centroid.
            C(t) = (1/|W|) * sum_{v in W} cos(v, mean(W))
        Measures directional agreement within the recent window.
        Distinct from stability_score: consistency captures directional alignment,
        stability captures amplitude variance.

    sigma_ref : float
        The drift scale parameter sigma used for exp-normalization.
        Logged for diagnostics. Candidate for per-user adaptation in production.
    """
    window_vector: np.ndarray
    short_drift: float
    long_drift: float
    stability_score: float
    variance_vector: np.ndarray
    behavioural_consistency: float
    sigma_ref: float
