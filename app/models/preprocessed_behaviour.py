from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PreprocessedBehaviour:
    window_vector: np.ndarray
    short_drift: float
    long_drift: float
    stability_score: float
    variance_vector: np.ndarray
