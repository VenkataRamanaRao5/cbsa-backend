from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass(slots=True)
class Prototype:
    prototype_id: int
    vector: np.ndarray
    variance: np.ndarray
    support_count: int
    created_at: datetime
    last_updated: datetime


@dataclass(slots=True)
class PrototypeMetrics:
    similarity_score: float
    short_drift: float
    long_drift: float
    stability_score: float
    matched_prototype_id: Optional[int]
