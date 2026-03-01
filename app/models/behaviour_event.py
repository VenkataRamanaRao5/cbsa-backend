from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BehaviourEvent:
    username: str
    session_id: str
    vector: np.ndarray
    timestamp: float
    nonce: str
    event_type: str
