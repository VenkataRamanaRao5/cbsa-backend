from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


VECTOR_SIZE = 48


@dataclass(slots=True)
class SessionState:
    short_window: deque = field(default_factory=lambda: deque(maxlen=5))
    running_mean: np.ndarray = field(default_factory=lambda: np.zeros(VECTOR_SIZE, dtype=np.float64))
    running_variance: np.ndarray = field(default_factory=lambda: np.zeros(VECTOR_SIZE, dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros(VECTOR_SIZE, dtype=np.float64))
    sample_count: int = 0
    event_history: List[np.ndarray] = field(default_factory=list)
    last_timestamp: Optional[float] = None
    seen_nonces: Set[str] = field(default_factory=set)
    fast_delta_count: int = 0


class MemoryStore:
    def __init__(self) -> None:
        self.sessions: Dict[str, SessionState] = {}
        self.warmup_buffers: Dict[str, List[np.ndarray]] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        return self.sessions[session_id]

    def get_or_create_warmup_buffer(self, username: str) -> List[np.ndarray]:
        if username not in self.warmup_buffers:
            self.warmup_buffers[username] = []
        return self.warmup_buffers[username]

    def clear_warmup_buffer(self, username: str) -> None:
        self.warmup_buffers.pop(username, None)


memory_store = MemoryStore()
