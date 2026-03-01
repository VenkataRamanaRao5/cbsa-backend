from __future__ import annotations

import numpy as np

from app.models.behaviour_event import BehaviourEvent
from app.storage.memory_store import SessionState, memory_store


MAX_HISTORY = 256


def update_session_buffer(event: BehaviourEvent) -> SessionState:
    session_state = memory_store.get_or_create_session(event.session_id)
    vector = event.vector

    session_state.short_window.append(vector)
    session_state.event_history.append(vector)
    if len(session_state.event_history) > MAX_HISTORY:
        session_state.event_history.pop(0)

    session_state.sample_count += 1
    sample_count = session_state.sample_count

    if sample_count == 1:
        session_state.running_mean = vector.copy()
        session_state.m2 = np.zeros_like(vector)
        session_state.running_variance = np.zeros_like(vector)
    else:
        delta = vector - session_state.running_mean
        session_state.running_mean = session_state.running_mean + (delta / sample_count)
        delta2 = vector - session_state.running_mean
        session_state.m2 = session_state.m2 + (delta * delta2)
        session_state.running_variance = session_state.m2 / max(sample_count - 1, 1)

    session_state.last_timestamp = event.timestamp
    return session_state
