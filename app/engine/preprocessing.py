from __future__ import annotations

import numpy as np

from app.engine.buffer_manager import update_session_buffer
from app.engine.drift_engine import l2_norm, normalize_to_unit
from app.models.behaviour_event import BehaviourEvent
from app.models.preprocessed_behaviour import PreprocessedBehaviour


def process_event(event: BehaviourEvent) -> PreprocessedBehaviour:
    session_state = update_session_buffer(event)

    current_vector = event.vector
    short_window_vectors = np.vstack(session_state.short_window)

    short_window_mean = np.mean(short_window_vectors, axis=0)
    long_term_mean = session_state.running_mean
    variance_vector = session_state.running_variance

    short_drift_raw = l2_norm(current_vector, short_window_mean)
    long_drift_raw = l2_norm(short_window_mean, long_term_mean)

    recent_events = session_state.event_history[-5:]
    if len(recent_events) < 2:
        stability_raw = 0.0
    else:
        diffs = [
            float(np.linalg.norm(recent_events[index + 1] - recent_events[index]))
            for index in range(len(recent_events) - 1)
        ]
        stability_raw = float(np.mean(diffs)) if diffs else 0.0

    short_drift = normalize_to_unit(short_drift_raw)
    long_drift = normalize_to_unit(long_drift_raw)
    stability_score = stability_raw

    return PreprocessedBehaviour(
        window_vector=short_window_mean,
        short_drift=short_drift,
        long_drift=long_drift,
        stability_score=stability_score,
        variance_vector=variance_vector,
    )
