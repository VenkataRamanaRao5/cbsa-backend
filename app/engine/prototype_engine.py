from __future__ import annotations

import numpy as np

from app.engine.similarity_engine import (
    composite_similarity,
    cosine_similarity,
    mahalanobis_distance,
    normalize_mahalanobis,
)
from app.models.preprocessed_behaviour import PreprocessedBehaviour
from app.models.prototype import Prototype, PrototypeMetrics
from app.storage.sqlite_store import SQLiteStore


MAX_PROTOTYPES_PER_USER = 15


def _effective_variance(prototype_variance: np.ndarray, current_variance: np.ndarray) -> np.ndarray:
    averaged = (prototype_variance + current_variance) / 2.0
    return np.maximum(averaged, 1e-6)


def _update_prototype(prototype: Prototype, current_vector: np.ndarray, current_variance: np.ndarray) -> Prototype:
    alpha = 1.0 / (prototype.support_count + 1)
    old_vector = prototype.vector.copy()

    updated_vector = (1.0 - alpha) * prototype.vector + alpha * current_vector
    updated_variance = (1.0 - alpha) * prototype.variance + alpha * np.square(current_vector - old_vector)
    updated_variance = np.maximum((updated_variance + current_variance) / 2.0, 1e-6)

    return Prototype(
        prototype_id=prototype.prototype_id,
        vector=updated_vector,
        variance=updated_variance,
        support_count=prototype.support_count + 1,
        created_at=prototype.created_at,
        last_updated=prototype.last_updated,
    )


def compute_prototype_metrics(
    store: SQLiteStore,
    username: str,
    preprocessed: PreprocessedBehaviour,
) -> PrototypeMetrics:
    prototypes = store.get_prototypes(username)

    if not prototypes:
        prototype_id = store.insert_prototype(
            username=username,
            vector=preprocessed.window_vector,
            variance=preprocessed.variance_vector,
            support_count=1,
        )
        return PrototypeMetrics(
            similarity_score=0.0,
            short_drift=preprocessed.short_drift,
            long_drift=preprocessed.long_drift,
            stability_score=preprocessed.stability_score,
            matched_prototype_id=prototype_id,
        )

    best_prototype = None
    best_similarity = -1.0

    for prototype in prototypes:
        effective_variance = _effective_variance(prototype.variance, preprocessed.variance_vector)

        cosine = cosine_similarity(preprocessed.window_vector, prototype.vector)
        mahalanobis = mahalanobis_distance(preprocessed.window_vector, prototype.vector, effective_variance)
        normalized_mahalanobis = normalize_mahalanobis(mahalanobis)

        similarity = composite_similarity(
            cosine=cosine,
            normalized_mahalanobis=normalized_mahalanobis,
            stability_score=preprocessed.stability_score,
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_prototype = prototype

    matched_prototype_id = best_prototype.prototype_id if best_prototype else None

    if best_prototype and best_similarity > 0.8 and preprocessed.short_drift < 0.3:
        updated = _update_prototype(best_prototype, preprocessed.window_vector, preprocessed.variance_vector)
        store.update_prototype(updated)
    elif best_similarity <= 0.8:
        matched_prototype_id = store.insert_prototype(
            username=username,
            vector=preprocessed.window_vector,
            variance=preprocessed.variance_vector,
            support_count=1,
        )
        store.enforce_prototype_limit(username, MAX_PROTOTYPES_PER_USER)

    return PrototypeMetrics(
        similarity_score=max(0.0, min(1.0, float(best_similarity if best_similarity >= 0.0 else 0.0))),
        short_drift=preprocessed.short_drift,
        long_drift=preprocessed.long_drift,
        stability_score=preprocessed.stability_score,
        matched_prototype_id=matched_prototype_id,
    )
