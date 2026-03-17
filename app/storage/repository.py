"""
app.storage.repository — BehaviourRepository: single access point for all storage.

All engine layers (preprocessing, prototype, trust, logging) should call
through this repository instead of directly instantiating or importing
store objects. This:

  1. Decouples engines from concrete storage implementations.
  2. Makes the storage backend swappable without touching engine code.
  3. Provides a single place to add caching, retry logic, or telemetry.

Usage:
    from app.storage.repository import repository

    protos = repository.get_prototypes(username)
    repository.update_prototype(proto)
    repository.log_event(username, session_id, ...)

The singleton `repository` wraps `cosmos_prototype_store` (which itself
delegates to `cosmos_unified_store` for production).  The underlying store
can be replaced for testing by assigning `repository._store = MockStore()`.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from app.models.prototype import Prototype, PrototypeMetrics

logger = logging.getLogger(__name__)


class BehaviourRepository:
    """
    Repository facade over the concrete Cosmos storage backend.

    All methods map 1-to-1 to CosmosUnifiedStore / CosmosPrototypeStore methods
    but add:
      - Uniform error logging (never propagate store failures to engines)
      - A single injectable _store for test isolation
      - Typed signatures that document the contract

    Engine code should only depend on this class, never on concrete stores.
    """

    def __init__(self) -> None:
        self._store = None   # lazy-loaded to avoid circular imports at startup

    # ── Store access ──────────────────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
            from app.storage.cosmos_prototype_store import cosmos_prototype_store
            self._store = cosmos_prototype_store
        return self._store

    # ── User management ───────────────────────────────────────────────────────

    def get_user(self, username: str) -> Optional[dict]:
        """Return user document dict or None if not found."""
        try:
            store = self._get_store()
            if hasattr(store, "get_user_adaptive_fields"):
                return store.get_user_adaptive_fields(username)
            return None
        except Exception as exc:
            logger.error("repository.get_user(%s): %s", username, exc)
            return None

    def update_user(self, username: str, **fields) -> None:
        """Upsert arbitrary user-level fields (e.g., adaptive sigma)."""
        try:
            store = self._get_store()
            if hasattr(store, "ensure_user"):
                store.ensure_user(username)
        except Exception as exc:
            logger.error("repository.update_user(%s): %s", username, exc)

    def get_user_initialized(self, username: str) -> bool:
        try:
            return self._get_store().get_user_initialized(username)
        except Exception as exc:
            logger.error("repository.get_user_initialized(%s): %s", username, exc)
            return False

    def set_user_initialized(self, username: str, initialized: bool) -> None:
        try:
            self._get_store().set_user_initialized(username, initialized)
        except Exception as exc:
            logger.error("repository.set_user_initialized(%s): %s", username, exc)

    # ── Prototype CRUD ────────────────────────────────────────────────────────

    def get_prototypes(self, username: str) -> List[Prototype]:
        """Return all prototypes for the user; empty list on failure."""
        try:
            return self._get_store().get_prototypes(username)
        except Exception as exc:
            logger.error("repository.get_prototypes(%s): %s", username, exc)
            return []

    def insert_prototype(
        self,
        username: str,
        vector: np.ndarray,
        variance: np.ndarray,
        support_count: int,
    ) -> int:
        """Insert a new prototype; return its id (0 on failure)."""
        try:
            return self._get_store().insert_prototype(
                username, vector, variance, support_count
            )
        except Exception as exc:
            logger.error("repository.insert_prototype(%s): %s", username, exc)
            return 0

    def update_prototype(self, prototype: Prototype) -> None:
        try:
            self._get_store().update_prototype(prototype)
        except Exception as exc:
            logger.error("repository.update_prototype(%s): %s", prototype.prototype_id, exc)

    def delete_prototype(self, proto_id: int, username: str) -> None:
        try:
            store = self._get_store()
            try:
                store.delete_prototype(proto_id, username)
            except TypeError:
                store.delete_prototype(proto_id)
        except Exception as exc:
            logger.error("repository.delete_prototype(%s): %s", proto_id, exc)

    # ── Quarantine ────────────────────────────────────────────────────────────

    def submit_quarantine_candidate(
        self,
        username: str,
        vector: np.ndarray,
        current_time: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Submit vector to quarantine.  Returns (centroid, variance, support) on
        promotion, or None if candidate is not yet ready.
        """
        try:
            store = self._get_store()
            if hasattr(store, "submit_quarantine_candidate"):
                return store.submit_quarantine_candidate(username, vector, current_time)
            # Fallback: in-memory quarantine manager
            from app.engine.quarantine_manager import quarantine_manager
            return quarantine_manager.submit(username, vector, current_time)
        except Exception as exc:
            logger.error("repository.submit_quarantine_candidate(%s): %s", username, exc)
            return None

    # ── Adaptive fields ───────────────────────────────────────────────────────

    def get_user_adaptive_fields(self, username: str) -> Optional[dict]:
        try:
            store = self._get_store()
            if hasattr(store, "get_user_adaptive_fields"):
                return store.get_user_adaptive_fields(username)
            return None
        except Exception as exc:
            logger.error("repository.get_user_adaptive_fields(%s): %s", username, exc)
            return None

    def update_user_adaptive_fields(
        self,
        username: str,
        similarity: float,
        drift: float,
    ) -> None:
        try:
            store = self._get_store()
            if hasattr(store, "update_user_adaptive_fields"):
                store.update_user_adaptive_fields(username, similarity, drift)
        except Exception as exc:
            logger.error("repository.update_user_adaptive_fields(%s): %s", username, exc)

    # ── Event logging ─────────────────────────────────────────────────────────

    def log_event(
        self,
        username: str,
        session_id: str,
        event_timestamp: float,
        event_type: str,
        proto_metrics: PrototypeMetrics,
        trust_result,
    ) -> None:
        """
        Write a structured event log record.

        Delegates to structured_logger which in turn calls
        store.log_behaviour_event().  Errors are caught and logged —
        the logging path must never interrupt pipeline execution.
        """
        try:
            from app.engine.structured_logger import structured_logger
            structured_logger.log(
                username=username,
                session_id=session_id,
                event_timestamp=event_timestamp,
                event_type=event_type,
                proto_metrics=proto_metrics,
                trust_result=trust_result,
            )
        except Exception as exc:
            logger.error("repository.log_event(%s): %s", username, exc)

    def log_behaviour_event_raw(
        self,
        username: str,
        session_id: str,
        event_timestamp: float,
        event_type: str,
        **kwargs,
    ) -> None:
        """Raw passthrough to store.insert_behaviour_log() for compatibility."""
        try:
            store = self._get_store()
            if hasattr(store, "insert_behaviour_log"):
                store.insert_behaviour_log(
                    username, session_id, event_timestamp, event_type, **kwargs
                )
        except Exception as exc:
            logger.error("repository.log_behaviour_event_raw(%s): %s", username, exc)

    # ── Warmup ────────────────────────────────────────────────────────────────

    def collect_warmup_window(self, username: str, window_vector: np.ndarray) -> dict:
        try:
            return self._get_store().collect_warmup_window(username, window_vector)
        except Exception as exc:
            logger.error("repository.collect_warmup_window(%s): %s", username, exc)
            return {"warmup": False, "collected_windows": 0}

    # ── Prototype analytics ───────────────────────────────────────────────────

    def get_prototype_stats(self, username: str) -> dict:
        """
        Return diagnostic statistics for a user's prototype set.

        Returns:
            {
                "total": int,
                "support_distribution": list[int],
                "min_support": int,
                "max_support": int,
                "avg_support": float,
            }
        """
        protos = self.get_prototypes(username)
        if not protos:
            return {"total": 0, "support_distribution": [], "min_support": 0,
                    "max_support": 0, "avg_support": 0.0}
        supports = [p.support_count for p in protos]
        return {
            "total": len(protos),
            "support_distribution": sorted(supports, reverse=True),
            "min_support": min(supports),
            "max_support": max(supports),
            "avg_support": sum(supports) / len(supports),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

repository: BehaviourRepository = BehaviourRepository()
