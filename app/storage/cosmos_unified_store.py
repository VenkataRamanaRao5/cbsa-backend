"""
CosmosDB Unified Store — Production-Grade Cloud Storage Backend

This is the SINGLE SOURCE OF TRUTH for all persistent data.
SQLite is NOT used anywhere in this module. All persistence goes to Azure CosmosDB.

Containers
----------
  users          (partition key: /id)
    Stores per-user profile, initialization state, and adaptive modeling fields.
    Schema: {id, initialized, created_at, adaptive_sigma, drift_mean, drift_std,
             drift_count, similarity_mean, similarity_std, similarity_count, similarity_m2}

  prototypes     (partition key: /username)
    Stores behavioural prototype vectors and variance for each user.
    Schema: {id, username, proto_id, vector, variance, support_count, created_at, updated_at}

  quarantine_pool (partition key: /username)
    Persists candidate prototypes awaiting promotion. Eliminates in-memory-only quarantine.
    Schema: {id, username, centroid, count, consistency, first_seen, last_seen, expires_at}

  behaviour_logs  (partition key: /username)
    Full structured event log with all Layer-2 and Layer-4 fields.
    Schema: {id, username, session_id, timestamp, event_type, vector,
             short_drift, long_drift, stability, similarity, trust, decision,
             prototype_id, layer3_used, created_at}

Adaptive Modeling
-----------------
Per-user adaptive sigma and similarity thresholds are stored in the users container.
These are updated incrementally using Welford's online algorithm after each event.

  adaptive_sigma: 75th percentile approximation of historical short_drift
    sigma_approx = drift_mean + 0.674 * drift_std  (Gaussian 75th percentile)
    Used in drift_engine for exp-normalization.

  theta_update = similarity_mean - 1.5 * similarity_std
  theta_create = similarity_mean - 3.0 * similarity_std
    Applied in prototype_engine (with fallback to global defaults if count < 30).

Quarantine Persistence
----------------------
Candidates are stored in CosmosDB with incremental consistency updates:
  - Centroid updated via online mean: mu_new = mu + (v - mu) / (n + 1)
  - Consistency updated via online mean: C_new = C + (cos(v,mu) - C) / (n + 1)
  - No individual observation vectors stored (reduces storage cost)
  - Expiry enforced via 'expires_at' field (application-side TTL)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppress Azure SDK noise
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

from app.config import settings
from app.models.prototype import Prototype

logger = logging.getLogger(__name__)

# ── Lazy Azure import ─────────────────────────────────────────────────────────
try:
    from azure.cosmos import CosmosClient, PartitionKey   # type: ignore[import]
    from azure.cosmos.exceptions import CosmosResourceNotFoundError  # type: ignore[import]
    _COSMOS_SDK_AVAILABLE = True
except ImportError:
    _COSMOS_SDK_AVAILABLE = False
    CosmosResourceNotFoundError = Exception
    logger.warning("azure-cosmos not installed — CosmosUnifiedStore disabled")

# ── Quarantine parameters ─────────────────────────────────────────────────────
QUARANTINE_N_MIN: int = 3
QUARANTINE_CONSISTENCY_THRESHOLD: float = 0.72
QUARANTINE_T_MIN_SPAN_SECONDS: float = 30.0
QUARANTINE_EXPIRE_SECONDS: float = 600.0
QUARANTINE_MATCH_THRESHOLD: float = 0.75
MAX_QUARANTINE_PER_USER: int = 20
WARMUP_WINDOW_COUNT: int = 20


def _utc_now() -> datetime:
    return datetime.utcnow()


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _to_list(arr: np.ndarray) -> list:
    return arr.astype(float).tolist()


def _from_list(lst: list) -> np.ndarray:
    return np.asarray(lst, dtype=np.float64)


class CosmosUnifiedStore:
    """
    Production-grade unified CosmosDB store.

    Connects on construction. If CosmosDB credentials are absent, all
    operations silently no-op (returning empty/default values) and log warnings.
    No fallback to SQLite or any local storage.

    All methods are synchronous — called via asyncio.to_thread() in main.py.
    """

    def __init__(self) -> None:
        self._users_container = None
        self._proto_container = None
        self._quarantine_container = None
        self._logs_container = None
        self._enabled = False
        self._try_connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _try_connect(self) -> None:
        if not _COSMOS_SDK_AVAILABLE:
            return
        endpoint = settings.COSMOS_ENDPOINT.strip()
        key = settings.COSMOS_KEY.strip()
        if not endpoint or not key:
            logger.warning(
                "COSMOS_ENDPOINT / COSMOS_KEY not configured — "
                "CosmosUnifiedStore disabled (all writes are no-ops)"
            )
            return
        db_name = settings.COSMOS_DATABASE
        try:
            client = CosmosClient(endpoint, credential=key)
            database = client.create_database_if_not_exists(id=db_name)

            self._users_container = database.create_container_if_not_exists(
                id=settings.COSMOS_USERS_CONTAINER,
                partition_key=PartitionKey(path="/id"),
                offer_throughput=400,
            )
            self._proto_container = database.create_container_if_not_exists(
                id=settings.COSMOS_PROTOTYPE_CONTAINER,
                partition_key=PartitionKey(path="/username"),
                offer_throughput=400,
            )
            self._quarantine_container = database.create_container_if_not_exists(
                id=settings.COSMOS_QUARANTINE_CONTAINER,
                partition_key=PartitionKey(path="/username"),
                offer_throughput=400,
            )
            self._logs_container = database.create_container_if_not_exists(
                id=settings.COSMOS_BEHAVIOUR_LOGS_CONTAINER,
                partition_key=PartitionKey(path="/username"),
                offer_throughput=400,
            )
            self._enabled = True
            logger.info(
                "CosmosUnifiedStore connected: database=%s | containers=[%s, %s, %s, %s]",
                db_name,
                settings.COSMOS_USERS_CONTAINER,
                settings.COSMOS_PROTOTYPE_CONTAINER,
                settings.COSMOS_QUARANTINE_CONTAINER,
                settings.COSMOS_BEHAVIOUR_LOGS_CONTAINER,
            )
        except Exception as exc:
            logger.error("CosmosUnifiedStore connection failed: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_item(self, container, item_id: str, partition_key: str) -> Optional[dict]:
        if container is None:
            return None
        try:
            return container.read_item(item=item_id, partition_key=partition_key)
        except Exception:
            return None

    def _upsert_item(self, container, doc: dict) -> bool:
        if container is None:
            return False
        try:
            container.upsert_item(doc)
            return True
        except Exception as exc:
            logger.error("CosmosDB upsert failed: %s", exc)
            return False

    def _delete_item(self, container, item_id: str, partition_key: str) -> bool:
        if container is None:
            return False
        try:
            container.delete_item(item=item_id, partition_key=partition_key)
            return True
        except Exception as exc:
            logger.debug("CosmosDB delete failed for %s: %s", item_id, exc)
            return False

    def _query(self, container, query: str, params: list, partition_key=None) -> List[dict]:
        if container is None:
            return []
        try:
            kwargs = {"query": query, "parameters": params}
            if partition_key is not None:
                kwargs["partition_key"] = partition_key
            else:
                kwargs["enable_cross_partition_query"] = True
            return list(container.query_items(**kwargs))
        except Exception as exc:
            logger.error("CosmosDB query failed: %s", exc)
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # USERS CONTAINER
    # ══════════════════════════════════════════════════════════════════════════

    def ensure_user(self, username: str) -> None:
        """Create user document if it does not already exist."""
        if not self._enabled:
            return
        existing = self._read_item(self._users_container, username, username)
        if existing is None:
            self._upsert_item(self._users_container, {
                "id": username,
                "username": username,
                "initialized": False,
                "created_at": _utc_now_iso(),
                "adaptive_sigma": settings.DEFAULT_ADAPTIVE_SIGMA,
                "drift_mean": 0.0,
                "drift_std": 0.0,
                "drift_count": 0,
                "drift_m2": 0.0,
                "similarity_mean": 0.0,
                "similarity_std": 0.0,
                "similarity_count": 0,
                "similarity_m2": 0.0,
            })

    def get_user_initialized(self, username: str) -> bool:
        if not self._enabled:
            return False
        self.ensure_user(username)
        doc = self._read_item(self._users_container, username, username)
        return bool(doc.get("initialized", False)) if doc else False

    def set_user_initialized(self, username: str, initialized: bool) -> None:
        if not self._enabled:
            return
        doc = self._read_item(self._users_container, username, username)
        if doc is None:
            self.ensure_user(username)
            doc = self._read_item(self._users_container, username, username)
        if doc:
            doc["initialized"] = initialized
            self._upsert_item(self._users_container, doc)

    def get_user_adaptive_fields(self, username: str) -> dict:
        """
        Return per-user adaptive modeling fields.

        Returns defaults when user does not exist or has insufficient history.
        """
        default = {
            "adaptive_sigma": settings.DEFAULT_ADAPTIVE_SIGMA,
            "similarity_mean": 0.75,
            "similarity_std": 0.10,
            "similarity_count": 0,
            "drift_mean": 0.15,
            "drift_std": 0.05,
            "drift_count": 0,
        }
        if not self._enabled:
            return default
        doc = self._read_item(self._users_container, username, username)
        if doc is None:
            return default
        return {
            "adaptive_sigma": float(doc.get("adaptive_sigma", settings.DEFAULT_ADAPTIVE_SIGMA)),
            "similarity_mean": float(doc.get("similarity_mean", 0.75)),
            "similarity_std": float(doc.get("similarity_std", 0.10)),
            "similarity_count": int(doc.get("similarity_count", 0)),
            "drift_mean": float(doc.get("drift_mean", 0.15)),
            "drift_std": float(doc.get("drift_std", 0.05)),
            "drift_count": int(doc.get("drift_count", 0)),
        }

    def update_user_adaptive_fields(
        self,
        username: str,
        new_similarity: float,
        new_short_drift: float,
    ) -> None:
        """
        Incrementally update per-user adaptive modeling fields.

        Uses Welford's online algorithm for both similarity and drift distributions.
        Called after every processed event.

        Adaptive sigma derivation:
            sigma_approx = drift_mean + 0.674 * drift_std
            This is the 75th percentile approximation for a Gaussian distribution:
            P(X <= mu + 0.674*sigma) ≈ 0.75.
            Using sigma as the 75th percentile means that ~75% of drift values
            are below the normalization scale — a deliberate choice to avoid
            over-normalizing high-drift events.
        """
        if not self._enabled:
            return
        doc = self._read_item(self._users_container, username, username)
        if doc is None:
            self.ensure_user(username)
            doc = self._read_item(self._users_container, username, username)
        if doc is None:
            return

        # ── Welford's update for similarity distribution ──────────────────
        n_sim = int(doc.get("similarity_count", 0)) + 1
        mean_sim = float(doc.get("similarity_mean", 0.0))
        m2_sim = float(doc.get("similarity_m2", 0.0))

        delta1 = new_similarity - mean_sim
        mean_sim += delta1 / n_sim
        delta2 = new_similarity - mean_sim
        m2_sim += delta1 * delta2
        std_sim = float(np.sqrt(m2_sim / max(n_sim - 1, 1))) if n_sim > 1 else 0.10

        # ── Welford's update for drift distribution ───────────────────────
        n_drift = int(doc.get("drift_count", 0)) + 1
        mean_drift = float(doc.get("drift_mean", 0.0))
        m2_drift = float(doc.get("drift_m2", 0.0))

        delta1d = new_short_drift - mean_drift
        mean_drift += delta1d / n_drift
        delta2d = new_short_drift - mean_drift
        m2_drift += delta1d * delta2d
        std_drift = float(np.sqrt(m2_drift / max(n_drift - 1, 1))) if n_drift > 1 else 0.05

        # ── Compute adaptive sigma: 75th percentile approximation ─────────
        adaptive_sigma = mean_drift + 0.674 * std_drift
        adaptive_sigma = max(0.05, float(adaptive_sigma))  # minimum floor

        doc.update({
            "similarity_count": n_sim,
            "similarity_mean": mean_sim,
            "similarity_std": std_sim,
            "similarity_m2": m2_sim,
            "drift_count": n_drift,
            "drift_mean": mean_drift,
            "drift_std": std_drift,
            "drift_m2": m2_drift,
            "adaptive_sigma": adaptive_sigma,
        })
        self._upsert_item(self._users_container, doc)

    # ══════════════════════════════════════════════════════════════════════════
    # PROTOTYPES CONTAINER
    # ══════════════════════════════════════════════════════════════════════════

    def get_prototypes(self, username: str) -> List[Prototype]:
        items = self._query(
            self._proto_container,
            "SELECT * FROM c WHERE c.username = @u ORDER BY c.proto_id ASC",
            [{"name": "@u", "value": username}],
            partition_key=username,
        )
        prototypes: List[Prototype] = []
        for item in items:
            created_at = (
                datetime.fromisoformat(item["created_at"]) if item.get("created_at")
                else _utc_now()
            )
            updated_at = (
                datetime.fromisoformat(item["updated_at"]) if item.get("updated_at")
                else created_at
            )
            prototypes.append(Prototype(
                prototype_id=int(item["proto_id"]),
                vector=_from_list(item["vector"]),
                variance=np.maximum(_from_list(item["variance"]), 1e-8),
                support_count=int(item.get("support_count", 0)),
                created_at=created_at,
                last_updated=updated_at,
            ))
        return prototypes

    def _next_proto_id(self, username: str) -> int:
        """Get next sequential prototype ID by reading the user counter."""
        doc = self._read_item(self._users_container, username, username)
        if doc is None:
            return 1
        return int(doc.get("proto_counter", 0)) + 1

    def _bump_proto_counter(self, username: str, new_id: int) -> None:
        doc = self._read_item(self._users_container, username, username)
        if doc is None:
            self.ensure_user(username)
            doc = self._read_item(self._users_container, username, username)
        if doc:
            doc["proto_counter"] = max(int(doc.get("proto_counter", 0)), new_id)
            self._upsert_item(self._users_container, doc)

    def insert_prototype(
        self,
        username: str,
        vector: np.ndarray,
        variance: np.ndarray,
        support_count: int,
    ) -> int:
        if not self._enabled:
            return 0
        self.ensure_user(username)
        proto_id = self._next_proto_id(username)
        now = _utc_now_iso()
        self._upsert_item(self._proto_container, {
            "id": f"{username}:proto:{proto_id}",
            "username": username,
            "proto_id": proto_id,
            "vector": _to_list(vector),
            "variance": _to_list(np.maximum(variance, 1e-8)),
            "support_count": int(support_count),
            "created_at": now,
            "updated_at": now,
        })
        self._bump_proto_counter(username, proto_id)
        return proto_id

    def update_prototype(self, prototype: Prototype) -> None:
        if not self._enabled:
            return
        # Find the document by proto_id scoped to username
        items = self._query(
            self._proto_container,
            "SELECT * FROM c WHERE c.username = @u AND c.proto_id = @pid",
            [
                {"name": "@u", "value": prototype.username if hasattr(prototype, "username") else ""},
                {"name": "@pid", "value": prototype.prototype_id},
            ],
            partition_key=None,   # cross-partition because we don't have username here
        )
        # Fallback: query without username if attribute missing
        if not items:
            items = self._query(
                self._proto_container,
                "SELECT * FROM c WHERE c.proto_id = @pid",
                [{"name": "@pid", "value": prototype.prototype_id}],
                partition_key=None,
            )
        for item in items:
            item["vector"] = _to_list(prototype.vector)
            item["variance"] = _to_list(np.maximum(prototype.variance, 1e-8))
            item["support_count"] = int(prototype.support_count)
            item["updated_at"] = _utc_now_iso()
            self._upsert_item(self._proto_container, item)

    def delete_prototype(self, proto_id: int, username: str) -> None:
        """Delete a prototype by proto_id and username (partition key)."""
        if not self._enabled:
            return
        doc_id = f"{username}:proto:{proto_id}"
        self._delete_item(self._proto_container, doc_id, username)

    def enforce_prototype_limit(self, username: str, limit: int) -> None:
        """Legacy method — quality-based deletion now done in prototype_engine."""
        pass  # Handled by _enforce_prototype_limit_quality in prototype_engine.py

    # ══════════════════════════════════════════════════════════════════════════
    # QUARANTINE POOL CONTAINER
    # ══════════════════════════════════════════════════════════════════════════

    def submit_quarantine_candidate(
        self,
        username: str,
        vector: np.ndarray,
        current_time: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Submit a behavioral vector to the persisted quarantine pool.

        Matching: assign to candidate with cos similarity >= QUARANTINE_MATCH_THRESHOLD.
        Promotion: when count >= N_MIN, time_span >= T_MIN, consistency >= threshold.

        Consistency is maintained as an online mean of cosine similarities:
            C_new = C_old + (cos(v, centroid) - C_old) / (count + 1)
        This avoids storing individual observation vectors in CosmosDB.

        Returns: (centroid, variance, support_count) on promotion, else None.
        """
        if not self._enabled:
            return None

        import time as _time
        t = current_time if current_time is not None else _time.time()
        now_iso = datetime.utcfromtimestamp(t).isoformat()
        expire_iso = datetime.utcfromtimestamp(t + QUARANTINE_EXPIRE_SECONDS).isoformat()

        # ── 1. Purge expired candidates ────────────────────────────────────
        self._purge_expired_candidates(username, now_iso)

        # ── 2. Load all candidates for this user ──────────────────────────
        candidates = self._query(
            self._quarantine_container,
            "SELECT * FROM c WHERE c.username = @u",
            [{"name": "@u", "value": username}],
            partition_key=username,
        )

        # ── 3. Find best matching candidate ───────────────────────────────
        best_candidate = None
        best_sim = QUARANTINE_MATCH_THRESHOLD - 1e-9

        for c in candidates:
            centroid = _from_list(c["centroid"])
            norm_c = float(np.linalg.norm(centroid))
            norm_v = float(np.linalg.norm(vector))
            if norm_c < 1e-10 or norm_v < 1e-10:
                continue
            sim = float(np.dot(vector, centroid) / (norm_c * norm_v))
            if sim > best_sim:
                best_sim = sim
                best_candidate = c

        # ── 4a. Update matched candidate ──────────────────────────────────
        if best_candidate is not None:
            centroid = _from_list(best_candidate["centroid"])
            n = int(best_candidate.get("count", 1))

            # Online mean update for centroid
            new_centroid = centroid + (vector - centroid) / (n + 1)

            # Online mean update for consistency (cos(v, centroid) against OLD centroid)
            norm_c = float(np.linalg.norm(centroid))
            norm_v = float(np.linalg.norm(vector))
            if norm_c > 1e-10 and norm_v > 1e-10:
                new_cos = float(np.dot(vector, centroid) / (norm_c * norm_v))
            else:
                new_cos = 0.0
            old_consistency = float(best_candidate.get("consistency", 1.0))
            new_consistency = old_consistency + (new_cos - old_consistency) / (n + 1)

            best_candidate["centroid"] = _to_list(new_centroid)
            best_candidate["count"] = n + 1
            best_candidate["consistency"] = float(new_consistency)
            best_candidate["last_seen"] = now_iso
            best_candidate["expires_at"] = expire_iso

            self._upsert_item(self._quarantine_container, best_candidate)

            # ── Check promotion criteria ───────────────────────────────────
            first_seen = datetime.fromisoformat(best_candidate["first_seen"])
            last_seen = datetime.fromisoformat(now_iso)
            time_span_s = (last_seen - first_seen).total_seconds()

            count = n + 1
            promoted = (
                count >= QUARANTINE_N_MIN
                and time_span_s >= QUARANTINE_T_MIN_SPAN_SECONDS
                and new_consistency >= QUARANTINE_CONSISTENCY_THRESHOLD
            )

            if promoted:
                # Remove from quarantine and return promotion payload
                self._delete_item(
                    self._quarantine_container,
                    best_candidate["id"],
                    username,
                )
                final_centroid = new_centroid
                # Variance estimated as: mean squared deviation from centroid
                # Approximation since we don't store individual vectors
                # Use a prior-weighted estimate: blend centroid-based estimate with
                # consistency-derived spread estimate
                approx_variance = np.full(
                    len(final_centroid),
                    max(1e-4, 1.0 - new_consistency),
                    dtype=np.float64,
                )
                return (final_centroid, approx_variance, count)

        # ── 4b. Create new candidate if pool not full ──────────────────────
        elif len(candidates) < MAX_QUARANTINE_PER_USER:
            candidate_id = f"{username}:q:{uuid.uuid4().hex[:12]}"
            self._upsert_item(self._quarantine_container, {
                "id": candidate_id,
                "username": username,
                "centroid": _to_list(vector),
                "count": 1,
                "consistency": 1.0,
                "first_seen": now_iso,
                "last_seen": now_iso,
                "expires_at": expire_iso,
            })

        return None

    def _purge_expired_candidates(self, username: str, now_iso: str) -> None:
        """Delete quarantine candidates past their expires_at timestamp."""
        expired = self._query(
            self._quarantine_container,
            "SELECT c.id FROM c WHERE c.username = @u AND c.expires_at < @now",
            [{"name": "@u", "value": username}, {"name": "@now", "value": now_iso}],
            partition_key=username,
        )
        for item in expired:
            self._delete_item(self._quarantine_container, item["id"], username)

    def get_quarantine_pool_status(self, username: str) -> List[dict]:
        """Return diagnostic info about the user's quarantine pool."""
        candidates = self._query(
            self._quarantine_container,
            "SELECT * FROM c WHERE c.username = @u",
            [{"name": "@u", "value": username}],
            partition_key=username,
        )
        return [
            {
                "id": c["id"],
                "count": c.get("count", 0),
                "consistency": round(float(c.get("consistency", 0.0)), 3),
                "first_seen": c.get("first_seen"),
                "last_seen": c.get("last_seen"),
            }
            for c in candidates
        ]

    def clear_quarantine_user(self, username: str) -> None:
        candidates = self._query(
            self._quarantine_container,
            "SELECT c.id FROM c WHERE c.username = @u",
            [{"name": "@u", "value": username}],
            partition_key=username,
        )
        for item in candidates:
            self._delete_item(self._quarantine_container, item["id"], username)

    # ══════════════════════════════════════════════════════════════════════════
    # BEHAVIOUR LOGS CONTAINER (extended schema)
    # ══════════════════════════════════════════════════════════════════════════

    def log_behaviour_event(
        self,
        username: str,
        session_id: str,
        timestamp: float,
        event_type: str,
        vector: np.ndarray,
        short_drift: float,
        long_drift: float,
        stability: float,
        similarity: float,
        trust: float,
        decision: str,
        prototype_id: Optional[int],
        layer3_used: bool,
    ) -> None:
        """
        Write a fully structured behaviour event log to CosmosDB.
        All Layer-2 and Layer-4 fields are included for post-hoc analysis.
        """
        if not self._enabled:
            return
        self._upsert_item(self._logs_container, {
            "id": str(uuid.uuid4()),
            "username": username,
            "session_id": session_id,
            "timestamp": float(timestamp),
            "event_type": event_type,
            "vector": _to_list(vector),
            "short_drift": float(short_drift),
            "long_drift": float(long_drift),
            "stability": float(stability),
            "similarity": float(similarity),
            "trust": float(trust),
            "decision": decision,
            "prototype_id": prototype_id,
            "layer3_used": bool(layer3_used),
            "created_at": _utc_now_iso(),
        })

    # ── Legacy insert_behaviour_log (backward compatible) ─────────────────
    def insert_behaviour_log(
        self,
        username: str,
        session_id: str,
        event_timestamp: float,
        event_type: str,
        vector: np.ndarray,
        short_drift: float,
        long_drift: float,
        stability_score: float,
    ) -> None:
        """
        Backward-compatible method used by existing callers.
        Delegates to log_behaviour_event with zero-values for Layer-4 fields.
        """
        self.log_behaviour_event(
            username=username,
            session_id=session_id,
            timestamp=event_timestamp,
            event_type=event_type,
            vector=vector,
            short_drift=short_drift,
            long_drift=long_drift,
            stability=stability_score,
            similarity=0.0,
            trust=0.0,
            decision="UNKNOWN",
            prototype_id=None,
            layer3_used=False,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # WARMUP / INITIALIZATION FLOW
    # ══════════════════════════════════════════════════════════════════════════

    def collect_warmup_window(
        self,
        username: str,
        window_vector: np.ndarray,
    ) -> Dict[str, int | bool]:
        """
        Accumulate warm-up windows in-memory (via memory_store) until threshold.

        The warmup buffer is ONLY in-memory — it is pre-enrollment data and is
        not worth persisting. If the server restarts during warmup, the user
        simply starts warmup over. This is an acceptable UX trade-off.
        """
        from app.storage.memory_store import memory_store

        initialized = self.get_user_initialized(username)
        if initialized:
            return {"warmup": False, "collected_windows": WARMUP_WINDOW_COUNT}

        warmup_buffer = memory_store.get_or_create_warmup_buffer(username)
        warmup_buffer.append(window_vector.copy())
        collected = len(warmup_buffer)

        if collected < WARMUP_WINDOW_COUNT:
            return {"warmup": True, "collected_windows": collected}

        buffer_matrix = np.vstack(warmup_buffer)
        mean_vector = np.mean(buffer_matrix, axis=0)
        variance_vector = np.var(buffer_matrix, axis=0)
        self.insert_prototype(
            username, mean_vector, np.maximum(variance_vector, 1e-8),
            support_count=collected,
        )
        self.set_user_initialized(username, True)
        memory_store.clear_warmup_buffer(username)

        return {"warmup": True, "collected_windows": collected}

    # ══════════════════════════════════════════════════════════════════════════
    # ADMIN / DELETION
    # ══════════════════════════════════════════════════════════════════════════

    def delete_user(self, username: str) -> None:
        """Delete all CosmosDB documents for a user across all containers."""
        if not self._enabled:
            return
        # Users container
        self._delete_item(self._users_container, username, username)

        # Prototypes container
        protos = self._query(
            self._proto_container,
            "SELECT c.id FROM c WHERE c.username = @u",
            [{"name": "@u", "value": username}],
            partition_key=username,
        )
        for p in protos:
            self._delete_item(self._proto_container, p["id"], username)

        # Quarantine container
        self.clear_quarantine_user(username)

        # Behaviour logs container
        logs = self._query(
            self._logs_container,
            "SELECT c.id FROM c WHERE c.username = @u",
            [{"name": "@u", "value": username}],
            partition_key=username,
        )
        for l in logs:
            self._delete_item(self._logs_container, l["id"], username)

    def delete_all(self) -> Dict[str, int]:
        """Delete all documents across all containers. For admin use only."""
        counts: Dict[str, int] = {
            "users": 0, "prototypes": 0, "quarantine": 0, "logs": 0
        }
        if not self._enabled:
            return counts

        for container, key in [
            (self._users_container, "users"),
            (self._proto_container, "prototypes"),
            (self._quarantine_container, "quarantine"),
            (self._logs_container, "logs"),
        ]:
            items = self._query(
                container,
                "SELECT c.id, c.username FROM c",
                [],
                partition_key=None,
            )
            for item in items:
                pk = item.get("username") or item.get("id", "")
                if self._delete_item(container, item["id"], pk):
                    counts[key] += 1

        return counts

    # ══════════════════════════════════════════════════════════════════════════
    # HEALTH / STATUS
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def enabled(self) -> bool:
        return self._enabled


# ── Module-level singleton ────────────────────────────────────────────────────
cosmos_prototype_store = CosmosUnifiedStore()
