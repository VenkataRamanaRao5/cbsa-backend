"""
Cosmos DB Prototype Store

Stores prototype vectors, behaviour logs, and user state in Azure Cosmos DB.
Mirrors the SQLiteStore interface so it can be used as a drop-in replacement.

Container: ``prototype-store``  (partition key: /username)
  - User documents  id = "user:<username>"
  - Prototype docs  id = "proto:<username>:<seqId>"

Container: ``behaviour-logs``  (partition key: /username)
  - One document per logged behaviour event.

Behaviour follows the pattern used by the rest of the codebase:
  - Production  (DEBUG_MODE=False): cloud only.
  - Development (DEBUG_MODE=True):  write to both Cosmos DB and local SQLite.

If Cosmos DB is not configured (endpoint/key absent), the store transparently
falls back to the SQLiteStore so the application still works offline.
"""

from __future__ import annotations

import json
import logging

# Suppress Azure SDK logs
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos._cosmos_http_logging_policy").setLevel(logging.WARNING)
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.models.prototype import Prototype
from app.storage.memory_store import memory_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of azure-cosmos
# ---------------------------------------------------------------------------
try:
    from azure.cosmos import CosmosClient, PartitionKey  # type: ignore[import]

    _COSMOS_SDK_AVAILABLE = True
except ImportError:
    _COSMOS_SDK_AVAILABLE = False
    logger.warning(
        "azure-cosmos package not installed – Cosmos DB prototype store disabled"
    )

WARMUP_WINDOW_COUNT = 20

_DB_PATH = Path(__file__).resolve().parents[2] / "cbsa.db"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _to_json_array(array: np.ndarray) -> str:
    return json.dumps(array.astype(float).tolist(), separators=(",", ":"))


def _from_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float64)


class CosmosPrototypeStore:
    """
    Cosmos-backed equivalent of SQLiteStore.

    In DEBUG_MODE writes to both Cosmos DB and SQLite.
    In production writes to Cosmos DB only.
    If Cosmos DB is unavailable, falls back to SQLite transparently.
    """

    def __init__(self) -> None:
        self._proto_container = None
        self._logs_container = None
        self._enabled = False
        self._sqlite: Optional[object] = None  # SQLiteStore, imported lazily
        self._try_connect()
        self._init_sqlite()

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        if not _COSMOS_SDK_AVAILABLE:
            return

        endpoint = settings.COSMOS_ENDPOINT.strip()
        key = settings.COSMOS_KEY.strip()
        if not endpoint or not key:
            logger.info(
                "COSMOS_ENDPOINT / COSMOS_KEY not set – "
                "Cosmos DB prototype store disabled"
            )
            return

        db_name = settings.COSMOS_DATABASE
        try:
            client = CosmosClient(endpoint, credential=key)
            database = client.create_database_if_not_exists(id=db_name)
            self._proto_container = database.create_container_if_not_exists(
                id=settings.COSMOS_PROTOTYPE_CONTAINER,
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
                "Cosmos DB prototype store connected: database=%s proto_container=%s logs_container=%s",
                db_name,
                settings.COSMOS_PROTOTYPE_CONTAINER,
                settings.COSMOS_BEHAVIOUR_LOGS_CONTAINER,
            )
        except Exception as exc:
            logger.error("Failed to connect Cosmos prototype store: %s", exc)

    def _init_sqlite(self) -> None:
        """Initialise the SQLite fallback / debug store."""
        if settings.DEBUG_MODE or not self._enabled:
            try:
                from app.storage.sqlite_store import SQLiteStore

                self._sqlite = SQLiteStore(_DB_PATH)
            except Exception as exc:
                logger.error("Failed to initialise SQLite fallback: %s", exc)

    # ------------------------------------------------------------------
    # Internal Cosmos helpers
    # ------------------------------------------------------------------

    def _user_doc_id(self, username: str) -> str:
        return f"user:{username}"

    def _proto_doc_id(self, username: str, seq_id: int) -> str:
        return f"proto:{username}:{seq_id}"

    def _get_user_doc(self, username: str) -> Optional[dict]:
        if self._proto_container is None:
            return None
        try:
            return self._proto_container.read_item(
                item=self._user_doc_id(username), partition_key=username
            )
        except Exception:
            return None

    def _upsert_user_doc(self, doc: dict) -> None:
        if self._proto_container is None:
            return
        try:
            self._proto_container.upsert_item(doc)
        except Exception as exc:
            logger.error("Failed to upsert user doc: %s", exc)

    def _next_proto_seq_id(self, username: str) -> int:
        """Return the next sequential prototype ID for a user."""
        doc = self._get_user_doc(username)
        if doc is None:
            return 1
        return int(doc.get("protoCounter", 0)) + 1

    def _increment_proto_counter(self, username: str, new_id: int) -> None:
        doc = self._get_user_doc(username)
        if doc is None:
            doc = {
                "id": self._user_doc_id(username),
                "username": username,
                "type": "user",
                "initialized": 0,
                "createdAt": _utc_now_iso(),
                "protoCounter": new_id,
            }
        else:
            doc["protoCounter"] = max(int(doc.get("protoCounter", 0)), new_id)
        self._upsert_user_doc(doc)

    # ------------------------------------------------------------------
    # SQLiteStore-compatible public interface
    # ------------------------------------------------------------------

    def ensure_user(self, username: str) -> None:
        if self._enabled and self._proto_container is not None:
            if self._get_user_doc(username) is None:
                self._upsert_user_doc(
                    {
                        "id": self._user_doc_id(username),
                        "username": username,
                        "type": "user",
                        "initialized": 0,
                        "createdAt": _utc_now_iso(),
                        "protoCounter": 0,
                    }
                )
        if self._sqlite:
            self._sqlite.ensure_user(username)

    def get_user_initialized(self, username: str) -> bool:
        if self._enabled and self._proto_container is not None:
            doc = self._get_user_doc(username)
            if doc is None:
                self.ensure_user(username)
                return False
            return bool(doc.get("initialized", 0))
        if self._sqlite:
            return self._sqlite.get_user_initialized(username)
        return False

    def set_user_initialized(self, username: str, initialized: bool) -> None:
        if self._enabled and self._proto_container is not None:
            doc = self._get_user_doc(username)
            if doc is None:
                doc = {
                    "id": self._user_doc_id(username),
                    "username": username,
                    "type": "user",
                    "initialized": 0,
                    "createdAt": _utc_now_iso(),
                    "protoCounter": 0,
                }
            doc["initialized"] = 1 if initialized else 0
            self._upsert_user_doc(doc)
        if settings.DEBUG_MODE and self._sqlite:
            self._sqlite.set_user_initialized(username, initialized)

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
        if self._enabled and self._logs_container is not None:
            try:
                self._logs_container.upsert_item(
                    {
                        "id": str(uuid.uuid4()),
                        "username": username,
                        "sessionId": session_id,
                        "eventTimestamp": float(event_timestamp),
                        "eventType": event_type,
                        "vectorJson": _to_json_array(vector),
                        "shortDrift": float(short_drift),
                        "longDrift": float(long_drift),
                        "stabilityScore": float(stability_score),
                        "createdAt": _utc_now_iso(),
                    }
                )
            except Exception as exc:
                logger.error("Failed to insert behaviour log to Cosmos: %s", exc)
        if settings.DEBUG_MODE and self._sqlite:
            self._sqlite.insert_behaviour_log(
                username,
                session_id,
                event_timestamp,
                event_type,
                vector,
                short_drift,
                long_drift,
                stability_score,
            )

    def get_prototypes(self, username: str) -> List[Prototype]:
        if self._enabled and self._proto_container is not None:
            try:
                items = list(
                    self._proto_container.query_items(
                        query=(
                            "SELECT * FROM c WHERE c.username = @u AND c.type = 'prototype' "
                            "ORDER BY c.protoId ASC"
                        ),
                        parameters=[{"name": "@u", "value": username}],
                        partition_key=username,
                    )
                )
                prototypes: List[Prototype] = []
                for item in items:
                    created_at = (
                        datetime.fromisoformat(item["createdAt"])
                        if item.get("createdAt")
                        else datetime.utcnow()
                    )
                    updated_at = (
                        datetime.fromisoformat(item["updatedAt"])
                        if item.get("updatedAt")
                        else created_at
                    )
                    prototypes.append(
                        Prototype(
                            prototype_id=int(item["protoId"]),
                            vector=_from_json_array(item["vectorJson"]),
                            variance=np.maximum(
                                _from_json_array(item["varianceJson"]), 1e-6
                            ),
                            support_count=int(item.get("supportCount", 0)),
                            created_at=created_at,
                            last_updated=updated_at,
                        )
                    )
                return prototypes
            except Exception as exc:
                logger.error("Failed to get prototypes from Cosmos: %s", exc)
        if self._sqlite:
            return self._sqlite.get_prototypes(username)
        return []

    def insert_prototype(
        self,
        username: str,
        vector: np.ndarray,
        variance: np.ndarray,
        support_count: int,
    ) -> int:
        if self._enabled and self._proto_container is not None:
            try:
                seq_id = self._next_proto_seq_id(username)
                now = _utc_now_iso()
                self._proto_container.upsert_item(
                    {
                        "id": self._proto_doc_id(username, seq_id),
                        "username": username,
                        "type": "prototype",
                        "protoId": seq_id,
                        "vectorJson": _to_json_array(vector),
                        "varianceJson": _to_json_array(np.maximum(variance, 1e-6)),
                        "supportCount": int(support_count),
                        "createdAt": now,
                        "updatedAt": now,
                    }
                )
                self._increment_proto_counter(username, seq_id)
                if settings.DEBUG_MODE and self._sqlite:
                    self._sqlite.insert_prototype(username, vector, variance, support_count)
                return seq_id
            except Exception as exc:
                logger.error("Failed to insert prototype to Cosmos: %s", exc)
        if self._sqlite:
            return self._sqlite.insert_prototype(username, vector, variance, support_count)
        return 0

    def update_prototype(self, prototype: Prototype) -> None:
        if self._enabled and self._proto_container is not None:
            # Prototype.prototype_id is the protoId integer; we need to find the doc
            try:
                items = list(
                    self._proto_container.query_items(
                        query=(
                            "SELECT * FROM c WHERE c.protoId = @pid AND c.type = 'prototype'"
                        ),
                        parameters=[{"name": "@pid", "value": prototype.prototype_id}],
                        enable_cross_partition_query=True,
                    )
                )
                for item in items:
                    item["vectorJson"] = _to_json_array(prototype.vector)
                    item["varianceJson"] = _to_json_array(
                        np.maximum(prototype.variance, 1e-6)
                    )
                    item["supportCount"] = int(prototype.support_count)
                    item["updatedAt"] = _utc_now_iso()
                    self._proto_container.upsert_item(item)
            except Exception as exc:
                logger.error("Failed to update prototype in Cosmos: %s", exc)
        if settings.DEBUG_MODE and self._sqlite:
            self._sqlite.update_prototype(prototype)

    def enforce_prototype_limit(self, username: str, limit: int) -> None:
        if self._enabled and self._proto_container is not None:
            try:
                items = list(
                    self._proto_container.query_items(
                        query=(
                            "SELECT c.id, c.protoId, c.supportCount FROM c "
                            "WHERE c.username = @u AND c.type = 'prototype' "
                            "ORDER BY c.supportCount ASC, c.protoId ASC"
                        ),
                        parameters=[{"name": "@u", "value": username}],
                        partition_key=username,
                    )
                )
                if len(items) <= limit:
                    return
                delete_count = len(items) - limit
                for item in items[:delete_count]:
                    try:
                        self._proto_container.delete_item(
                            item=item["id"], partition_key=username
                        )
                    except Exception as exc:
                        logger.debug("Failed to delete prototype %s: %s", item["id"], exc)
            except Exception as exc:
                logger.error("Failed to enforce prototype limit in Cosmos: %s", exc)
        if settings.DEBUG_MODE and self._sqlite:
            self._sqlite.enforce_prototype_limit(username, limit)

    def collect_warmup_window(
        self, username: str, window_vector: np.ndarray
    ) -> Dict[str, int | bool]:
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
            username, mean_vector, np.maximum(variance_vector, 1e-6), support_count=collected
        )
        self.set_user_initialized(username, True)
        memory_store.clear_warmup_buffer(username)

        return {"warmup": True, "collected_windows": collected}

    def export_user(self, username: str) -> Dict[str, object]:
        """Export all data for a user as a serialisable dict."""
        self.ensure_user(username)

        user_doc: Dict = {}
        prototypes: List[Dict] = []
        behaviour_logs: List[Dict] = []

        if self._enabled and self._proto_container is not None:
            try:
                doc = self._get_user_doc(username)
                if doc:
                    user_doc = {
                        "username": username,
                        "initialized": doc.get("initialized", 0),
                        "created_at": doc.get("createdAt"),
                    }
                items = list(
                    self._proto_container.query_items(
                        query="SELECT * FROM c WHERE c.username = @u AND c.type = 'prototype' ORDER BY c.protoId ASC",
                        parameters=[{"name": "@u", "value": username}],
                        partition_key=username,
                    )
                )
                for item in items:
                    prototypes.append(
                        {
                            "id": item["protoId"],
                            "username": username,
                            "vector_json": item["vectorJson"],
                            "variance_json": item["varianceJson"],
                            "support_count": item.get("supportCount", 0),
                            "created_at": item.get("createdAt"),
                            "updated_at": item.get("updatedAt"),
                        }
                    )
            except Exception as exc:
                logger.error("Failed to export user proto data from Cosmos: %s", exc)

        if self._enabled and self._logs_container is not None:
            try:
                items = list(
                    self._logs_container.query_items(
                        query="SELECT * FROM c WHERE c.username = @u ORDER BY c.eventTimestamp ASC",
                        parameters=[{"name": "@u", "value": username}],
                        partition_key=username,
                    )
                )
                for item in items:
                    behaviour_logs.append(
                        {
                            "id": item.get("id"),
                            "username": username,
                            "session_id": item.get("sessionId"),
                            "event_timestamp": item.get("eventTimestamp"),
                            "event_type": item.get("eventType"),
                            "vector_json": item.get("vectorJson"),
                            "short_drift": item.get("shortDrift"),
                            "long_drift": item.get("longDrift"),
                            "stability_score": item.get("stabilityScore"),
                            "created_at": item.get("createdAt"),
                        }
                    )
            except Exception as exc:
                logger.error("Failed to export user behaviour logs from Cosmos: %s", exc)

        if not user_doc and self._sqlite:
            return self._sqlite.export_user(username)

        return {
            "username": username,
            "user": user_doc or {"username": username, "initialized": 0, "created_at": None},
            "prototypes": prototypes,
            "behaviour_logs": behaviour_logs,
        }

    def import_user(self, data: Dict[str, object]) -> None:
        """Import user data (used by the upload-legacy migration endpoint)."""
        username = str(data.get("username", "")).strip()
        if not username:
            raise ValueError("Invalid import payload: username is required")

        self.ensure_user(username)

        incoming_user = data.get("user") if isinstance(data.get("user"), dict) else {}
        incoming_initialized = int(incoming_user.get("initialized", 0))

        if self._enabled and self._proto_container is not None:
            try:
                doc = self._get_user_doc(username)
                if doc is None:
                    doc = {
                        "id": self._user_doc_id(username),
                        "username": username,
                        "type": "user",
                        "initialized": incoming_initialized,
                        "createdAt": (
                            incoming_user.get("created_at") if isinstance(incoming_user, dict) else None
                        ) or _utc_now_iso(),
                        "protoCounter": 0,
                    }
                else:
                    doc["initialized"] = max(int(doc.get("initialized", 0)), incoming_initialized)
                self._upsert_user_doc(doc)

                for prototype in data.get("prototypes", []):
                    if not isinstance(prototype, dict):
                        continue
                    vector_json = prototype.get("vector_json")
                    variance_json = prototype.get("variance_json")
                    if not isinstance(vector_json, str) or not isinstance(variance_json, str):
                        continue
                    seq_id = self._next_proto_seq_id(username)
                    now = _utc_now_iso()
                    self._proto_container.upsert_item(
                        {
                            "id": self._proto_doc_id(username, seq_id),
                            "username": username,
                            "type": "prototype",
                            "protoId": seq_id,
                            "vectorJson": vector_json,
                            "varianceJson": variance_json,
                            "supportCount": int(prototype.get("support_count", 1)),
                            "createdAt": prototype.get("created_at") or now,
                            "updatedAt": prototype.get("updated_at") or prototype.get("created_at") or now,
                        }
                    )
                    self._increment_proto_counter(username, seq_id)

                for log_item in data.get("behaviour_logs", []):
                    if not isinstance(log_item, dict):
                        continue
                    if self._logs_container is not None:
                        self._logs_container.upsert_item(
                            {
                                "id": str(uuid.uuid4()),
                                "username": username,
                                "sessionId": str(log_item.get("session_id", "")),
                                "eventTimestamp": float(log_item.get("event_timestamp", 0.0)),
                                "eventType": str(log_item.get("event_type", "")),
                                "vectorJson": str(log_item.get("vector_json", "[]")),
                                "shortDrift": float(log_item.get("short_drift", 0.0)),
                                "longDrift": float(log_item.get("long_drift", 0.0)),
                                "stabilityScore": float(log_item.get("stability_score", 0.0)),
                                "createdAt": str(log_item.get("created_at") or _utc_now_iso()),
                            }
                        )
            except Exception as exc:
                logger.error("Failed to import user data to Cosmos: %s", exc)

        if settings.DEBUG_MODE and self._sqlite:
            self._sqlite.import_user(data)

    # ------------------------------------------------------------------
    # Deletion helpers (used by admin endpoints)
    # ------------------------------------------------------------------

    def delete_user(self, username: str) -> None:
        """Delete all Cosmos documents for a user."""
        if self._enabled and self._proto_container is not None:
            try:
                items = list(
                    self._proto_container.query_items(
                        query="SELECT c.id FROM c WHERE c.username = @u",
                        parameters=[{"name": "@u", "value": username}],
                        partition_key=username,
                    )
                )
                for item in items:
                    try:
                        self._proto_container.delete_item(
                            item=item["id"], partition_key=username
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to delete Cosmos prototype docs for %s: %s", username, exc)

        if self._enabled and self._logs_container is not None:
            try:
                items = list(
                    self._logs_container.query_items(
                        query="SELECT c.id FROM c WHERE c.username = @u",
                        parameters=[{"name": "@u", "value": username}],
                        partition_key=username,
                    )
                )
                for item in items:
                    try:
                        self._logs_container.delete_item(
                            item=item["id"], partition_key=username
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to delete Cosmos behaviour logs for %s: %s", username, exc)

    def delete_all(self) -> Dict[str, int]:
        """Delete every document across both containers. Returns counts."""
        proto_count = 0
        log_count = 0

        if self._enabled and self._proto_container is not None:
            try:
                items = list(
                    self._proto_container.query_items(
                        query="SELECT c.id, c.username FROM c",
                        enable_cross_partition_query=True,
                    )
                )
                for item in items:
                    try:
                        self._proto_container.delete_item(
                            item=item["id"], partition_key=item["username"]
                        )
                        proto_count += 1
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to truncate Cosmos prototype store: %s", exc)

        if self._enabled and self._logs_container is not None:
            try:
                items = list(
                    self._logs_container.query_items(
                        query="SELECT c.id, c.username FROM c",
                        enable_cross_partition_query=True,
                    )
                )
                for item in items:
                    try:
                        self._logs_container.delete_item(
                            item=item["id"], partition_key=item["username"]
                        )
                        log_count += 1
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to truncate Cosmos behaviour logs: %s", exc)

        return {"prototypes_deleted": proto_count, "logs_deleted": log_count}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
cosmos_prototype_store = CosmosPrototypeStore()
