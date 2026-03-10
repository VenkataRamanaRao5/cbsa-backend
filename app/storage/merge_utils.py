from __future__ import annotations

from typing import Dict

from app.storage.sqlite_store import SQLiteStore, DB_PATH


def export_user(username: str) -> Dict[str, object]:
    store = SQLiteStore(DB_PATH)
    return store.export_user(username)


def import_user(data: Dict[str, object]) -> None:
    store = SQLiteStore(DB_PATH)
    store.import_user(data)
