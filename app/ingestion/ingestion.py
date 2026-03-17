"""app.ingestion.ingestion — Re-export from canonical app.engine.ingestion."""
from app.engine.ingestion import validate_and_extract  # noqa: F401

__all__ = ["validate_and_extract"]
