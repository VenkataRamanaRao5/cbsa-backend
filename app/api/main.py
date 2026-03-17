"""
app.api.main — FastAPI application using the new layered architecture.

This module re-exports the `app` FastAPI instance from the canonical
app/main.py.  It serves as the preferred entry point for new code and
documentation, while the original app/main.py continues to run unchanged.

To start the server using the new architecture path:
    uvicorn app.api.main:app --reload

To start via the original path (backward-compatible):
    uvicorn app.main:app --reload
"""
from app.main import app  # noqa: F401

__all__ = ["app"]
