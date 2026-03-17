"""
app.core.config — Re-export from canonical location app.config.

The implementation lives in app/config.py.  New code should import from
app.core.config; old imports from app.config continue to work unchanged.
"""
from app.config import settings, configure_logging  # noqa: F401

__all__ = ["settings", "configure_logging"]
