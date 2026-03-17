"""app.prototype.quarantine_manager — Re-export."""
from app.engine.quarantine_manager import (  # noqa: F401
    QuarantineManager, CandidatePrototype, quarantine_manager,
)
__all__ = ["QuarantineManager", "CandidatePrototype", "quarantine_manager"]
