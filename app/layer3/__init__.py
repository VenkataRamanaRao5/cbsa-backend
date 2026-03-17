"""app.layer3 — Layer-3: Graph Attention Network (GAT) integration.

DO NOT modify GAT internals (gat/ subpackage).
This package manages escalation routing, session windowing, and result handling.
"""
from app.layer3.layer3_manager import Layer3GATManager   # noqa: F401

__all__ = ["Layer3GATManager"]
