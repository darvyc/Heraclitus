"""Heraclitus — a 3D Dual-Flow Transformer.

Public API:
    DualFlowTransformer  — the main module
    CounterFlow          — frozen shadow of a previous incarnation
    FlowRegistry         — global directory of live transformers
    FlowConnection       — directed edge between two transformers
    Direction            — utilities for 3D unit vectors on S^2
"""
from .core import DualFlowTransformer
from .counter_flow import CounterFlow
from .network import FlowRegistry
from .connections import FlowConnection
from .direction import Direction

__all__ = [
    "DualFlowTransformer",
    "CounterFlow",
    "FlowRegistry",
    "FlowConnection",
    "Direction",
]

__version__ = "0.1.0"
