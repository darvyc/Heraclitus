"""Public API for the Heraclitus research prototype."""
from .core import DualFlowTransformer
from .counter_flow import CounterFlow
from .network import FlowRegistry
from .connections import FlowConnection
from .direction import Direction
from .mathematics import (
    expected_random_degree,
    orthogonal_procrustes,
    random_alignment_probability,
    threshold_for_expected_random_degree,
)

__all__ = [
    "DualFlowTransformer",
    "CounterFlow",
    "FlowRegistry",
    "FlowConnection",
    "Direction",
    "random_alignment_probability",
    "expected_random_degree",
    "threshold_for_expected_random_degree",
    "orthogonal_procrustes",
]

__version__ = "0.2.0"
