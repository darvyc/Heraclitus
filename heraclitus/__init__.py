"""Public API for Heraclitus."""
from .config import HeraclitusConfig
from .mathematics import (
    bounded_frobenius,
    geodesic_fraction,
    orthogonal_procrustes,
    orthogonality_error,
    safe_unit,
    spherical_ema,
)
from .parameter import HeraclitusDiagnostics, HeraclitusOutput, HeraclitusParameter
from .state import HeraclitusState

__all__ = [
    "HeraclitusConfig",
    "HeraclitusDiagnostics",
    "HeraclitusOutput",
    "HeraclitusParameter",
    "HeraclitusState",
    "bounded_frobenius",
    "geodesic_fraction",
    "orthogonal_procrustes",
    "orthogonality_error",
    "safe_unit",
    "spherical_ema",
]

__version__ = "2.0.0"
