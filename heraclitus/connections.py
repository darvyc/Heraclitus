"""Directed connections between DualFlowTransformers.

A FlowConnection is forged when, after an update, a transformer scans the
shared FlowRegistry and finds peers whose direction lies inside its alignment
cone. The connection is weighted by the cosine similarity of the two
directions at the moment of forging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from .core import DualFlowTransformer


@dataclass
class FlowConnection:
    """A weighted, directed edge from `source_id` to `target_id`.

    Attributes
    ----------
    source_id, target_id : str
        Stable ids of the two transformers.
    weight : float
        Cosine similarity at forge time, ∈ [-1, 1] (typically the threshold..1).
    forged_at_step : int
        The source transformer's update step when the edge was created.
    direction_at_forge : Tensor
        Snapshot (3,) of the source's direction at forge time, for audit.
    """
    source_id: str
    target_id: str
    weight: float
    forged_at_step: int
    direction_at_forge: Tensor = field(default_factory=lambda: torch.zeros(3))

    @property
    def id(self) -> str:
        return f"{self.source_id}->{self.target_id}@{self.forged_at_step}"

    def __repr__(self) -> str:
        return (
            f"<FlowConnection {self.source_id} -> {self.target_id} "
            f"w={self.weight:+.3f} step={self.forged_at_step}>"
        )
