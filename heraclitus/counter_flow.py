"""The Counter-Flow.

Heraclitus held that opposites are one — the road up and the road down are the
same road. In this implementation, every time a `DualFlowTransformer` updates,
its prior incarnation is frozen as a `CounterFlow`: a derivative module that
holds the previous parameters, the previous direction, and the previous
connection set. Counter-Flows form a backwards-pointing shadow of the live
network.

Counter-Flows are queryable but not trainable. They serve three purposes:
    1. *Audit*: full provenance of what the transformer believed an instant ago.
    2. *Distillation*: the live model can be regularised against the Counter-Flow
       to slow catastrophic drift (see `CounterFlow.distillation_target`).
    3. *Opposition*: by definition the Counter-Flow's direction is the *prior*
       direction; the geodesic between them measures the magnitude of the most
       recent conceptual revision.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from torch import Tensor, nn


@dataclass
class CounterFlowSnapshot:
    """Pure-data record of a single past incarnation."""
    step: int
    direction: Tensor                        # (3,)
    state_dict: Dict[str, Tensor]
    connection_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CounterFlow(nn.Module):
    """A frozen, queryable shadow of a previous DualFlowTransformer state.

    Parameters
    ----------
    parent_id : str
        Stable id of the live transformer this Counter-Flow shadows.
    snapshot : CounterFlowSnapshot
        The frozen state captured before the most recent update.
    parent_module : nn.Module | None
        Optional reference to the live module, used purely to construct an
        identically-shaped frozen copy for distillation queries.
    """

    def __init__(
        self,
        parent_id: str,
        snapshot: CounterFlowSnapshot,
        parent_module: nn.Module | None = None,
    ):
        super().__init__()
        self.parent_id = parent_id
        self.snapshot = snapshot
        self._frozen_module: nn.Module | None = None
        if parent_module is not None:
            self._frozen_module = copy.deepcopy(parent_module)
            self._frozen_module.load_state_dict(snapshot.state_dict)
            for p in self._frozen_module.parameters():
                p.requires_grad_(False)
            self._frozen_module.eval()

    # ------------------------------------------------------------------ API

    @property
    def direction(self) -> Tensor:
        """The S^2 direction this past incarnation was pointing in."""
        return self.snapshot.direction

    @property
    def step(self) -> int:
        return self.snapshot.step

    def distillation_target(self, x: Tensor) -> Tensor | None:
        """Run the frozen prior module on `x`, for use as a soft target.

        Returns None if no parent module was attached at construction time.
        """
        if self._frozen_module is None:
            return None
        with torch.no_grad():
            return self._frozen_module(x, learn=False)

    def opposition(self, live_direction: Tensor) -> Tensor:
        """Geodesic angle (radians) between the live direction and this past one.

        Larger values indicate a sharper recent revision.
        """
        from .direction import Direction
        return Direction.angle(live_direction, self.direction)

    def summary(self) -> str:
        d = self.direction.detach().cpu().tolist()
        return (
            f"<CounterFlow parent={self.parent_id} step={self.step} "
            f"direction=[{d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}] "
            f"connections={len(self.snapshot.connection_ids)}>"
        )

    def __repr__(self) -> str:
        return self.summary()
