"""Frozen, queryable snapshots of previous transformer states."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn


@dataclass
class CounterFlowSnapshot:
    step: int
    direction: Tensor
    state_dict: Dict[str, Tensor]
    connection_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CounterFlow(nn.Module):
    def __init__(
        self,
        parent_id: str,
        snapshot: CounterFlowSnapshot,
        frozen_module: Optional[nn.Module] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.parent_id = parent_id
        self.snapshot = snapshot
        if frozen_module is not None and parent_module is not None:
            raise ValueError("provide only one of frozen_module or parent_module")
        if frozen_module is None and parent_module is not None:
            clone_factory = getattr(parent_module, "_frozen_clone", None)
            if clone_factory is None:
                raise TypeError("parent_module must provide _frozen_clone()")
            frozen_module = clone_factory()
        self._frozen_module = frozen_module
        if self._frozen_module is not None:
            for parameter in self._frozen_module.parameters():
                parameter.requires_grad_(False)
            self._frozen_module.eval()

    @property
    def direction(self) -> Tensor:
        return self.snapshot.direction

    @property
    def step(self) -> int:
        return self.snapshot.step

    def distillation_target(self, x: Tensor) -> Optional[Tensor]:
        if self._frozen_module is None:
            return None
        with torch.no_grad():
            return self._frozen_module(x, learn=False)

    def opposition(self, live_direction: Tensor) -> Tensor:
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
