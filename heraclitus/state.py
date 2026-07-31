"""Explicit runtime state for Heraclitus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeraclitusState:
    """Persistent, batch-isolated memory slots and their usage statistics."""

    memory: Tensor
    usage: Tensor
    steps: Tensor

    @property
    def mean(self) -> Tensor:
        weights = self.usage.softmax(dim=-1).unsqueeze(-1)
        return (weights * self.memory).sum(dim=1)

    @property
    def variance(self) -> Tensor:
        centre = self.mean.unsqueeze(1)
        weights = self.usage.softmax(dim=-1).unsqueeze(-1)
        return (weights * (self.memory - centre).square()).sum(dim=1)

    def validate(self, batch_size: int, memory_slots: int, state_size: int) -> None:
        if self.memory.shape != (batch_size, memory_slots, state_size):
            raise ValueError("memory has an invalid shape")
        if self.usage.shape != (batch_size, memory_slots):
            raise ValueError("usage has an invalid shape")
        if self.steps.shape != (batch_size,) or self.steps.dtype != torch.long:
            raise ValueError("steps must have shape (batch_size,) and dtype torch.long")
        if len({self.memory.device, self.usage.device, self.steps.device}) != 1:
            raise ValueError("all state tensors must share a device")
        if not self.memory.is_floating_point() or not self.usage.is_floating_point():
            raise ValueError("memory and usage must be floating-point tensors")
        if not torch.isfinite(self.memory).all() or not torch.isfinite(self.usage).all():
            raise ValueError("state contains non-finite values")

    def detach(self) -> "HeraclitusState":
        return HeraclitusState(self.memory.detach(), self.usage.detach(), self.steps.detach())

    def clone(self) -> "HeraclitusState":
        return HeraclitusState(self.memory.clone(), self.usage.clone(), self.steps.clone())

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "HeraclitusState":
        memory = self.memory.to(device=device, dtype=dtype or self.memory.dtype)
        return HeraclitusState(
            memory,
            self.usage.to(device=memory.device, dtype=memory.dtype),
            self.steps.to(device=memory.device),
        )

    def index_select(self, indices: Tensor) -> "HeraclitusState":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 torch.long tensor")
        indices = indices.to(self.memory.device)
        return HeraclitusState(
            self.memory.index_select(0, indices),
            self.usage.index_select(0, indices),
            self.steps.index_select(0, indices),
        )

    def as_dict(self) -> Dict[str, Tensor]:
        return {"memory": self.memory, "usage": self.usage, "steps": self.steps}

    @classmethod
    def from_dict(cls, values: Dict[str, Tensor]) -> "HeraclitusState":
        if set(values) != {"memory", "usage", "steps"}:
            raise ValueError("state dictionary must contain memory, usage and steps")
        return cls(**values)
