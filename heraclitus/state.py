"""Explicit runtime state for the Heraclitus LLM parameter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeraclitusState:
    """Per-sequence live flow, counter-flow, and valid-token count."""

    live: Tensor
    counter: Tensor
    steps: Tensor

    def validate(self, batch_size: int, state_size: int) -> None:
        if self.live.shape != (batch_size, state_size):
            raise ValueError(
                f"live must have shape ({batch_size}, {state_size}), "
                f"got {tuple(self.live.shape)}"
            )
        if self.counter.shape != (batch_size, state_size):
            raise ValueError(
                f"counter must have shape ({batch_size}, {state_size}), "
                f"got {tuple(self.counter.shape)}"
            )
        if self.steps.shape != (batch_size,):
            raise ValueError(
                f"steps must have shape ({batch_size},), got {tuple(self.steps.shape)}"
            )
        if self.steps.dtype != torch.long:
            raise ValueError("steps must use torch.long")
        if self.live.device != self.counter.device or self.live.device != self.steps.device:
            raise ValueError("all state tensors must be on the same device")

    def detach(self) -> "HeraclitusState":
        return HeraclitusState(
            live=self.live.detach(),
            counter=self.counter.detach(),
            steps=self.steps.detach(),
        )

    def clone(self) -> "HeraclitusState":
        return HeraclitusState(
            live=self.live.clone(),
            counter=self.counter.clone(),
            steps=self.steps.clone(),
        )

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "HeraclitusState":
        target_dtype = dtype if dtype is not None else self.live.dtype
        live = self.live.to(device=device, dtype=target_dtype)
        counter = self.counter.to(device=device, dtype=target_dtype)
        steps = self.steps.to(device=live.device)
        return HeraclitusState(live=live, counter=counter, steps=steps)

    def index_select(self, indices: Tensor) -> "HeraclitusState":
        """Reorder or select batch rows for beam search and dynamic batching."""
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 torch.long tensor")
        indices = indices.to(device=self.live.device)
        return HeraclitusState(
            live=self.live.index_select(0, indices),
            counter=self.counter.index_select(0, indices),
            steps=self.steps.index_select(0, indices),
        )

    def as_dict(self) -> Dict[str, Tensor]:
        """Return a tensor-only representation suitable for checkpointing."""
        return {
            "live": self.live,
            "counter": self.counter,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Tensor]) -> "HeraclitusState":
        """Restore state from ``as_dict`` output."""
        required = {"live", "counter", "steps"}
        if set(values) != required:
            raise ValueError(f"state dictionary keys must be {sorted(required)}")
        return cls(
            live=values["live"],
            counter=values["counter"],
            steps=values["steps"],
        )
