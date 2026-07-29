"""Explicit runtime state for Heraclitus 1.0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeraclitusState:
    """Per-sequence posterior mean, variance, shadow weights, and token count."""

    mean: Tensor
    variance: Tensor
    shadow_log_weights: Tensor
    steps: Tensor

    def validate(self, batch_size: int, state_size: int, num_shadows: int) -> None:
        if self.mean.shape != (batch_size, state_size):
            raise ValueError(f"mean must have shape ({batch_size}, {state_size})")
        if self.variance.shape != (batch_size, state_size):
            raise ValueError(f"variance must have shape ({batch_size}, {state_size})")
        if self.shadow_log_weights.shape != (batch_size, num_shadows):
            raise ValueError(
                f"shadow_log_weights must have shape ({batch_size}, {num_shadows})"
            )
        if self.steps.shape != (batch_size,):
            raise ValueError(f"steps must have shape ({batch_size},)")
        if self.steps.dtype != torch.long:
            raise ValueError("steps must use torch.long")
        devices = {
            self.mean.device,
            self.variance.device,
            self.shadow_log_weights.device,
            self.steps.device,
        }
        if len(devices) != 1:
            raise ValueError("all state tensors must be on the same device")
        if torch.any(self.variance <= 0):
            raise ValueError("variance must be strictly positive")

    def detach(self) -> "HeraclitusState":
        return HeraclitusState(
            mean=self.mean.detach(),
            variance=self.variance.detach(),
            shadow_log_weights=self.shadow_log_weights.detach(),
            steps=self.steps.detach(),
        )

    def clone(self) -> "HeraclitusState":
        return HeraclitusState(
            mean=self.mean.clone(),
            variance=self.variance.clone(),
            shadow_log_weights=self.shadow_log_weights.clone(),
            steps=self.steps.clone(),
        )

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "HeraclitusState":
        target_dtype = dtype if dtype is not None else self.mean.dtype
        mean = self.mean.to(device=device, dtype=target_dtype)
        return HeraclitusState(
            mean=mean,
            variance=self.variance.to(device=mean.device, dtype=target_dtype),
            shadow_log_weights=self.shadow_log_weights.to(
                device=mean.device, dtype=target_dtype
            ),
            steps=self.steps.to(device=mean.device),
        )

    def index_select(self, indices: Tensor) -> "HeraclitusState":
        """Reorder or duplicate rows for beam search and dynamic batching."""
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 torch.long tensor")
        indices = indices.to(device=self.mean.device)
        return HeraclitusState(
            mean=self.mean.index_select(0, indices),
            variance=self.variance.index_select(0, indices),
            shadow_log_weights=self.shadow_log_weights.index_select(0, indices),
            steps=self.steps.index_select(0, indices),
        )

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "shadow_log_weights": self.shadow_log_weights,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Tensor]) -> "HeraclitusState":
        required = {"mean", "variance", "shadow_log_weights", "steps"}
        if set(values) != required:
            raise ValueError(f"state dictionary keys must be {sorted(required)}")
        return cls(
            mean=values["mean"],
            variance=values["variance"],
            shadow_log_weights=values["shadow_log_weights"],
            steps=values["steps"],
        )
