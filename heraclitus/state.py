"""Explicit runtime state for Heraclitus 2.0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeraclitusState:
    """Persistent mixture state with mode-specific means and covariance factors."""

    mode_means: Tensor
    mode_variances: Tensor
    covariance_factors: Tensor
    mode_log_weights: Tensor
    steps: Tensor

    @property
    def mean(self) -> Tensor:
        weights = self.mode_log_weights.softmax(dim=-1).unsqueeze(-1)
        return (weights * self.mode_means).sum(dim=1)

    @property
    def variance(self) -> Tensor:
        weights = self.mode_log_weights.softmax(dim=-1).unsqueeze(-1)
        centre = self.mean.unsqueeze(1)
        within = self.mode_variances + self.covariance_factors.square().sum(dim=-1)
        return (weights * (within + (self.mode_means - centre).square())).sum(dim=1)

    @property
    def shadow_log_weights(self) -> Tensor:
        """Compatibility alias for the 1.x state name."""
        return self.mode_log_weights

    def validate(
        self,
        batch_size: int,
        state_size: int,
        num_modes: int,
        covariance_rank: Optional[int] = None,
    ) -> None:
        rank = self.covariance_factors.shape[-1] if covariance_rank is None else covariance_rank
        expected = (batch_size, num_modes, state_size)
        if self.mode_means.shape != expected:
            raise ValueError(f"mode_means must have shape {expected}")
        if self.mode_variances.shape != expected:
            raise ValueError(f"mode_variances must have shape {expected}")
        if self.covariance_factors.shape != (batch_size, num_modes, state_size, rank):
            raise ValueError(
                "covariance_factors must have shape "
                f"({batch_size}, {num_modes}, {state_size}, {rank})"
            )
        if self.mode_log_weights.shape != (batch_size, num_modes):
            raise ValueError(f"mode_log_weights must have shape ({batch_size}, {num_modes})")
        if self.steps.shape != (batch_size,) or self.steps.dtype != torch.long:
            raise ValueError("steps must have shape (batch_size,) and dtype torch.long")
        tensors = (
            self.mode_means,
            self.mode_variances,
            self.covariance_factors,
            self.mode_log_weights,
            self.steps,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all state tensors must be on the same device")
        if torch.any(self.mode_variances <= 0):
            raise ValueError("mode_variances must be strictly positive")
        if not all(torch.isfinite(tensor).all() for tensor in tensors[:-1]):
            raise ValueError("state contains non-finite values")

    def detach(self) -> "HeraclitusState":
        return HeraclitusState(
            self.mode_means.detach(),
            self.mode_variances.detach(),
            self.covariance_factors.detach(),
            self.mode_log_weights.detach(),
            self.steps.detach(),
        )

    def clone(self) -> "HeraclitusState":
        return HeraclitusState(
            self.mode_means.clone(),
            self.mode_variances.clone(),
            self.covariance_factors.clone(),
            self.mode_log_weights.clone(),
            self.steps.clone(),
        )

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "HeraclitusState":
        target_dtype = self.mode_means.dtype if dtype is None else dtype
        means = self.mode_means.to(device=device, dtype=target_dtype)
        return HeraclitusState(
            means,
            self.mode_variances.to(device=means.device, dtype=target_dtype),
            self.covariance_factors.to(device=means.device, dtype=target_dtype),
            self.mode_log_weights.to(device=means.device, dtype=target_dtype),
            self.steps.to(device=means.device),
        )

    def index_select(self, indices: Tensor) -> "HeraclitusState":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 torch.long tensor")
        indices = indices.to(self.mode_means.device)
        return HeraclitusState(
            self.mode_means.index_select(0, indices),
            self.mode_variances.index_select(0, indices),
            self.covariance_factors.index_select(0, indices),
            self.mode_log_weights.index_select(0, indices),
            self.steps.index_select(0, indices),
        )

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "mode_means": self.mode_means,
            "mode_variances": self.mode_variances,
            "covariance_factors": self.covariance_factors,
            "mode_log_weights": self.mode_log_weights,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Tensor]) -> "HeraclitusState":
        required = {
            "mode_means",
            "mode_variances",
            "covariance_factors",
            "mode_log_weights",
            "steps",
        }
        if set(values) != required:
            raise ValueError(f"state dictionary keys must be {sorted(required)}")
        return cls(**values)
