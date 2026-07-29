"""Configuration for Heraclitus 1.0."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeraclitusConfig:
    """Validated configuration for the predictive low-rank state adapter."""

    hidden_size: int
    state_size: int = 64
    num_shadows: int = 4
    min_retention: float = 0.50
    max_retention: float = 0.999
    process_noise_floor: float = 1e-4
    observation_noise_floor: float = 1e-3
    initial_variance: float = 1.0
    shadow_scale: float = 0.25
    max_residual_scale: float = 0.10
    projection_norm_bound: float = 1.0
    reconstruction_norm_bound: float = 1.0
    dropout: float = 0.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.state_size < 2:
            raise ValueError("state_size must be at least 2")
        if self.state_size > self.hidden_size:
            raise ValueError("state_size must not exceed hidden_size")
        if self.num_shadows < 2:
            raise ValueError("num_shadows must be at least 2")
        if not 0.0 <= self.min_retention < self.max_retention < 1.0:
            raise ValueError("retention bounds must satisfy 0 <= min < max < 1")
        for name, value in (
            ("process_noise_floor", self.process_noise_floor),
            ("observation_noise_floor", self.observation_noise_floor),
            ("initial_variance", self.initial_variance),
            ("shadow_scale", self.shadow_scale),
            ("max_residual_scale", self.max_residual_scale),
            ("projection_norm_bound", self.projection_norm_bound),
            ("reconstruction_norm_bound", self.reconstruction_norm_bound),
            ("epsilon", self.epsilon),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
