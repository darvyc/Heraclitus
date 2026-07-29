"""Configuration for Heraclitus 2.0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HeraclitusConfig:
    """Validated configuration for the multimodal predictive-state parameter."""

    hidden_size: int
    state_size: int = 64
    num_modes: int = 4
    covariance_rank: int = 4
    transition_reflections: int = 4
    min_retention: float = 0.50
    max_retention: float = 0.999
    process_noise_floor: float = 1e-4
    observation_noise_floor: float = 1e-3
    initial_variance: float = 1.0
    min_mode_probability: float = 1e-4
    mode_weight_memory: float = 0.95
    max_residual_scale: float = 0.10
    projection_norm_bound: float = 1.0
    reconstruction_norm_bound: float = 1.0
    context_noise_scale: float = 0.25
    covariance_factor_scale: float = 0.05
    dropout: float = 0.0
    epsilon: float = 1e-6
    num_shadows: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_shadows is not None:
            if self.num_modes != 4 and self.num_modes != self.num_shadows:
                raise ValueError("num_modes and num_shadows disagree")
            object.__setattr__(self, "num_modes", self.num_shadows)
        object.__setattr__(self, "num_shadows", self.num_modes)
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 2 <= self.state_size <= self.hidden_size:
            raise ValueError("state_size must lie in [2, hidden_size]")
        if self.num_modes < 2:
            raise ValueError("num_modes must be at least 2")
        if not 0 <= self.covariance_rank <= self.state_size:
            raise ValueError("covariance_rank must lie in [0, state_size]")
        if not 1 <= self.transition_reflections <= self.state_size:
            raise ValueError("transition_reflections must lie in [1, state_size]")
        if not 0.0 <= self.min_retention < self.max_retention < 1.0:
            raise ValueError("retention bounds must satisfy 0 <= min < max < 1")
        if not 0.0 <= self.mode_weight_memory <= 1.0:
            raise ValueError("mode_weight_memory must lie in [0, 1]")
        if not 0.0 <= self.min_mode_probability < 1.0 / self.num_modes:
            raise ValueError("min_mode_probability must lie in [0, 1 / num_modes)")
        for name, value in (
            ("process_noise_floor", self.process_noise_floor),
            ("observation_noise_floor", self.observation_noise_floor),
            ("initial_variance", self.initial_variance),
            ("max_residual_scale", self.max_residual_scale),
            ("projection_norm_bound", self.projection_norm_bound),
            ("reconstruction_norm_bound", self.reconstruction_norm_bound),
            ("context_noise_scale", self.context_noise_scale),
            ("covariance_factor_scale", self.covariance_factor_scale),
            ("epsilon", self.epsilon),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
