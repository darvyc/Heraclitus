"""Configuration for the Heraclitus LLM parameter."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeraclitusConfig:
    """Validated configuration for a state-conditioned low-rank LLM parameter."""

    hidden_size: int
    state_size: int = 16
    state_decay: float = 0.95
    counter_decay: float = 0.995
    max_residual_scale: float = 0.10
    temperature: float = 0.25
    opposition_strength: float = 0.50
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
        for name, value in (
            ("state_decay", self.state_decay),
            ("counter_decay", self.counter_decay),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.max_residual_scale <= 0.0:
            raise ValueError("max_residual_scale must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.opposition_strength <= 1.0:
            raise ValueError("opposition_strength must lie in [0, 1]")
        if self.projection_norm_bound <= 0.0:
            raise ValueError("projection_norm_bound must be positive")
        if self.reconstruction_norm_bound <= 0.0:
            raise ValueError("reconstruction_norm_bound must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
