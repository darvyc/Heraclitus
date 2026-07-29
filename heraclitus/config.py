"""Validated configuration for Heraclitus."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeraclitusConfig:
    """Configuration for the bounded recurrent memory adapter."""

    hidden_size: int
    state_size: int = 128
    memory_slots: int = 8
    num_heads: int = 4
    min_retention: float = 0.90
    max_retention: float = 0.9995
    max_write_rate: float = 0.25
    max_residual_scale: float = 0.10
    usage_decay: float = 0.995
    usage_penalty: float = 0.10
    dropout: float = 0.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 2 <= self.state_size <= self.hidden_size:
            raise ValueError("state_size must lie in [2, hidden_size]")
        if self.memory_slots < 1:
            raise ValueError("memory_slots must be positive")
        if self.num_heads < 1 or self.state_size % self.num_heads:
            raise ValueError("num_heads must divide state_size")
        if not 0.0 <= self.min_retention < self.max_retention <= 1.0:
            raise ValueError("retention bounds must satisfy 0 <= min < max <= 1")
        if not 0.0 < self.max_write_rate <= 1.0:
            raise ValueError("max_write_rate must lie in (0, 1]")
        if not 0.0 < self.max_residual_scale <= 1.0:
            raise ValueError("max_residual_scale must lie in (0, 1]")
        if not 0.0 < self.usage_decay <= 1.0:
            raise ValueError("usage_decay must lie in (0, 1]")
        if self.usage_penalty < 0.0:
            raise ValueError("usage_penalty must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
