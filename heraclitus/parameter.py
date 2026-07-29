"""A causal, state-conditioned, low-rank parameter for transformer language models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .config import HeraclitusConfig
from .mathematics import (
    bounded_frobenius,
    geodesic_fraction,
    orthogonality_error,
    safe_unit,
    spherical_ema,
)
from .state import HeraclitusState


@dataclass(frozen=True)
class HeraclitusDiagnostics:
    gate_mean: Tensor
    state_drift: Tensor
    opposition: Tensor
    residual_ratio: Tensor


@dataclass(frozen=True)
class HeraclitusOutput:
    hidden_states: Tensor
    state: HeraclitusState
    regularization: Dict[str, Tensor]
    diagnostics: HeraclitusDiagnostics

    def regularization_loss(
        self,
        orthogonality_weight: float = 1e-3,
        counter_weight: float = 1e-3,
        drift_weight: float = 1e-4,
        residual_weight: float = 1e-4,
    ) -> Tensor:
        """Return the weighted auxiliary objective for training."""
        return (
            orthogonality_weight * self.regularization["orthogonality"]
            + counter_weight * self.regularization["counter_consistency"]
            + drift_weight * self.regularization["state_drift"]
            + residual_weight * self.regularization["residual_energy"]
        )


class HeraclitusParameter(nn.Module):
    """Insert a dual-flow low-rank parameter into an LLM residual stream.

    The module is causal: token t is modulated only by state accumulated from
    valid tokens before t. Runtime state is explicit and no trainable parameter
    is mutated during forward execution.
    """

    def __init__(self, config: HeraclitusConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        state_size = config.state_size

        projection = torch.empty(hidden_size, state_size)
        nn.init.orthogonal_(projection)
        self.projection = nn.Parameter(projection)

        reconstruction = torch.empty(state_size, hidden_size)
        nn.init.normal_(reconstruction, mean=0.0, std=1e-3 / state_size**0.5)
        self.reconstruction = nn.Parameter(reconstruction)

        self.state_seed = nn.Parameter(torch.randn(state_size))
        self.gate_bias = nn.Parameter(torch.zeros(()))
        self.residual_scale_logit = nn.Parameter(torch.zeros(()))
        self.opposition_logit = nn.Parameter(
            torch.logit(torch.tensor(config.opposition_strength).clamp(1e-4, 1.0 - 1e-4))
        )
        self.dropout = nn.Dropout(config.dropout)

    def effective_projection(self) -> Tensor:
        """Return the norm-bounded projection used in computation."""
        return bounded_frobenius(
            self.projection.float(),
            self.config.projection_norm_bound,
            self.config.epsilon,
        )

    def effective_reconstruction(self) -> Tensor:
        """Return the norm-bounded reconstruction used in computation."""
        return bounded_frobenius(
            self.reconstruction.float(),
            self.config.reconstruction_norm_bound,
            self.config.epsilon,
        )

    def residual_bound(self, training: Optional[bool] = None) -> float:
        """Return the per-token L2 residual bound from the mathematical contract."""
        use_training = self.training if training is None else training
        dropout_factor = 1.0
        if use_training and self.config.dropout > 0.0:
            dropout_factor = 1.0 / (1.0 - self.config.dropout)
        return (
            2.0
            * self.config.max_residual_scale
            * self.config.projection_norm_bound
            * self.config.reconstruction_norm_bound
            * self.config.hidden_size**0.5
            * dropout_factor
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.config.hidden_size}, "
            f"state_size={self.config.state_size}, "
            f"max_residual_scale={self.config.max_residual_scale}"
        )

    def initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> HeraclitusState:
        """Create a batch-isolated initial runtime state."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        target_device = device if device is not None else self.state_seed.device
        seed = safe_unit(self.state_seed.float(), epsilon=self.config.epsilon)
        seed = seed.to(target_device).unsqueeze(0).expand(batch_size, -1).clone()
        steps = torch.zeros(batch_size, dtype=torch.long, device=target_device)
        return HeraclitusState(live=seed, counter=seed.clone(), steps=steps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the parameter from its learned initial state."""
        return self.forward_with_state(
            hidden_states=hidden_states,
            state=None,
            attention_mask=attention_mask,
        ).hidden_states

    def forward_with_state(
        self,
        hidden_states: Tensor,
        state: Optional[HeraclitusState] = None,
        attention_mask: Optional[Tensor] = None,
        detach_state: bool = True,
    ) -> HeraclitusOutput:
        """Apply the parameter and return the continuation state."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, hidden)")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {hidden_size}"
            )
        if sequence_length < 1:
            raise ValueError("sequence length must be positive")

        mask = self._normalise_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=hidden_states.device,
        )
        runtime_state = state or self.initial_state(batch_size, hidden_states.device)
        runtime_state.validate(batch_size, self.config.state_size)
        if runtime_state.live.device != hidden_states.device:
            runtime_state = runtime_state.to(device=hidden_states.device)

        x = hidden_states.float()
        root_mean_square = x.square().mean(dim=-1, keepdim=True).add(
            self.config.epsilon
        ).sqrt()
        normalised = x / root_mean_square

        projection = self.effective_projection()
        reconstruction = self.effective_reconstruction()
        latent = normalised @ projection
        observations = safe_unit(latent, epsilon=self.config.epsilon)

        live = safe_unit(runtime_state.live.float(), epsilon=self.config.epsilon)
        counter = safe_unit(runtime_state.counter.float(), epsilon=self.config.epsilon)
        steps = runtime_state.steps

        residual_scale = self.config.max_residual_scale * torch.sigmoid(
            self.residual_scale_logit.float()
        )
        opposition_strength = torch.sigmoid(self.opposition_logit.float())

        deltas = []
        gates = []
        drifts = []
        oppositions = []

        for token_index in range(sequence_length):
            valid = mask[:, token_index].unsqueeze(-1)
            dual_flow = safe_unit(
                (1.0 + opposition_strength) * live
                - opposition_strength * counter,
                epsilon=self.config.epsilon,
                fallback=live,
            )

            token_latent = latent[:, token_index, :]
            token_direction = observations[:, token_index, :]
            score = (token_direction * dual_flow).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid(
                score / self.config.temperature + self.gate_bias.float()
            )

            aligned = (token_latent * dual_flow).sum(dim=-1, keepdim=True) * dual_flow
            adapted_latent = token_latent + opposition_strength * aligned
            token_delta = residual_scale * gate * (adapted_latent @ reconstruction)
            token_delta = self.dropout(token_delta)
            token_delta = torch.where(valid, token_delta, torch.zeros_like(token_delta))
            deltas.append(token_delta)
            gates.append(torch.where(valid, gate, torch.zeros_like(gate)))

            next_counter = spherical_ema(
                counter,
                live,
                decay=self.config.counter_decay,
                epsilon=self.config.epsilon,
            )
            next_live = spherical_ema(
                live,
                token_direction,
                decay=self.config.state_decay,
                epsilon=self.config.epsilon,
            )
            drifts.append(
                torch.where(
                    valid.squeeze(-1),
                    geodesic_fraction(live, next_live, self.config.epsilon),
                    torch.zeros(batch_size, device=hidden_states.device),
                )
            )
            oppositions.append(
                torch.where(
                    valid.squeeze(-1),
                    geodesic_fraction(live, counter, self.config.epsilon),
                    torch.zeros(batch_size, device=hidden_states.device),
                )
            )
            live = torch.where(valid, next_live, live)
            counter = torch.where(valid, next_counter, counter)
            steps = steps + valid.squeeze(-1).to(torch.long)

        delta = torch.stack(deltas, dim=1)
        output = hidden_states + delta.to(dtype=hidden_states.dtype)
        valid_count = mask.sum().clamp_min(1).to(dtype=torch.float32)
        gate_mean = torch.stack(gates, dim=1).sum() / valid_count
        state_drift = torch.stack(drifts, dim=1).sum() / valid_count
        opposition = torch.stack(oppositions, dim=1).sum() / valid_count
        residual_energy = delta.square().sum() / x.square().sum().clamp_min(
            self.config.epsilon
        )
        residual_ratio = delta.norm() / x.norm().clamp_min(self.config.epsilon)

        next_state = HeraclitusState(live=live, counter=counter, steps=steps)
        if detach_state:
            next_state = next_state.detach()

        regularization = {
            "orthogonality": orthogonality_error(
                self.projection.float(), self.config.epsilon
            ),
            "counter_consistency": opposition,
            "state_drift": state_drift,
            "residual_energy": residual_energy,
        }
        diagnostics = HeraclitusDiagnostics(
            gate_mean=gate_mean,
            state_drift=state_drift,
            opposition=opposition,
            residual_ratio=residual_ratio,
        )
        return HeraclitusOutput(
            hidden_states=output,
            state=next_state,
            regularization=regularization,
            diagnostics=diagnostics,
        )

    @staticmethod
    def parameter_count(hidden_size: int, state_size: int) -> int:
        """Return the exact number of trainable scalar parameters."""
        if hidden_size < 1 or state_size < 2:
            raise ValueError("invalid dimensions")
        return 2 * hidden_size * state_size + state_size + 3

    @staticmethod
    def _normalise_mask(
        attention_mask: Optional[Tensor],
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones(
                batch_size, sequence_length, dtype=torch.bool, device=device
            )
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"attention_mask must have shape ({batch_size}, {sequence_length})"
            )
        return attention_mask.to(device=device, dtype=torch.bool)
