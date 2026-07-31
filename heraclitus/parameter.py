"""Heraclitus 3 bounded recurrent memory adapter."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import HeraclitusConfig
from .state import HeraclitusState


@dataclass(frozen=True)
class HeraclitusDiagnostics:
    """Scalar observability signals for a completed forward pass."""

    read_entropy: Tensor
    write_entropy: Tensor
    write_rate: Tensor
    residual_ratio: Tensor
    maximum_residual_ratio: Tensor
    memory_rms: Tensor
    effective_slots: Tensor
    maximum_slot_norm: Tensor
    usage_maximum: Tensor


@dataclass(frozen=True)
class HeraclitusOutput:
    """Adapted hidden states, continuation state and training diagnostics."""

    hidden_states: Tensor
    state: HeraclitusState
    regularization: dict[str, Tensor]
    diagnostics: HeraclitusDiagnostics

    def regularization_loss(
        self,
        usage_weight: float = 1e-4,
        write_weight: float = 1e-4,
        residual_weight: float = 1e-4,
    ) -> Tensor:
        for name, value in {
            "usage_weight": usage_weight,
            "write_weight": write_weight,
            "residual_weight": residual_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        return (
            usage_weight * self.regularization["slot_balance"]
            + write_weight * self.regularization["write_energy"]
            + residual_weight * self.regularization["residual_energy"]
        )


class HeraclitusAdapter(nn.Module):
    """Add compact, explicit recurrent memory to transformer hidden states.

    Computation is performed in float32 even when the module parameters or host
    hidden stream use reduced precision. Every residual is projected onto a
    per-token norm ball before it is added to the host stream.
    """

    def __init__(self, config: HeraclitusConfig):
        super().__init__()
        self.config = config
        d, r, m = config.hidden_size, config.state_size, config.memory_slots

        self.input_norm_weight = nn.Parameter(torch.ones(d))
        self.query = nn.Linear(d, r, bias=False)
        self.key = nn.Linear(r, r, bias=False)
        self.value = nn.Linear(r, r, bias=False)
        self.candidate = nn.Linear(d + r, r, bias=False)
        self.erase = nn.Linear(d + r, r)
        self.write_gate = nn.Linear(d + r, 1)
        self.output = nn.Linear(r, d, bias=False)

        self.initial_memory = nn.Parameter(torch.empty(m, r))
        self.retention_logits = nn.Parameter(torch.empty(m))
        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.initial_memory, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.write_gate.bias, -2.0)
        nn.init.constant_(self.retention_logits, self._retention_logit(config.initial_retention))

    @staticmethod
    def parameter_count(
        hidden_size: int,
        state_size: int,
        memory_slots: int = 8,
        num_heads: int = 1,
    ) -> int:
        config = HeraclitusConfig(
            hidden_size=hidden_size,
            state_size=state_size,
            memory_slots=memory_slots,
            num_heads=num_heads,
            write_topk=min(2, memory_slots),
        )
        return sum(parameter.numel() for parameter in HeraclitusAdapter(config).parameters())

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> HeraclitusState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        target = self.initial_memory.device if device is None else device
        memory = (
            self.initial_memory.float()
            .to(target)
            .clamp(-1.0, 1.0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        usage = torch.zeros(batch_size, self.config.memory_slots, device=target)
        steps = torch.zeros(batch_size, dtype=torch.long, device=target)
        return HeraclitusState(memory, usage, steps)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.forward_with_state(hidden_states, attention_mask=attention_mask).hidden_states

    def forward_step(
        self,
        hidden_state: Tensor,
        state: HeraclitusState | None = None,
        valid: Tensor | None = None,
        detach_state: bool = True,
    ) -> HeraclitusOutput:
        """Process one token per batch element."""
        if hidden_state.ndim != 2:
            raise ValueError("hidden_state must have shape (batch, hidden)")
        mask = None if valid is None else valid.reshape(hidden_state.shape[0], 1)
        return self.forward_with_state(
            hidden_state.unsqueeze(1),
            state=state,
            attention_mask=mask,
            detach_state=detach_state,
        )

    def forward_with_state(
        self,
        hidden_states: Tensor,
        state: HeraclitusState | None = None,
        attention_mask: Tensor | None = None,
        detach_state: bool = True,
    ) -> HeraclitusOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, hidden)")
        batch, sequence, hidden = hidden_states.shape
        if hidden != self.config.hidden_size or sequence < 1:
            raise ValueError("hidden size mismatch or empty sequence")
        if not torch.isfinite(hidden_states).all():
            raise ValueError("hidden_states contains non-finite values")

        mask = self._normalise_mask(attention_mask, batch, sequence, hidden_states.device)
        runtime = self.initial_state(batch, hidden_states.device) if state is None else state
        if runtime.memory.device != hidden_states.device:
            runtime = runtime.to(device=hidden_states.device)
        runtime.validate(batch, self.config.memory_slots, self.config.state_size)

        source = hidden_states.float()
        inverse_rms = torch.rsqrt(
            source.square().mean(dim=-1, keepdim=True) + self.config.epsilon
        )
        normalised = source * inverse_rms * self.input_norm_weight.float()
        memory = runtime.memory.float()
        usage = runtime.usage.float()
        steps = runtime.steps

        retention = self._retention()
        residual_gate = torch.sigmoid(self.residual_scale_logit.float())

        outputs = []
        read_entropies = []
        write_entropies = []
        write_rates = []
        residual_energies = []
        write_energies = []
        residual_ratios = []

        for index in range(sequence):
            token_source = source[:, index]
            token = normalised[:, index]
            valid = mask[:, index].view(batch, 1)

            read, read_weights = self._read(token, memory)
            raw_delta = self._linear(read, self.output)
            raw_delta = self.dropout(raw_delta) * residual_gate
            delta, token_residual_ratio = self._bound_residual(raw_delta, token_source)
            delta = torch.where(valid, delta, torch.zeros_like(delta))
            token_residual_ratio = torch.where(
                valid.squeeze(-1),
                token_residual_ratio,
                torch.zeros_like(token_residual_ratio),
            )
            outputs.append(token_source + delta)

            controller = torch.cat([token, read], dim=-1)
            candidate = torch.tanh(self._linear(controller, self.candidate))
            write_weights = self._allocate(candidate, memory, usage)
            write_rate = self.config.max_write_rate * torch.sigmoid(
                self._linear(controller, self.write_gate)
            )
            erase = torch.sigmoid(self._linear(controller, self.erase)).unsqueeze(1)
            amount = write_rate.unsqueeze(1) * write_weights.unsqueeze(-1)

            retained = memory * retention.view(1, -1, 1)
            updated = retained * (1.0 - amount * erase) + amount * candidate.unsqueeze(1)
            updated = updated.clamp(-1.0, 1.0)
            usage_updated = (
                usage * self.config.usage_decay + write_rate * write_weights
            ).clamp(0.0, self.config.max_usage)
            memory = torch.where(valid.unsqueeze(-1), updated, memory)
            usage = torch.where(valid, usage_updated, usage)
            steps = steps + valid.squeeze(-1).to(torch.long)

            mean_read_weights = read_weights.mean(dim=1)
            read_entropy = self._entropy(mean_read_weights)
            write_entropy = self._entropy(write_weights)
            read_entropies.append(self._mask_scalar(read_entropy, valid))
            write_entropies.append(self._mask_scalar(write_entropy, valid))
            write_rates.append(self._mask_scalar(write_rate.squeeze(-1), valid))
            residual_energies.append(self._mask_scalar(delta.square().mean(dim=-1), valid))
            write_energies.append(
                self._mask_scalar(
                    (amount * candidate.unsqueeze(1)).square().mean(dim=(1, 2)),
                    valid,
                )
            )
            residual_ratios.append(token_residual_ratio)

        output = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        final_state = HeraclitusState(memory, usage, steps)
        if detach_state:
            final_state = final_state.detach()

        valid_count = mask.sum().clamp_min(1).to(source.dtype)
        read_entropy = torch.stack(read_entropies).sum() / valid_count
        write_entropy = torch.stack(write_entropies).sum() / valid_count
        mean_write_rate = torch.stack(write_rates).sum() / valid_count
        residual_energy = torch.stack(residual_energies).sum() / valid_count
        write_energy = torch.stack(write_energies).sum() / valid_count
        maximum_residual_ratio = torch.stack(residual_ratios).amax()

        slot_probability = usage / usage.sum(dim=-1, keepdim=True).clamp_min(
            self.config.epsilon
        )
        uniform = torch.full_like(slot_probability, 1.0 / self.config.memory_slots)
        slot_probability = torch.where(
            usage.sum(dim=-1, keepdim=True) > self.config.epsilon,
            slot_probability,
            uniform,
        )
        slot_balance = (
            slot_probability.mean(dim=0) - 1.0 / self.config.memory_slots
        ).square().mean()
        residual_ratio = (output.float() - source).norm() / source.norm().clamp_min(
            self.config.epsilon
        )
        effective_slots = torch.exp(self._entropy(slot_probability)).mean()

        return HeraclitusOutput(
            hidden_states=output,
            state=final_state,
            regularization={
                "slot_balance": slot_balance,
                "write_energy": write_energy,
                "residual_energy": residual_energy,
            },
            diagnostics=HeraclitusDiagnostics(
                read_entropy=read_entropy,
                write_entropy=write_entropy,
                write_rate=mean_write_rate,
                residual_ratio=residual_ratio,
                maximum_residual_ratio=maximum_residual_ratio,
                memory_rms=memory.square().mean().sqrt(),
                effective_slots=effective_slots,
                maximum_slot_norm=memory.norm(dim=-1).amax(),
                usage_maximum=usage.amax(),
            ),
        )

    def _read(self, token: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        batch = token.shape[0]
        heads = self.config.num_heads
        head_size = self.config.state_size // heads
        query = self._linear(token, self.query).view(batch, heads, head_size)
        keys = self._linear(memory, self.key).view(
            batch, self.config.memory_slots, heads, head_size
        )
        values = self._linear(memory, self.value).view(
            batch, self.config.memory_slots, heads, head_size
        )
        logits = torch.einsum("bhd,bmhd->bhm", query, keys) * head_size**-0.5
        weights = logits.softmax(dim=-1)
        read = torch.einsum("bhm,bmhd->bhd", weights, values).reshape(
            batch, self.config.state_size
        )
        return read, weights

    def _allocate(self, candidate: Tensor, memory: Tensor, usage: Tensor) -> Tensor:
        candidate_norm = F.normalize(candidate, dim=-1, eps=self.config.epsilon)
        memory_norm = F.normalize(memory, dim=-1, eps=self.config.epsilon)
        novelty = 1.0 - torch.einsum("br,bmr->bm", candidate_norm, memory_norm)
        normalised_usage = usage / usage.mean(dim=-1, keepdim=True).clamp_min(
            self.config.epsilon
        )
        logits = (
            novelty - self.config.usage_penalty * normalised_usage
        ) / self.config.write_temperature
        if self.config.write_topk < self.config.memory_slots:
            top_indices = logits.topk(self.config.write_topk, dim=-1).indices
            sparse_logits = torch.full_like(logits, -torch.inf)
            sparse_logits.scatter_(dim=-1, index=top_indices, src=logits.gather(-1, top_indices))
            logits = sparse_logits
        return logits.softmax(dim=-1)

    def _bound_residual(self, delta: Tensor, source: Tensor) -> tuple[Tensor, Tensor]:
        source_norm = source.norm(dim=-1, keepdim=True)
        delta_norm = delta.norm(dim=-1, keepdim=True)
        bound = self.config.max_residual_ratio * source_norm
        multiplier = torch.minimum(
            torch.ones_like(delta_norm),
            bound / delta_norm.clamp_min(self.config.epsilon),
        )
        bounded = delta * multiplier
        ratio = bounded.norm(dim=-1) / source_norm.squeeze(-1).clamp_min(self.config.epsilon)
        ratio = torch.where(source_norm.squeeze(-1) > 0.0, ratio, torch.zeros_like(ratio))
        return bounded, ratio

    def _retention(self) -> Tensor:
        return self.config.min_retention + (
            self.config.max_retention - self.config.min_retention
        ) * torch.sigmoid(self.retention_logits.float())

    def _retention_logit(self, retention: float) -> float:
        span = self.config.max_retention - self.config.min_retention
        probability = (retention - self.config.min_retention) / span
        probability = min(max(probability, self.config.epsilon), 1.0 - self.config.epsilon)
        return math.log(probability / (1.0 - probability))

    @staticmethod
    def _linear(value: Tensor, layer: nn.Linear) -> Tensor:
        bias = None if layer.bias is None else layer.bias.float()
        return F.linear(value.float(), layer.weight.float(), bias)

    def _entropy(self, probability: Tensor) -> Tensor:
        return -(
            probability * probability.clamp_min(self.config.epsilon).log()
        ).sum(dim=-1)

    @staticmethod
    def _mask_scalar(value: Tensor, valid: Tensor) -> Tensor:
        return torch.where(valid.squeeze(-1), value, torch.zeros_like(value))

    @staticmethod
    def _normalise_mask(
        mask: Tensor | None,
        batch: int,
        sequence: int,
        device: torch.device,
    ) -> Tensor:
        if mask is None:
            return torch.ones(batch, sequence, dtype=torch.bool, device=device)
        if mask.shape != (batch, sequence):
            raise ValueError(f"attention_mask must have shape ({batch}, {sequence})")
        return mask.to(device=device, dtype=torch.bool)


HeraclitusParameter = HeraclitusAdapter
