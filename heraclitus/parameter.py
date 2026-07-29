"""Heraclitus bounded recurrent memory adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
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
    memory_rms: Tensor
    effective_slots: Tensor
    maximum_slot_norm: Tensor


@dataclass(frozen=True)
class HeraclitusOutput:
    """Adapted hidden states, continuation state and training diagnostics."""

    hidden_states: Tensor
    state: HeraclitusState
    regularization: Dict[str, Tensor]
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
    """Add bounded, explicit recurrent memory to transformer hidden states.

    Each token retrieves a multi-head content-addressed summary from a compact
    slot bank. A novelty and usage-aware allocator then performs a bounded
    erase-and-add update. Runtime state is explicit, causal, serialisable,
    batch-isolated and reorderable for beam search.
    """

    def __init__(self, config: HeraclitusConfig):
        super().__init__()
        self.config = config
        d, r, m = config.hidden_size, config.state_size, config.memory_slots

        self.input_norm = nn.RMSNorm(d)
        self.query = nn.Linear(d, r, bias=False)
        self.key = nn.Linear(r, r, bias=False)
        self.value = nn.Linear(r, r, bias=False)
        self.candidate = nn.Linear(d, r, bias=False)
        self.erase = nn.Linear(d, r)
        self.write_gate = nn.Linear(d + r, 1)
        self.output = nn.Linear(r, d, bias=False)

        self.initial_memory = nn.Parameter(torch.empty(m, r))
        self.retention_logits = nn.Parameter(torch.zeros(m))
        self.residual_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.initial_memory, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.write_gate.bias, -2.0)

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
        )
        return sum(parameter.numel() for parameter in HeraclitusAdapter(config).parameters())

    def initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> HeraclitusState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        target = self.initial_memory.device if device is None else device
        memory = (
            self.initial_memory.float()
            .to(target)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        usage = torch.zeros(batch_size, self.config.memory_slots, device=target)
        steps = torch.zeros(batch_size, dtype=torch.long, device=target)
        return HeraclitusState(memory, usage, steps)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        return self.forward_with_state(hidden_states, attention_mask=attention_mask).hidden_states

    def forward_with_state(
        self,
        hidden_states: Tensor,
        state: Optional[HeraclitusState] = None,
        attention_mask: Optional[Tensor] = None,
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
        normalised = self.input_norm(source)
        memory = runtime.memory.float()
        usage = runtime.usage.float()
        steps = runtime.steps

        heads = self.config.num_heads
        head_size = self.config.state_size // heads
        attention_scale = head_size**-0.5
        retention = self.config.min_retention + (
            self.config.max_retention - self.config.min_retention
        ) * torch.sigmoid(self.retention_logits.float())
        residual_scale = self.config.max_residual_scale * torch.sigmoid(
            self.residual_scale_logit.float()
        )

        outputs = []
        read_entropies = []
        write_entropies = []
        write_rates = []
        residual_energies = []
        write_energies = []

        for index in range(sequence):
            valid = mask[:, index].view(batch, 1)
            token = normalised[:, index]

            query = self.query(token).view(batch, heads, head_size)
            keys = self.key(memory).view(batch, self.config.memory_slots, heads, head_size)
            values = self.value(memory).view(batch, self.config.memory_slots, heads, head_size)
            read_logits = torch.einsum("bhd,bmhd->bhm", query, keys) * attention_scale
            read_weights = read_logits.softmax(dim=-1)
            read = torch.einsum("bhm,bmhd->bhd", read_weights, values).reshape(
                batch, self.config.state_size
            )

            delta = self.dropout(residual_scale * self.output(read))
            delta = torch.where(valid, delta, torch.zeros_like(delta))
            outputs.append(source[:, index] + delta)

            candidate = torch.tanh(self.candidate(token))
            candidate_norm = torch.nn.functional.normalize(
                candidate, dim=-1, eps=self.config.epsilon
            )
            memory_norm = torch.nn.functional.normalize(memory, dim=-1, eps=self.config.epsilon)
            novelty = 1.0 - torch.einsum("br,bmr->bm", candidate_norm, memory_norm)
            normalised_usage = usage / usage.mean(dim=-1, keepdim=True).clamp_min(
                self.config.epsilon
            )
            write_logits = novelty - self.config.usage_penalty * normalised_usage
            write_weights = write_logits.softmax(dim=-1)

            gate_input = torch.cat([token, read], dim=-1)
            write_rate = self.config.max_write_rate * torch.sigmoid(self.write_gate(gate_input))
            erase = torch.sigmoid(self.erase(token)).unsqueeze(1)
            amount = write_rate.unsqueeze(1) * write_weights.unsqueeze(-1)

            retained = memory * retention.view(1, -1, 1)
            updated = retained * (1.0 - amount * erase) + amount * candidate.unsqueeze(1)
            updated = updated.clamp(-1.0, 1.0)
            usage_updated = usage * self.config.usage_decay + write_rate * write_weights
            memory = torch.where(valid.unsqueeze(-1), updated, memory)
            usage = torch.where(valid, usage_updated, usage)
            steps = steps + valid.squeeze(-1).to(torch.long)

            mean_read_weights = read_weights.mean(dim=1)
            read_entropy = -(
                mean_read_weights
                * mean_read_weights.clamp_min(self.config.epsilon).log()
            ).sum(-1)
            write_entropy = -(
                write_weights * write_weights.clamp_min(self.config.epsilon).log()
            ).sum(-1)
            read_entropies.append(
                torch.where(valid.squeeze(-1), read_entropy, torch.zeros_like(read_entropy))
            )
            write_entropies.append(
                torch.where(valid.squeeze(-1), write_entropy, torch.zeros_like(write_entropy))
            )
            write_rates.append(torch.where(valid, write_rate, torch.zeros_like(write_rate)))
            residual_energies.append(delta.square().mean())
            write_energies.append((amount * candidate.unsqueeze(1)).square().mean())

        output = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        final_state = HeraclitusState(memory, usage, steps)
        if detach_state:
            final_state = final_state.detach()

        valid_count = mask.sum().clamp_min(1).to(source.dtype)
        read_entropy = torch.stack(read_entropies).sum() / valid_count
        write_entropy = torch.stack(write_entropies).sum() / valid_count
        mean_write_rate = torch.stack(write_rates).sum() / valid_count
        residual_energy = torch.stack(residual_energies).mean()
        write_energy = torch.stack(write_energies).mean()
        slot_probability = usage.softmax(dim=-1)
        slot_balance = (
            slot_probability.mean(dim=0) - 1.0 / self.config.memory_slots
        ).square().mean()
        residual_ratio = (output.float() - source).norm() / source.norm().clamp_min(
            self.config.epsilon
        )
        effective_slots = torch.exp(
            -(
                slot_probability
                * slot_probability.clamp_min(self.config.epsilon).log()
            ).sum(-1)
        ).mean()

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
                memory_rms=memory.square().mean().sqrt(),
                effective_slots=effective_slots,
                maximum_slot_norm=memory.norm(dim=-1).amax(),
            ),
        )

    @staticmethod
    def _normalise_mask(
        mask: Optional[Tensor],
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
