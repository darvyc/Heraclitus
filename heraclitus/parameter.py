"""Heraclitus 1.0: a causal predictive low-rank transformer parameter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .config import HeraclitusConfig
from .mathematics import bounded_frobenius, orthogonality_error
from .state import HeraclitusState


@dataclass(frozen=True)
class HeraclitusDiagnostics:
    innovation_rms: Tensor
    surprise_mean: Tensor
    residual_ratio: Tensor
    posterior_variance: Tensor
    shadow_entropy: Tensor
    effective_shadows: Tensor
    best_shadow_probability: Tensor
    next_best_shadow_probability: Tensor


@dataclass(frozen=True)
class HeraclitusOutput:
    hidden_states: Tensor
    state: HeraclitusState
    regularization: Dict[str, Tensor]
    diagnostics: HeraclitusDiagnostics

    def regularization_loss(
        self,
        predictive_weight: float = 1e-3,
        diversity_weight: float = 1e-4,
        orthogonality_weight: float = 1e-4,
        residual_weight: float = 1e-4,
    ) -> Tensor:
        """Return the weighted auxiliary objective used during training."""
        return (
            predictive_weight * self.regularization["predictive_nll"]
            + diversity_weight * self.regularization["shadow_collapse"]
            + orthogonality_weight * self.regularization["orthogonality"]
            + residual_weight * self.regularization["residual_energy"]
        )


class HeraclitusParameter(nn.Module):
    """Maintain predictive latent state and write bounded innovation to an LLM.

    The module is causal and batch-isolated. A contractive diagonal transition
    predicts a latent observation. Context-conditioned Gaussian shadows retain
    several plausible local futures. Evidence updates their probabilities, a
    diagonal uncertainty filter corrects the state, and only normalised
    prediction error is reconstructed into the transformer residual stream.
    """

    def __init__(self, config: HeraclitusConfig):
        super().__init__()
        self.config = config
        d, r, k = config.hidden_size, config.state_size, config.num_shadows

        projection = torch.empty(d, r)
        nn.init.orthogonal_(projection)
        self.projection = nn.Parameter(projection)

        reconstruction = torch.empty(r, d)
        nn.init.normal_(reconstruction, mean=0.0, std=1e-3 / r**0.5)
        self.reconstruction = nn.Parameter(reconstruction)

        self.initial_mean = nn.Parameter(torch.zeros(r))
        self.retention_logits = nn.Parameter(torch.zeros(r))
        self.process_noise_logits = nn.Parameter(torch.full((r,), -4.0))
        self.observation_noise_logits = nn.Parameter(torch.full((r,), -2.0))

        offsets = torch.randn(k, r)
        offsets = offsets - offsets.mean(dim=0, keepdim=True)
        offsets = offsets / offsets.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
            config.epsilon
        )
        self.shadow_offsets = nn.Parameter(offsets)
        self.shadow_context = nn.Parameter(torch.zeros(r, k * r))
        self.shadow_prior_logits = nn.Parameter(torch.zeros(k))

        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.0))
        self.surprise_gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.surprise_gate_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(config.dropout)

    def effective_projection(self) -> Tensor:
        return bounded_frobenius(
            self.projection.float(),
            self.config.projection_norm_bound,
            self.config.epsilon,
        )

    def effective_reconstruction(self) -> Tensor:
        return bounded_frobenius(
            self.reconstruction.float(),
            self.config.reconstruction_norm_bound,
            self.config.epsilon,
        )

    def retention(self) -> Tensor:
        span = self.config.max_retention - self.config.min_retention
        return self.config.min_retention + span * torch.sigmoid(
            self.retention_logits.float()
        )

    def process_noise(self) -> Tensor:
        return self.config.process_noise_floor + torch.nn.functional.softplus(
            self.process_noise_logits.float()
        )

    def observation_noise(self) -> Tensor:
        return self.config.observation_noise_floor + torch.nn.functional.softplus(
            self.observation_noise_logits.float()
        )

    def initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> HeraclitusState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        target = device if device is not None else self.initial_mean.device
        mean = self.initial_mean.float().to(target).unsqueeze(0).expand(batch_size, -1).clone()
        variance = torch.full_like(mean, self.config.initial_variance)
        log_weights = torch.log_softmax(self.shadow_prior_logits.float(), dim=0)
        log_weights = log_weights.to(target).unsqueeze(0).expand(batch_size, -1).clone()
        steps = torch.zeros(batch_size, dtype=torch.long, device=target)
        return HeraclitusState(mean, variance, log_weights, steps)

    def predictive_distribution(self, state: HeraclitusState) -> Tuple[Tensor, Tensor, Tensor]:
        """Return one-step shadow means, diagonal variance, and prior probabilities."""
        batch_size = state.mean.shape[0]
        state.validate(batch_size, self.config.state_size, self.config.num_shadows)
        mean = state.mean.float()
        variance = state.variance.float().clamp_min(self.config.epsilon)
        a = self.retention().unsqueeze(0)
        prior_mean = a * mean
        prior_variance = (a.square() * variance + self.process_noise().unsqueeze(0)).clamp_min(
            self.config.epsilon
        )
        shadow_means = self._shadow_means(prior_mean, prior_variance)
        prior_log_weights = self._prior_log_weights(state.shadow_log_weights.float())
        return shadow_means, prior_variance, prior_log_weights.exp()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.forward_with_state(hidden_states, None, attention_mask).hidden_states

    def forward_with_state(
        self,
        hidden_states: Tensor,
        state: Optional[HeraclitusState] = None,
        attention_mask: Optional[Tensor] = None,
        detach_state: bool = True,
    ) -> HeraclitusOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, hidden)")
        b, t, d = hidden_states.shape
        if d != self.config.hidden_size:
            raise ValueError(f"expected hidden size {self.config.hidden_size}, got {d}")
        if t < 1:
            raise ValueError("sequence length must be positive")

        mask = self._normalise_mask(attention_mask, b, t, hidden_states.device)
        runtime = state or self.initial_state(b, hidden_states.device)
        runtime.validate(b, self.config.state_size, self.config.num_shadows)
        if runtime.mean.device != hidden_states.device:
            runtime = runtime.to(device=hidden_states.device)

        x = hidden_states.float()
        rms = x.square().mean(dim=-1, keepdim=True).add(self.config.epsilon).sqrt()
        latent = (x / rms) @ self.effective_projection()
        reconstruction = self.effective_reconstruction()

        mean = runtime.mean.float()
        variance = runtime.variance.float().clamp_min(self.config.epsilon)
        shadow_log_weights = runtime.shadow_log_weights.float()
        steps = runtime.steps

        a = self.retention().unsqueeze(0)
        q = self.process_noise().unsqueeze(0)
        obs_noise = self.observation_noise().unsqueeze(0)
        residual_scale = self.config.max_residual_scale * torch.sigmoid(
            self.residual_scale_logit.float()
        )

        deltas = []
        nlls = []
        collapse_terms = []
        innovation_terms = []
        surprise_terms = []
        entropy_terms = []
        best_terms = []
        second_terms = []
        log_two_pi = latent.new_tensor(1.8378770664093453)

        for token_index in range(t):
            valid = mask[:, token_index].unsqueeze(-1)
            z = latent[:, token_index, :]

            prior_mean = a * mean
            prior_variance = (a.square() * variance + q).clamp_min(self.config.epsilon)
            shadow_means = self._shadow_means(prior_mean, prior_variance)
            predictive_variance = (prior_variance + obs_noise).unsqueeze(1)
            prior_log_weights = self._prior_log_weights(shadow_log_weights)

            error_by_shadow = z.unsqueeze(1) - shadow_means
            component_nll = 0.5 * (
                error_by_shadow.square() / predictive_variance
                + predictive_variance.log()
                + log_two_pi
            ).sum(dim=-1)
            component_log_prob = prior_log_weights - component_nll
            mixture_log_prob = torch.logsumexp(component_log_prob, dim=-1)
            raw_posterior = torch.softmax(component_log_prob, dim=-1)
            floor = self.config.min_shadow_probability
            posterior_weights = raw_posterior * (1.0 - floor * self.config.num_shadows) + floor
            posterior_log_weights = posterior_weights.log()

            predicted_mean = (posterior_weights.unsqueeze(-1) * shadow_means).sum(dim=1)
            innovation = z - predicted_mean
            gain = prior_variance / (prior_variance + obs_noise)
            posterior_mean = predicted_mean + gain * innovation

            between_shadow_variance = (
                posterior_weights.unsqueeze(-1)
                * (shadow_means - predicted_mean.unsqueeze(1)).square()
            ).sum(dim=1)
            posterior_variance = (
                (1.0 - gain) * prior_variance + between_shadow_variance
            ).clamp_min(self.config.epsilon)

            surprise = (
                innovation.square() / (prior_variance + obs_noise)
            ).mean(dim=-1, keepdim=True)
            gate = torch.sigmoid(
                self.surprise_gate_bias.float()
                + torch.nn.functional.softplus(self.surprise_gate_scale.float())
                * surprise.sqrt()
            )
            whitened_innovation = innovation / (prior_variance + obs_noise).sqrt()
            token_delta = residual_scale * gate * (whitened_innovation @ reconstruction)
            token_delta = self.dropout(token_delta)
            token_delta = torch.where(valid, token_delta, torch.zeros_like(token_delta))
            deltas.append(token_delta)

            entropy = -(posterior_weights * posterior_log_weights).sum(dim=-1, keepdim=True)
            sorted_weights = posterior_weights.sort(dim=-1, descending=True).values
            best_terms.append(torch.where(valid, sorted_weights[:, :1], torch.zeros_like(valid)))
            second_terms.append(torch.where(valid, sorted_weights[:, 1:2], torch.zeros_like(valid)))
            entropy_terms.append(torch.where(valid, entropy, torch.zeros_like(entropy)))
            nlls.append(
                torch.where(valid.squeeze(-1), -mixture_log_prob, torch.zeros_like(mixture_log_prob))
            )
            pairwise = torch.cdist(shadow_means, shadow_means, p=2)
            eye = torch.eye(self.config.num_shadows, device=pairwise.device, dtype=torch.bool)
            off_diagonal = pairwise.masked_select(~eye.unsqueeze(0)).view(b, -1)
            collapse_terms.append(torch.exp(-off_diagonal.square().mean(dim=-1)).mean())
            innovation_terms.append(
                torch.where(
                    valid,
                    innovation.square().mean(dim=-1, keepdim=True).sqrt(),
                    torch.zeros_like(valid, dtype=x.dtype),
                )
            )
            surprise_terms.append(torch.where(valid, surprise, torch.zeros_like(surprise)))

            mean = torch.where(valid, posterior_mean, mean)
            variance = torch.where(valid, posterior_variance, variance)
            shadow_log_weights = torch.where(valid, posterior_log_weights, shadow_log_weights)
            steps = steps + valid.squeeze(-1).to(torch.long)

        delta = torch.stack(deltas, dim=1)
        output = hidden_states + delta.to(hidden_states.dtype)
        valid_count = mask.sum().clamp_min(1).to(torch.float32)
        predictive_nll = torch.stack(nlls, dim=1).sum() / valid_count
        shadow_collapse = torch.stack(collapse_terms).mean()
        residual_energy = delta.square().sum() / x.square().sum().clamp_min(self.config.epsilon)
        residual_ratio = delta.norm() / x.norm().clamp_min(self.config.epsilon)
        shadow_entropy = torch.stack(entropy_terms, dim=1).sum() / valid_count

        next_state = HeraclitusState(mean, variance, shadow_log_weights, steps)
        if detach_state:
            next_state = next_state.detach()

        regularization = {
            "predictive_nll": predictive_nll,
            "shadow_collapse": shadow_collapse,
            "orthogonality": orthogonality_error(self.projection.float(), self.config.epsilon),
            "residual_energy": residual_energy,
        }
        diagnostics = HeraclitusDiagnostics(
            innovation_rms=torch.stack(innovation_terms, dim=1).sum() / valid_count,
            surprise_mean=torch.stack(surprise_terms, dim=1).sum() / valid_count,
            residual_ratio=residual_ratio,
            posterior_variance=variance.mean(),
            shadow_entropy=shadow_entropy,
            effective_shadows=shadow_entropy.exp(),
            best_shadow_probability=torch.stack(best_terms, dim=1).sum() / valid_count,
            next_best_shadow_probability=torch.stack(second_terms, dim=1).sum() / valid_count,
        )
        return HeraclitusOutput(output, next_state, regularization, diagnostics)

    def _shadow_means(self, prior_mean: Tensor, prior_variance: Tensor) -> Tensor:
        b, r = prior_mean.shape
        k = self.config.num_shadows
        base = self.shadow_offsets.float().unsqueeze(0).expand(b, -1, -1)
        contextual = torch.tanh(prior_mean @ self.shadow_context.float()).view(b, k, r)
        offsets = base + self.config.shadow_context_scale * contextual
        offsets = offsets - offsets.mean(dim=1, keepdim=True)
        offsets = offsets / offsets.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
            self.config.epsilon
        )
        return prior_mean.unsqueeze(1) + (
            self.config.shadow_scale * prior_variance.sqrt().unsqueeze(1) * offsets
        )

    def _prior_log_weights(self, previous_log_weights: Tensor) -> Tensor:
        learned_prior = torch.log_softmax(self.shadow_prior_logits.float(), dim=0).unsqueeze(0)
        memory = self.config.shadow_weight_memory
        blended = memory * previous_log_weights + (1.0 - memory) * learned_prior
        return torch.log_softmax(blended, dim=-1)

    @staticmethod
    def parameter_count(hidden_size: int, state_size: int, num_shadows: int = 4) -> int:
        if hidden_size < 1 or state_size < 2 or num_shadows < 2:
            raise ValueError("invalid dimensions")
        return (
            2 * hidden_size * state_size
            + num_shadows * state_size * state_size
            + num_shadows * state_size
            + 4 * state_size
            + num_shadows
            + 3
        )

    @staticmethod
    def _normalise_mask(
        attention_mask: Optional[Tensor],
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"attention_mask must have shape ({batch_size}, {sequence_length})"
            )
        return attention_mask.to(device=device, dtype=torch.bool)
