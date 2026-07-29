"""Heraclitus 2.0: multimodal higher-dimensional predictive state."""
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
    mode_entropy: Tensor
    effective_modes: Tensor
    best_mode_probability: Tensor
    next_best_mode_probability: Tensor
    mode_separation: Tensor
    latent_standard_deviation: Tensor

    @property
    def shadow_entropy(self) -> Tensor:
        return self.mode_entropy

    @property
    def effective_shadows(self) -> Tensor:
        return self.effective_modes


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
        information_weight: float = 1e-4,
        calibration_weight: float = 1e-4,
    ) -> Tensor:
        return (
            predictive_weight * self.regularization["predictive_nll"]
            + diversity_weight * self.regularization["mode_collapse"]
            + orthogonality_weight * self.regularization["orthogonality"]
            + residual_weight * self.regularization["residual_energy"]
            + information_weight * self.regularization["information_floor"]
            + calibration_weight * self.regularization["calibration"]
        )


class HeraclitusParameter(nn.Module):
    """Maintain persistent mode-specific state and bounded predictive innovation.

    Each mode owns its own mean, diagonal covariance and low-rank covariance
    factor. A product of learned Householder reflections supplies norm-preserving
    cross-coordinate dynamics; context-dependent contraction keeps the spectral
    radius below one. Likelihoods use the Woodbury identity, avoiding dense
    state-size covariance inversion.
    """

    def __init__(self, config: HeraclitusConfig):
        super().__init__()
        self.config = config
        d, r, k, c = (
            config.hidden_size,
            config.state_size,
            config.num_modes,
            config.covariance_rank,
        )

        projection = torch.empty(d, r)
        nn.init.orthogonal_(projection)
        self.projection = nn.Parameter(projection)
        reconstruction = torch.empty(r, d)
        nn.init.normal_(reconstruction, mean=0.0, std=1e-3 / r**0.5)
        self.reconstruction = nn.Parameter(reconstruction)

        self.initial_mode_means = nn.Parameter(torch.zeros(k, r))
        nn.init.normal_(self.initial_mode_means, std=0.02)
        self.transition_vectors = nn.Parameter(torch.randn(config.transition_reflections, r))
        self.retention_logits = nn.Parameter(torch.zeros(k, r))
        self.retention_context = nn.Parameter(torch.zeros(r, k * r))

        self.process_noise_logits = nn.Parameter(torch.full((k, r), -4.0))
        self.observation_noise_logits = nn.Parameter(torch.full((r,), -2.0))
        self.noise_context = nn.Parameter(torch.zeros(r, 2 * r))
        self.process_factors = nn.Parameter(torch.zeros(k, r, c))
        if c:
            nn.init.normal_(self.process_factors, std=config.covariance_factor_scale / r**0.5)

        self.mode_prior_logits = nn.Parameter(torch.zeros(k))
        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.0))
        self.novelty_gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.novelty_gate_scale = nn.Parameter(torch.tensor(1.0))
        self.reliability_gate_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(config.dropout)

    def effective_projection(self) -> Tensor:
        return bounded_frobenius(
            self.projection.float(), self.config.projection_norm_bound, self.config.epsilon
        )

    def effective_reconstruction(self) -> Tensor:
        return bounded_frobenius(
            self.reconstruction.float(), self.config.reconstruction_norm_bound, self.config.epsilon
        )

    def transition_basis(self) -> Tensor:
        return self.transition_vectors.float() / self.transition_vectors.float().norm(
            dim=-1, keepdim=True
        ).clamp_min(self.config.epsilon)

    def _apply_orthogonal_transition(self, values: Tensor) -> Tensor:
        """Apply a product of Householder reflections along the state axis."""
        result = values
        state_axis = -1 if values.ndim == 3 else -2
        for vector in self.transition_basis():
            if state_axis == -1:
                coefficient = (result * vector).sum(dim=-1, keepdim=True)
                result = result - 2.0 * coefficient * vector
            else:
                coefficient = (result * vector.view(1, 1, -1, 1)).sum(
                    dim=-2, keepdim=True
                )
                result = result - 2.0 * coefficient * vector.view(1, 1, -1, 1)
        return result

    def _retention(self, context: Tensor) -> Tensor:
        b, k, r = context.shape
        contextual = torch.tanh(context.mean(dim=1) @ self.retention_context.float()).view(b, k, r)
        logits = self.retention_logits.float().unsqueeze(0) + contextual
        span = self.config.max_retention - self.config.min_retention
        return self.config.min_retention + span * torch.sigmoid(logits)

    def _noise(self, context: Tensor) -> Tuple[Tensor, Tensor]:
        b, k, r = context.shape
        summary = context.mean(dim=1)
        contextual = torch.tanh(summary @ self.noise_context.float())
        process_context, observation_context = contextual.chunk(2, dim=-1)
        q = self.config.process_noise_floor + torch.nn.functional.softplus(
            self.process_noise_logits.float().unsqueeze(0)
            + self.config.context_noise_scale * process_context[:, None, :]
        )
        obs = self.config.observation_noise_floor + torch.nn.functional.softplus(
            self.observation_noise_logits.float().unsqueeze(0)
            + self.config.context_noise_scale * observation_context
        )
        return q, obs

    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> HeraclitusState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        target = self.initial_mode_means.device if device is None else device
        k, r, c = self.config.num_modes, self.config.state_size, self.config.covariance_rank
        means = self.initial_mode_means.float().to(target).unsqueeze(0).expand(batch_size, -1, -1).clone()
        variances = torch.full_like(means, self.config.initial_variance)
        factors = torch.zeros(batch_size, k, r, c, device=target, dtype=torch.float32)
        weights = torch.log_softmax(self.mode_prior_logits.float(), dim=0)
        weights = weights.to(target).unsqueeze(0).expand(batch_size, -1).clone()
        steps = torch.zeros(batch_size, dtype=torch.long, device=target)
        return HeraclitusState(means, variances, factors, weights, steps)

    def predictive_distribution(self, state: HeraclitusState) -> Tuple[Tensor, Tensor, Tensor]:
        state.validate(
            state.mode_means.shape[0],
            self.config.state_size,
            self.config.num_modes,
            self.config.covariance_rank,
        )
        prior = self._predict(state)
        total_variance = prior.mode_variances + prior.covariance_factors.square().sum(dim=-1)
        return prior.mode_means, total_variance, prior.mode_log_weights.exp()

    def _predict(self, state: HeraclitusState) -> HeraclitusState:
        rotated_means = self._apply_orthogonal_transition(state.mode_means.float())
        retention = self._retention(rotated_means)
        prior_means = retention * rotated_means
        q, _ = self._noise(prior_means)
        prior_variances = (retention.square() * state.mode_variances.float() + q).clamp_min(
            self.config.epsilon
        )
        factors = self._apply_orthogonal_transition(state.covariance_factors.float())
        factors = retention.unsqueeze(-1) * factors
        if self.config.covariance_rank:
            factors = factors + self.process_factors.float().unsqueeze(0)
        learned = torch.log_softmax(self.mode_prior_logits.float(), dim=0).unsqueeze(0)
        memory = self.config.mode_weight_memory
        weights = torch.log_softmax(memory * state.mode_log_weights.float() + (1.0 - memory) * learned, dim=-1)
        return HeraclitusState(prior_means, prior_variances, factors, weights, state.steps)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
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
        if d != self.config.hidden_size or t < 1:
            raise ValueError("hidden size mismatch or empty sequence")
        mask = self._normalise_mask(attention_mask, b, t, hidden_states.device)
        runtime = self.initial_state(b, hidden_states.device) if state is None else state
        if runtime.mode_means.device != hidden_states.device:
            runtime = runtime.to(device=hidden_states.device)
        runtime.validate(b, self.config.state_size, self.config.num_modes, self.config.covariance_rank)

        x = hidden_states.float()
        normalised = x / x.square().mean(dim=-1, keepdim=True).add(self.config.epsilon).sqrt()
        latent = normalised @ self.effective_projection()
        reconstruction = self.effective_reconstruction()
        residual_scale = self.config.max_residual_scale * torch.sigmoid(self.residual_scale_logit.float())

        deltas = []
        nlls = []
        entropy_terms = []
        separation_terms = []
        surprise_terms = []
        innovation_terms = []
        calibration_terms = []
        best_terms = []
        second_terms = []

        for token_index in range(t):
            valid = mask[:, token_index].view(b, 1)
            z = latent[:, token_index, :]
            prior = self._predict(runtime)
            _, observation_noise = self._noise(prior.mode_means)
            error = z[:, None, :] - prior.mode_means
            log_likelihood, solved = self._low_rank_gaussian(error, prior, observation_noise)
            component_log_prob = prior.mode_log_weights + log_likelihood
            mixture_log_prob = torch.logsumexp(component_log_prob, dim=-1)
            raw_weights = torch.softmax(component_log_prob, dim=-1)
            floor = self.config.min_mode_probability
            posterior_weights = raw_weights * (1.0 - floor * self.config.num_modes) + floor
            posterior_log_weights = posterior_weights.log()

            factor_projection = torch.einsum("bkrc,bkr->bkc", prior.covariance_factors, solved)
            correction = prior.mode_variances * solved + torch.einsum(
                "bkrc,bkc->bkr", prior.covariance_factors, factor_projection
            )
            posterior_means = prior.mode_means + correction
            total_prior_variance = prior.mode_variances + prior.covariance_factors.square().sum(dim=-1)
            gain = total_prior_variance / (total_prior_variance + observation_noise[:, None, :])
            posterior_variances = (
                (1.0 - gain) * prior.mode_variances
            ).clamp_min(self.config.epsilon)
            posterior_factors = prior.covariance_factors * (1.0 - gain).sqrt().unsqueeze(-1)

            mixture_prior = (posterior_weights[:, :, None] * prior.mode_means).sum(dim=1)
            innovation = z - mixture_prior
            within = total_prior_variance
            between = (prior.mode_means - mixture_prior[:, None, :]).square()
            mixture_variance = (
                posterior_weights[:, :, None] * (within + between)
            ).sum(dim=1).clamp_min(self.config.epsilon)
            surprise = (innovation.square() / (mixture_variance + observation_noise)).mean(
                dim=-1, keepdim=True
            )
            entropy = -(posterior_weights * posterior_log_weights).sum(dim=-1, keepdim=True)
            reliability = torch.exp(-torch.nn.functional.softplus(self.reliability_gate_scale.float()) * mixture_variance.mean(dim=-1, keepdim=True).sqrt())
            novelty = torch.sigmoid(
                self.novelty_gate_bias.float()
                + torch.nn.functional.softplus(self.novelty_gate_scale.float()) * surprise.sqrt()
            )
            gate = novelty * reliability
            whitened = innovation / (mixture_variance + observation_noise).sqrt()
            token_delta = residual_scale * gate * (whitened @ reconstruction)
            token_delta = self.dropout(token_delta)
            token_delta = torch.where(valid, token_delta, torch.zeros_like(token_delta))
            deltas.append(token_delta)

            pairwise = torch.cdist(posterior_means, posterior_means, p=2)
            eye = torch.eye(self.config.num_modes, dtype=torch.bool, device=x.device)
            off_diagonal = pairwise.masked_select(~eye.unsqueeze(0)).view(b, -1)
            separation = off_diagonal.square().mean(dim=-1, keepdim=True)
            sorted_weights = posterior_weights.sort(dim=-1, descending=True).values
            best_terms.append(torch.where(valid, sorted_weights[:, :1], torch.zeros_like(valid)))
            second_terms.append(torch.where(valid, sorted_weights[:, 1:2], torch.zeros_like(valid)))
            entropy_terms.append(torch.where(valid, entropy, torch.zeros_like(entropy)))
            separation_terms.append(torch.where(valid, separation, torch.zeros_like(separation)))
            surprise_terms.append(torch.where(valid, surprise, torch.zeros_like(surprise)))
            innovation_terms.append(torch.where(valid, innovation.square().mean(dim=-1, keepdim=True).sqrt(), torch.zeros_like(valid, dtype=x.dtype)))
            expected_sq_error = (posterior_weights[:, :, None] * error.square()).sum(dim=(1, 2), keepdim=False) / self.config.state_size
            predicted_error = (mixture_variance + observation_noise).mean(dim=-1)
            calibration_terms.append(torch.where(valid.squeeze(-1), (expected_sq_error - predicted_error).square(), torch.zeros_like(predicted_error)))
            nlls.append(torch.where(valid.squeeze(-1), -mixture_log_prob, torch.zeros_like(mixture_log_prob)))

            next_state = HeraclitusState(
                posterior_means,
                posterior_variances,
                posterior_factors,
                posterior_log_weights,
                runtime.steps + valid.squeeze(-1).long(),
            )
            runtime = self._masked_state(runtime, next_state, valid)

        delta = torch.stack(deltas, dim=1)
        output = hidden_states + delta.to(hidden_states.dtype)
        valid_count = mask.sum().clamp_min(1).float()
        latent_std = latent[mask].std(dim=0, unbiased=False).mean() if mask.any() else latent.new_tensor(0.0)
        mode_separation = torch.stack(separation_terms, dim=1).sum() / valid_count
        information_floor = torch.relu(latent.new_tensor(0.5) - latent_std).square()
        mode_collapse = torch.exp(-mode_separation)
        residual_energy = delta.square().sum() / x.square().sum().clamp_min(self.config.epsilon)
        entropy = torch.stack(entropy_terms, dim=1).sum() / valid_count

        if detach_state:
            runtime = runtime.detach()
        regularization = {
            "predictive_nll": torch.stack(nlls, dim=1).sum() / valid_count,
            "mode_collapse": mode_collapse,
            "orthogonality": orthogonality_error(self.projection.float(), self.config.epsilon),
            "residual_energy": residual_energy,
            "information_floor": information_floor,
            "calibration": torch.stack(calibration_terms, dim=1).sum() / valid_count,
        }
        diagnostics = HeraclitusDiagnostics(
            innovation_rms=torch.stack(innovation_terms, dim=1).sum() / valid_count,
            surprise_mean=torch.stack(surprise_terms, dim=1).sum() / valid_count,
            residual_ratio=delta.norm() / x.norm().clamp_min(self.config.epsilon),
            posterior_variance=runtime.variance.mean(),
            mode_entropy=entropy,
            effective_modes=entropy.exp(),
            best_mode_probability=torch.stack(best_terms, dim=1).sum() / valid_count,
            next_best_mode_probability=torch.stack(second_terms, dim=1).sum() / valid_count,
            mode_separation=mode_separation,
            latent_standard_deviation=latent_std,
        )
        return HeraclitusOutput(output, runtime, regularization, diagnostics)

    def _low_rank_gaussian(
        self, error: Tensor, prior: HeraclitusState, observation_noise: Tensor
    ) -> Tuple[Tensor, Tensor]:
        diagonal = (prior.mode_variances + observation_noise[:, None, :]).clamp_min(
            self.config.epsilon
        )
        inverse_diagonal = diagonal.reciprocal()
        base_solution = inverse_diagonal * error
        logdet = diagonal.log().sum(dim=-1)
        quadratic = (error * base_solution).sum(dim=-1)
        factors = prior.covariance_factors
        if self.config.covariance_rank:
            weighted_factors = inverse_diagonal.unsqueeze(-1) * factors
            middle = torch.einsum("bkrc,bkrd->bkcd", factors, weighted_factors)
            identity = torch.eye(self.config.covariance_rank, device=error.device, dtype=error.dtype)
            middle = middle + identity.view(1, 1, self.config.covariance_rank, self.config.covariance_rank)
            rhs = torch.einsum("bkrc,bkr->bkc", factors, base_solution)
            solved_middle = torch.linalg.solve(middle, rhs.unsqueeze(-1)).squeeze(-1)
            correction = torch.einsum("bkrc,bkc->bkr", weighted_factors, solved_middle)
            solved = base_solution - correction
            quadratic = (error * solved).sum(dim=-1)
            logdet = logdet + torch.linalg.slogdet(middle).logabsdet
        else:
            solved = base_solution
        constant = self.config.state_size * 1.8378770664093453
        return -0.5 * (quadratic + logdet + constant), solved

    @staticmethod
    def _masked_state(old: HeraclitusState, new: HeraclitusState, valid: Tensor) -> HeraclitusState:
        mode_mask = valid.unsqueeze(-1)
        factor_mask = mode_mask.unsqueeze(-1)
        return HeraclitusState(
            torch.where(mode_mask, new.mode_means, old.mode_means),
            torch.where(mode_mask, new.mode_variances, old.mode_variances),
            torch.where(factor_mask, new.covariance_factors, old.covariance_factors),
            torch.where(valid, new.mode_log_weights, old.mode_log_weights),
            torch.where(valid.squeeze(-1), new.steps, old.steps),
        )

    @staticmethod
    def parameter_count(
        hidden_size: int,
        state_size: int,
        num_modes: int = 4,
        covariance_rank: int = 4,
        transition_reflections: int = 4,
    ) -> int:
        if min(hidden_size, state_size, num_modes, transition_reflections) < 1:
            raise ValueError("invalid dimensions")
        return (
            2 * hidden_size * state_size
            + num_modes * state_size
            + transition_reflections * state_size
            + num_modes * state_size
            + state_size * num_modes * state_size
            + num_modes * state_size
            + state_size
            + state_size * 2 * state_size
            + num_modes * state_size * covariance_rank
            + num_modes
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
