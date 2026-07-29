"""Mathematical primitives used by the Heraclitus parameter."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def safe_unit(
    value: Tensor,
    dim: int = -1,
    epsilon: float = 1e-6,
    fallback: Optional[Tensor] = None,
) -> Tensor:
    """Normalise vectors and use a deterministic unit fallback for zero vectors."""
    norm = value.norm(dim=dim, keepdim=True)
    normalised = value / norm.clamp_min(epsilon)
    if fallback is None:
        fallback_value = torch.zeros_like(value)
        fallback_value.select(dim, 0).fill_(1.0)
    else:
        fallback_value = fallback.to(device=value.device, dtype=value.dtype)
        fallback_value = fallback_value.expand_as(value)
        fallback_value = fallback_value / fallback_value.norm(
            dim=dim, keepdim=True
        ).clamp_min(epsilon)
    return torch.where(norm > epsilon, normalised, fallback_value)


def bounded_frobenius(weight: Tensor, max_norm: float, epsilon: float = 1e-6) -> Tensor:
    """Differentiably project a matrix into a Frobenius-norm ball."""
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    norm = weight.norm(p="fro")
    scale = (max_norm / norm.clamp_min(epsilon)).clamp(max=1.0)
    return weight * scale


def spherical_ema(previous: Tensor, observation: Tensor, decay: float, epsilon: float) -> Tensor:
    """Exponential moving average followed by projection to the unit sphere."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1)")
    return safe_unit(
        decay * previous + (1.0 - decay) * observation,
        epsilon=epsilon,
        fallback=previous,
    )


def geodesic_fraction(left: Tensor, right: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Return spherical disagreement in [0, 1] using (1 - cosine) / 2."""
    left_unit = safe_unit(left, epsilon=epsilon)
    right_unit = safe_unit(right, epsilon=epsilon)
    cosine = (left_unit * right_unit).sum(dim=-1).clamp(-1.0, 1.0)
    return 0.5 * (1.0 - cosine)


def orthogonality_error(weight: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Mean squared departure of column directions from an orthonormal frame."""
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    columns = weight / weight.norm(dim=0, keepdim=True).clamp_min(epsilon)
    gram = columns.transpose(0, 1) @ columns
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - identity).square().mean()


def orthogonal_procrustes(source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Fit a proper rotation aligning source observations to target observations."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be rank-2 matrices")
    if source.shape != target.shape:
        raise ValueError("source and target must have identical shape")
    if source.shape[0] < source.shape[1]:
        raise ValueError("the number of anchors must cover the state dimension")
    cross_covariance = source.transpose(0, 1) @ target
    u, singular_values, vh = torch.linalg.svd(cross_covariance)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation, singular_values
