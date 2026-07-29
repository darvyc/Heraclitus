"""Null models and alignment tools for the Heraclitus geometry."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def random_alignment_probability(threshold: float) -> float:
    """Exact P[u dot v >= threshold] for independent uniform u, v on S^2."""
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [-1, 1]")
    return 0.5 * (1.0 - threshold)


def expected_random_degree(n_members: int, threshold: float) -> float:
    """Expected outgoing matches under the independent-uniform S^2 null."""
    if n_members < 1:
        raise ValueError("n_members must be positive")
    return (n_members - 1) * random_alignment_probability(threshold)


def threshold_for_expected_random_degree(
    n_members: int,
    expected_degree: float,
) -> float:
    """Invert the S^2 null to choose a cosine threshold."""
    if n_members < 2:
        raise ValueError("n_members must be at least 2")
    if not 0.0 <= expected_degree <= n_members - 1:
        raise ValueError("expected_degree must lie in [0, n_members - 1]")
    return 1.0 - 2.0 * expected_degree / (n_members - 1)


def orthogonal_procrustes(source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Fit a proper 3D rotation aligning source directions to target directions.

    Returns (rotation, singular_values), where source @ rotation approximates
    target in least-squares Frobenius norm. At least three non-collinear anchor
    pairs are required for a well-identified rotation.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be rank-2 matrices")
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("source and target must have identical shape (n, 3)")
    if source.shape[0] < 3:
        raise ValueError("at least three anchor pairs are required")
    cross_covariance = source.transpose(0, 1) @ target
    u, singular_values, vh = torch.linalg.svd(cross_covariance)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation, singular_values
