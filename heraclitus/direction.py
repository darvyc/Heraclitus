"""3D direction vectors on the unit sphere S^2.

Every DualFlowTransformer carries a single learned vector `d ∈ R^3` constrained
to the unit sphere. Two transformers are 'aligned' when the cosine between
their directions exceeds a threshold — equivalently, when one falls inside the
other's alignment cone.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


class Direction:
    """Helpers for unit 3-vectors. Stateless; all methods are static."""

    EPS = 1e-8

    @staticmethod
    def random(generator: torch.Generator | None = None) -> Tensor:
        """Sample uniformly from S^2 via the standard normal trick."""
        v = torch.randn(3, generator=generator)
        return Direction.normalize(v)

    @staticmethod
    def normalize(v: Tensor) -> Tensor:
        """Project v onto S^2. Safe against zero vectors."""
        n = v.norm(dim=-1, keepdim=True).clamp(min=Direction.EPS)
        return v / n

    @staticmethod
    def cosine(a: Tensor, b: Tensor) -> Tensor:
        """Cosine similarity between two (already-normalised) directions."""
        return (a * b).sum(dim=-1).clamp(-1.0, 1.0)

    @staticmethod
    def angle(a: Tensor, b: Tensor) -> Tensor:
        """Geodesic angle in radians on S^2."""
        return torch.arccos(Direction.cosine(a, b))

    @staticmethod
    def slerp(a: Tensor, b: Tensor, t: float) -> Tensor:
        """Spherical linear interpolation between two unit vectors."""
        cos = Direction.cosine(a, b).item()
        if cos > 1.0 - 1e-6:
            return Direction.normalize(a + t * (b - a))
        omega = math.acos(cos)
        sin_o = math.sin(omega)
        return (math.sin((1 - t) * omega) / sin_o) * a + (math.sin(t * omega) / sin_o) * b

    @staticmethod
    def from_features(x: Tensor, projection: Tensor) -> Tensor:
        """Project a (..., d_model) feature tensor down to a single S^2 vector.

        `projection` is a (d_model, 3) matrix. We mean-pool over all leading
        dims so that arbitrary batch / sequence shapes collapse to one
        direction per call.
        """
        flat = x.reshape(-1, x.shape[-1]).mean(dim=0)  # (d_model,)
        return Direction.normalize(flat @ projection)  # (3,)
