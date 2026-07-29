"""Geometry for 3D direction vectors on the unit sphere S^2."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


class Direction:
    """Numerically stable helpers for unit 3-vectors."""

    EPS = 1e-8

    @staticmethod
    def random(generator: Optional[torch.Generator] = None) -> Tensor:
        v = torch.randn(3, generator=generator)
        return Direction.normalize(v)

    @staticmethod
    def random_projection(
        d_model: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Return a d_model by 3 orthonormal projection frame."""
        if d_model < 3:
            raise ValueError("d_model must be at least 3 for an S^2 projection")
        matrix = torch.randn(d_model, 3, generator=generator)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        return q[:, :3]

    @staticmethod
    def normalize(v: Tensor) -> Tensor:
        """Map non-zero vectors to S^2 and map zero vectors to a fixed pole."""
        if v.shape[-1] != 3:
            raise ValueError("direction vectors must have final dimension 3")
        norm = v.norm(dim=-1, keepdim=True)
        normalised = v / norm.clamp(min=Direction.EPS)
        pole = torch.zeros_like(v)
        pole[..., 0] = 1.0
        return torch.where(norm > Direction.EPS, normalised, pole)

    @staticmethod
    def cosine(a: Tensor, b: Tensor) -> Tensor:
        a_unit = Direction.normalize(a)
        b_unit = Direction.normalize(b)
        return (a_unit * b_unit).sum(dim=-1).clamp(-1.0, 1.0)

    @staticmethod
    def angle(a: Tensor, b: Tensor) -> Tensor:
        """Stable geodesic angle in radians on S^2."""
        a_unit = Direction.normalize(a)
        b_unit = Direction.normalize(b)
        cross_norm = torch.cross(a_unit, b_unit, dim=-1).norm(dim=-1)
        dot = (a_unit * b_unit).sum(dim=-1)
        return torch.atan2(cross_norm, dot)

    @staticmethod
    def slerp(a: Tensor, b: Tensor, t: float) -> Tensor:
        """Spherical interpolation, including the antipodal case."""
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must lie in [0, 1]")
        a_unit = Direction.normalize(a)
        b_unit = Direction.normalize(b)
        cos = float(Direction.cosine(a_unit, b_unit))
        if cos > 1.0 - 1e-6:
            return Direction.normalize(a_unit + t * (b_unit - a_unit))
        if cos < -1.0 + 1e-6:
            axis = int(torch.argmin(a_unit.abs()).item())
            basis = torch.zeros_like(a_unit)
            basis[axis] = 1.0
            orthogonal = Direction.normalize(
                basis - torch.dot(basis, a_unit) * a_unit
            )
            return Direction.normalize(
                math.cos(math.pi * t) * a_unit
                + math.sin(math.pi * t) * orthogonal
            )
        omega = math.acos(cos)
        sin_omega = math.sin(omega)
        return Direction.normalize(
            (math.sin((1.0 - t) * omega) / sin_omega) * a_unit
            + (math.sin(t * omega) / sin_omega) * b_unit
        )

    @staticmethod
    def from_features(x: Tensor, projection: Tensor) -> Tensor:
        if projection.shape != (x.shape[-1], 3):
            raise ValueError(
                f"projection must have shape ({x.shape[-1]}, 3), "
                f"got {tuple(projection.shape)}"
            )
        pooled = x.reshape(-1, x.shape[-1]).mean(dim=0)
        return Direction.normalize(pooled @ projection)
