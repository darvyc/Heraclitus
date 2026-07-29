import torch

from heraclitus import (
    bounded_frobenius,
    geodesic_fraction,
    orthogonal_procrustes,
    orthogonality_error,
    safe_unit,
    spherical_ema,
)


def test_safe_unit_zero_has_unit_norm():
    value = safe_unit(torch.zeros(3))
    assert torch.allclose(value.norm(), torch.tensor(1.0))


def test_bounded_frobenius_enforces_bound():
    weight = torch.randn(8, 4) * 100.0
    bounded = bounded_frobenius(weight, max_norm=2.0)
    assert float(bounded.norm(p="fro")) <= 2.0 + 1e-5


def test_spherical_ema_stays_on_sphere():
    previous = safe_unit(torch.randn(3, 5))
    observation = safe_unit(torch.randn(3, 5))
    updated = spherical_ema(previous, observation, decay=0.9, epsilon=1e-6)
    assert torch.allclose(updated.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_geodesic_fraction_endpoints():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(geodesic_fraction(a, a), torch.zeros(1))
    assert torch.allclose(geodesic_fraction(a, -a), torch.ones(1))


def test_orthogonality_error_zero_for_frame():
    frame, _ = torch.linalg.qr(torch.randn(12, 4), mode="reduced")
    assert float(orthogonality_error(frame)) < 1e-10


def test_procrustes_recovers_rotation_in_general_dimension():
    source = torch.randn(12, 4)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    if torch.det(rotation) < 0:
        rotation[:, -1] *= -1
    target = source @ rotation
    fitted, singular_values = orthogonal_procrustes(source, target)
    assert singular_values.shape == (4,)
    assert torch.allclose(source @ fitted, target, atol=1e-5, rtol=1e-5)
    assert float(torch.det(fitted)) > 0.0
