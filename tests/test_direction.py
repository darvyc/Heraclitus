import math
import torch
from heraclitus import Direction


def test_random_is_unit_norm():
    for _ in range(10):
        v = Direction.random()
        assert abs(float(v.norm()) - 1.0) < 1e-5


def test_normalize_zero_safe():
    v = Direction.normalize(torch.zeros(3))
    assert torch.isfinite(v).all()


def test_cosine_bounds():
    a = Direction.random()
    b = Direction.random()
    c = float(Direction.cosine(a, b))
    assert -1.0 <= c <= 1.0


def test_cosine_self_is_one():
    a = Direction.random()
    assert abs(float(Direction.cosine(a, a)) - 1.0) < 1e-5


def test_angle_orthogonal():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    angle = float(Direction.angle(a, b))
    assert abs(angle - math.pi / 2) < 1e-5


def test_slerp_endpoints():
    a = Direction.random()
    b = Direction.random()
    s0 = Direction.slerp(a, b, 0.0)
    s1 = Direction.slerp(a, b, 1.0)
    assert torch.allclose(Direction.normalize(s0), a, atol=1e-5)
    assert torch.allclose(Direction.normalize(s1), b, atol=1e-5)


def test_from_features_shape_and_norm():
    x = torch.randn(2, 5, 16)
    proj = torch.randn(16, 3)
    d = Direction.from_features(x, proj)
    assert d.shape == (3,)
    assert abs(float(d.norm()) - 1.0) < 1e-5
