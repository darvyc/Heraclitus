import math

import torch

from heraclitus import (
    Direction,
    DualFlowTransformer,
    FlowRegistry,
    expected_random_degree,
    orthogonal_procrustes,
    random_alignment_probability,
    threshold_for_expected_random_degree,
)
from heraclitus.attention import DirectionModulatedAttention
from heraclitus.forward_learner import ForwardLearner


def test_direction_changes_attention_output():
    torch.manual_seed(0)
    attention = DirectionModulatedAttention(d_model=12, n_heads=3).eval()
    x = torch.randn(2, 5, 12)
    positive = attention(x, torch.tensor([1.0, 0.0, 0.0]))
    negative = attention(x, torch.tensor([-1.0, 0.0, 0.0]))
    assert not torch.allclose(positive, negative)


def test_registry_members_share_projection_frame():
    registry = FlowRegistry(frame_seed=7)
    a = DualFlowTransformer(d_model=12, n_heads=3, n_layers=1, registry=registry)
    b = DualFlowTransformer(d_model=12, n_heads=3, n_layers=1, registry=registry)
    assert torch.equal(a.direction_projection, b.direction_projection)


def test_s2_null_model_and_inverse():
    assert math.isclose(random_alignment_probability(0.7), 0.15)
    assert math.isclose(expected_random_degree(101, 0.7), 15.0)
    threshold = threshold_for_expected_random_degree(101, 1.0)
    assert math.isclose(threshold, 0.98)


def test_predictive_update_descends_additive_squared_error():
    learner = ForwardLearner(lr=0.1)
    bias = torch.nn.Parameter(torch.zeros(3))
    output = torch.tensor([[2.0, -1.0, 0.5]])
    ema = torch.zeros(3)
    before = (output.mean(dim=0) - ema).norm()
    learner.predictive_update(bias, output, ema)
    after_output = output + bias
    after = (after_output.mean(dim=0) - ema).norm()
    assert after < before


def test_antipodal_slerp_is_finite_and_unit_norm():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = -a
    midpoint = Direction.slerp(a, b, 0.5)
    assert torch.isfinite(midpoint).all()
    assert torch.allclose(midpoint.norm(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(torch.dot(midpoint, a))) < 1e-6


def test_procrustes_recovers_rotation():
    source = torch.eye(3)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    target = source @ rotation
    fitted, singular_values = orthogonal_procrustes(source, target)
    assert torch.allclose(fitted, rotation, atol=1e-6)
    assert torch.all(singular_values > 0)


def test_counter_flow_clone_does_not_copy_registry_or_history():
    registry = FlowRegistry(frame_seed=1)
    net = DualFlowTransformer(d_model=12, n_heads=3, n_layers=1, registry=registry)
    x = torch.randn(1, 4, 12)
    net(x, learn=True)
    frozen = net.counter_flow._frozen_module
    assert frozen.registry is None
    assert frozen.counter_flows == []
