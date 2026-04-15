import torch
from heraclitus import DualFlowTransformer, FlowRegistry


def test_forward_shape():
    net = DualFlowTransformer(d_model=16, n_heads=4, n_layers=2)
    x = torch.randn(2, 5, 16)
    y = net(x, learn=False)
    assert y.shape == x.shape


def test_learn_creates_counter_flow():
    net = DualFlowTransformer(d_model=16, n_heads=4, n_layers=2)
    assert net.counter_flow is None
    x = torch.randn(1, 4, 16)
    net(x, learn=True)
    assert net.counter_flow is not None
    assert int(net.step) == 1


def test_direction_is_unit_norm_after_learning():
    net = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1)
    x = torch.randn(1, 4, 16)
    for _ in range(5):
        net(x, learn=True)
    assert abs(float(net.direction.norm()) - 1.0) < 1e-5


def test_counter_flow_distillation_target():
    net = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1)
    x = torch.randn(1, 4, 16)
    net(x, learn=True)
    cf = net.counter_flow
    target = cf.distillation_target(x)
    assert target is not None
    assert target.shape == x.shape


def test_keep_counter_flows_bound():
    net = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, keep_counter_flows=3)
    x = torch.randn(1, 4, 16)
    for _ in range(10):
        net(x, learn=True)
    assert len(net.counter_flows) == 3
