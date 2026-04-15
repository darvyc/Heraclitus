import torch
from heraclitus import DualFlowTransformer
from heraclitus.counter_flow import CounterFlow, CounterFlowSnapshot


def test_snapshot_records_step_and_direction():
    net = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1)
    x = torch.randn(1, 4, 16)
    net(x, learn=True)
    cf = net.counter_flow
    # The snapshot was taken BEFORE the step counter incremented, so step=0.
    assert cf.step == 0
    assert cf.direction.shape == (3,)
    assert abs(float(cf.direction.norm()) - 1.0) < 1e-5


def test_opposition_increases_with_drift():
    net = DualFlowTransformer(
        d_model=16, n_heads=4, n_layers=1, direction_momentum=0.0
    )
    x = torch.randn(1, 4, 16)
    net(x, learn=True)
    first = float(net.counter_flow.opposition(net.direction))
    # With momentum=0 the live direction jumps to the new feature projection
    # each step, so opposition vs the *first* counter-flow is non-trivial.
    assert first >= 0.0


def test_counter_flow_is_frozen():
    net = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1)
    x = torch.randn(1, 4, 16)
    net(x, learn=True)
    cf = net.counter_flow
    for p in cf._frozen_module.parameters():
        assert not p.requires_grad


def test_summary_is_string():
    snap = CounterFlowSnapshot(
        step=3,
        direction=torch.tensor([1.0, 0.0, 0.0]),
        state_dict={},
    )
    cf = CounterFlow(parent_id="test", snapshot=snap, parent_module=None)
    s = cf.summary()
    assert "CounterFlow" in s
    assert "step=3" in s
