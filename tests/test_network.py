import torch
from heraclitus import DualFlowTransformer, FlowRegistry


def test_register_and_count():
    reg = FlowRegistry()
    a = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    b = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    assert len(reg) == 2
    assert a.flow_id in reg.ids()
    assert b.flow_id in reg.ids()


def test_query_excludes_self():
    reg = FlowRegistry()
    a = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    matches = reg.query_aligned(a.direction, threshold=-1.0, exclude=a.flow_id)
    assert all(m[0].flow_id != a.flow_id for m in matches)


def test_scan_and_connect_threshold_filter():
    reg = FlowRegistry()
    a = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    b = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    # Force directions to be nearly anti-parallel.
    a.direction = torch.tensor([1.0, 0.0, 0.0])
    b.direction = torch.tensor([-1.0, 0.0, 0.0])
    edges = a.scan_and_connect(threshold=0.9)
    assert edges == []


def test_scan_and_connect_finds_aligned_peer():
    reg = FlowRegistry()
    a = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    b = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    a.direction = torch.tensor([1.0, 0.0, 0.0])
    b.direction = torch.tensor([1.0, 0.0, 0.0])
    edges = a.scan_and_connect(threshold=0.9)
    assert len(edges) == 1
    assert edges[0].target_id == b.flow_id
    assert edges[0].weight > 0.99


def test_replace_clears_existing():
    reg = FlowRegistry()
    a = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    b = DualFlowTransformer(d_model=16, n_heads=4, n_layers=1, registry=reg)
    a.direction = torch.tensor([1.0, 0.0, 0.0])
    b.direction = torch.tensor([1.0, 0.0, 0.0])
    a.scan_and_connect(threshold=0.9)
    assert len(a.connections) == 1
    a.scan_and_connect(threshold=0.9, replace=True)
    assert len(a.connections) == 1  # one fresh edge, not two
