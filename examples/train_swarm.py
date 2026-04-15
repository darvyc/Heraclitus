"""Train a swarm on a toy task, alternating forward learning and re-wiring.

This demo is illustrative — the gradient-free updates here will not match a
back-prop baseline, but they show the full lifecycle: forward learning,
direction drift, Counter-Flow snapshots, and dynamic re-wiring.
"""
import torch
from heraclitus import DualFlowTransformer, FlowRegistry

torch.manual_seed(7)

registry = FlowRegistry()
swarm = [
    DualFlowTransformer(d_model=32, n_heads=4, n_layers=2, registry=registry)
    for _ in range(4)
]

# Each net sees a slightly different data stream so their directions diverge.
streams = [torch.randn(8, 8, 32) + 0.1 * torch.randn(1, 1, 32) for _ in swarm]

for epoch in range(5):
    for net, stream in zip(swarm, streams):
        net(stream, learn=True)
    # Re-wire after every epoch.
    for net in swarm:
        net.scan_and_connect(threshold=0.6, top_k=2, replace=True)

    print(f"\n-- epoch {epoch} --")
    for net in swarm:
        d = [round(v, 2) for v in net.direction.tolist()]
        opp = float(net.counter_flow.opposition(net.direction)) if net.counter_flow else 0.0
        print(
            f"{net.flow_id} dir={d} "
            f"|conn|={len(net.connections)} opposition={opp:.3f}rad "
            f"|counter|={len(net.counter_flows)}"
        )
