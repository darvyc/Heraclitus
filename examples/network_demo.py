"""Swarm demo: many transformers, shared registry, alignment-based wiring."""
import torch
from heraclitus import DualFlowTransformer, FlowRegistry

torch.manual_seed(42)

registry = FlowRegistry()
swarm = [
    DualFlowTransformer(d_model=32, n_heads=4, n_layers=2, registry=registry)
    for _ in range(6)
]
print(f"swarm size: {len(registry)}")

x = torch.randn(2, 8, 32)

# One round of forward-pass learning across the swarm.
for net in swarm:
    net(x, learn=True)

# Each transformer scans the registry and wires up to its alignment cone.
for net in swarm:
    edges = net.scan_and_connect(threshold=0.5, top_k=3)
    print(f"{net.flow_id} forged {len(edges)} edges:")
    for e in edges:
        print(f"   {e}")
