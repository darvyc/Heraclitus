# Heraclitus

> *"πάντα ῥεῖ" — everything flows.*
>
> *"The road up and the road down are one and the same."* — Heraclitus, Fragment 60

**Heraclitus** is a reference implementation of a **3D Dual-Flow Transformer**: a transformer that (1) learns during its forward pass, (2) carries a 3D directional vector that represents the orientation of its current hypothesis in a shared latent space, (3) discovers and forms connections with peer transformers whose directions align with its newly trained orientation, and (4) preserves its prior parametric self as a *Counter-Flow* — a frozen, time-shifted derivative that flows against the live network.

The name is taken from Heraclitus of Ephesus, whose doctrine of *unity of opposites* and ever-flowing river (`πάντα ῥεῖ`) is the conceptual ancestor of every dual-flow / opposing-current architecture in the literature. The forward flow is the river; the Counter-Flow is the same river an instant ago.

## Core ideas

| Concept | Implementation |
|---|---|
| **Forward-pass learning** | A local, gradient-free Hebbian + predictive-coding update applied inline during `forward()`. No backward pass is required for adaptation. |
| **3D direction vector** | Each `DualFlowTransformer` carries a learned unit vector `d ∈ S²` summarising its current representational orientation. |
| **Network-wide scanning** | A lightweight `FlowRegistry` indexes every live transformer by its `d`. After each update, the transformer queries the registry for peers whose directions fall inside an alignment cone and forms weighted `FlowConnection`s. |
| **Counter-Flow** | Before any update, the prior weights, direction, and connection map are snapshot-frozen as a `CounterFlow` derivative. Counter-Flows form a backwards-pointing shadow network that can be queried, distilled from, or used as a regulariser. |

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Quickstart

```python
import torch
from heraclitus import DualFlowTransformer, FlowRegistry

registry = FlowRegistry()

# Build three transformers and register them in a shared flow network.
nets = [
    DualFlowTransformer(d_model=64, n_heads=4, n_layers=2, registry=registry)
    for _ in range(3)
]

x = torch.randn(2, 16, 64)
for net in nets:
    y = net(x, learn=True)            # forward-pass learning + direction update
    net.scan_and_connect(threshold=0.7)  # forge connections to aligned peers

# The previous incarnation of net[0] is preserved as a Counter-Flow:
print(nets[0].counter_flow.summary())
```

See `examples/` for a runnable swarm demo.

## Repository layout

```
heraclitus/
├── heraclitus/        # library source
├── examples/          # runnable demos
├── tests/             # pytest suite
└── docs/ARCHITECTURE.md
```

## Citation

If you use Heraclitus in academic work, please cite the architecture document
in `docs/ARCHITECTURE.md`.

## License

MIT.
