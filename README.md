# Heraclitus

> *"πάντα ῥεῖ" - everything flows.*
>
> *"The road up and the road down are one and the same."* - Heraclitus, Fragment 60

Heraclitus is an experimental 3D Dual-Flow Transformer research prototype. It combines inline local adaptation, a sphere-valued state vector, direction-conditioned attention, similarity-based peer discovery, and frozen prior-state snapshots called Counter-Flows.

## Research status

This repository is not a validated learning method. It currently provides mechanisms and mathematical diagnostics that must still be tested against task-level objectives, strong baselines, repeated seeds, uncertainty estimates, and ablations.

Read `docs/MATHEMATICAL_AUDIT.md` before treating any architectural claim as established. That document states what the model now implements, what was mathematically defective in the initial version, and what remains absent.

## Core mechanisms

| Mechanism | Current implementation |
|---|---|
| Forward adaptation | A bounded multi-output Oja update on the attention output projection and a local squared-error descent step on the MLP output bias. |
| 3D state | Each transformer carries a unit vector `d` on `S^2`, updated from pooled hidden features through a registry-shared projection frame. |
| Direction-conditioned attention | Each key token receives a direction-dependent logit bias that varies along the softmax axis. |
| Peer discovery | A `FlowRegistry` finds peers above a cosine threshold and exposes an exact random-alignment null model for threshold calibration. |
| Counter-Flow | Each update stores a clean, frozen prior model clone without recursively copying the registry or earlier snapshots. |
| Frame alignment | An orthogonal Procrustes utility aligns 3D frames from shared anchor observations. |

## Important limitations

Connections currently record graph edges but do not exchange activations or messages. Counter-Flows provide frozen targets but are not yet included in an explicit regularisation objective. A shared projection frame does not by itself guarantee that independently trained hidden spaces have the same semantic basis. Three dimensions may be too severe a bottleneck.

## Install

```bash
pip install -e .
```

Requires Python 3.9 or later and PyTorch 2.0 or later.

## Quickstart

```python
import torch
from heraclitus import DualFlowTransformer, FlowRegistry

registry = FlowRegistry(frame_seed=7)
nets = [
    DualFlowTransformer(d_model=64, n_heads=4, n_layers=2, registry=registry)
    for _ in range(3)
]

x = torch.randn(2, 16, 64)
for net in nets:
    y = net(x, learn=True)

# Calibrate a threshold to one expected random match in this registry.
threshold = registry.threshold_for_null_degree(expected_degree=1.0)
for net in nets:
    net.scan_and_connect(threshold=threshold)

print(nets[0].counter_flow.summary())
```

## Repository layout

```text
heraclitus/
├── heraclitus/
├── examples/
├── tests/
└── docs/
    ├── ARCHITECTURE.md
    └── MATHEMATICAL_AUDIT.md
```

## Testing

```bash
pytest -q
```

The mathematical regression tests verify that direction changes attention, the shared frame is actually shared, the predictive update descends its declared local error, antipodal sphere interpolation is finite, Procrustes alignment recovers a known rotation, and Counter-Flow snapshots do not recursively copy registry history.

## License

MIT.
