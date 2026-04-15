# Heraclitus — Architecture

> *"The road up and the road down are one and the same."* — Heraclitus, Fragment 60

This document describes the four behaviours required of a Dual-Flow Transformer
and how each maps onto the modules in this repository.

## 1. Forward-pass learning

The transformer adapts its parameters during the forward pass, with no
backward pass required. Two local, gradient-free rules are applied at each
block during `forward(..., learn=True)`:

* **Hebbian update** on the attention output projection
  `W_o`: `ΔW ∝ ⟨post · preᵀ⟩ / N`, normalised so a single update cannot
  destabilise the weight scale (`forward_learner.py::hebbian_update`).
* **Predictive-coding update** on the MLP output bias: each block keeps an
  EMA of its recent output; the difference between the current output and the
  EMA — the "surprise" — drives an additive bias correction
  (`forward_learner.py::predictive_update`).

Because both rules use only locally available statistics and run inside
`torch.no_grad()`, a `DualFlowTransformer` can be embedded inside a normal
autograd graph without interfering with it. Setting `learn=False` reduces the
module to a vanilla transformer.

## 2. The 3D direction vector

Each transformer carries a single learned unit vector `d ∈ S²` representing
the orientation of its current hypothesis in a shared latent space.

* `direction.py` provides the S² utilities (`random`, `normalize`, `cosine`,
  `angle`, `slerp`, `from_features`).
* `core.py` holds the live direction in a registered buffer and updates it
  via momentum-EMA after every learning forward pass:
  `d ← normalize(m · d + (1 − m) · target)`,
  where `target = normalize(mean(features) · W_dir)`.
* The same direction also modulates attention: in
  `attention.py::DirectionModulatedAttention` each head carries a learned
  3-axis, and the head's attention logits receive an additive bias equal to
  the dot product of that axis with the live direction. This means the
  transformer's S² orientation continuously shapes what it attends to,
  without re-allocating any structural capacity.

## 3. Scanning and connection

After each update the transformer can scan the shared `FlowRegistry` for
peers whose direction now lies inside its alignment cone, and forge weighted
`FlowConnection`s with them.

* `network.py::FlowRegistry` is an in-process directory of every live
  transformer. Its `query_aligned(d, threshold)` method returns peers whose
  cosine with `d` exceeds the threshold, sorted by descending alignment.
* `core.py::DualFlowTransformer.scan_and_connect()` calls the registry,
  builds `FlowConnection` records (with weight = cosine at forge-time and a
  snapshot of the source's direction for audit), and stores them on the
  transformer's `.connections` dict.
* The `top_k` and `replace` flags let callers cap the fan-out and decide
  whether each scan augments or rebuilds the connection set.

The registry is intentionally a brute-force dict; a kd-tree or HNSW index
could be substituted without changing the public API.

## 4. The Counter-Flow

Before any update, the transformer's prior incarnation is frozen as a
`CounterFlow` derivative — a backwards-pointing shadow.

* `counter_flow.py::CounterFlowSnapshot` records `{step, direction,
  state_dict, connection_ids, metadata}`.
* `CounterFlow` wraps that snapshot together with a deep-copied,
  parameter-frozen replica of the parent module. It exposes:
    - `direction` — the prior S² orientation;
    - `distillation_target(x)` — runs the frozen prior on `x` for use as a
      soft target / regulariser;
    - `opposition(live_direction)` — geodesic angle on S² between the live
      direction and this past one, a scalar measure of how sharp the most
      recent revision was;
    - `summary()` — a one-line audit string.
* The transformer keeps a bounded chain of Counter-Flows
  (`keep_counter_flows`); the most recent is exposed as `.counter_flow` and
  the full chain as `.counter_flows`.

Together, the live network and the chain of Counter-Flows realise the
Heraclitean image: the river flows forward, while every prior moment of the
same river is preserved, frozen, and addressable.

## Module map

```
heraclitus/
├── core.py              DualFlowTransformer (the headline module)
├── attention.py         DirectionModulatedAttention
├── forward_learner.py   ForwardLearner (Hebbian + predictive-coding rules)
├── direction.py         Direction (S² utilities)
├── counter_flow.py      CounterFlow + CounterFlowSnapshot
├── network.py           FlowRegistry
├── connections.py       FlowConnection
└── utils.py             new_flow_id
```

## Lifecycle of one update

1. `net(x, learn=True)` is called.
2. `_snapshot_counter_flow()` deep-copies the current state into a new
   `CounterFlow` and appends it to `net.counter_flows`.
3. The stack runs. For each block, after the standard attention + MLP, the
   Hebbian and predictive-coding rules update `W_o` and the MLP bias in place.
4. `_update_direction(out)` slides the live S² direction toward the
   projection of the latest features.
5. `net.step` is incremented.
6. The caller invokes `net.scan_and_connect(threshold=...)`. The registry is
   queried, new `FlowConnection`s are forged, and the connection map is
   updated.
