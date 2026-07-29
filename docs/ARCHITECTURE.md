# Heraclitus Architecture

This document describes the mechanisms implemented by the current research prototype. It does not claim that those mechanisms improve task performance. See `MATHEMATICAL_AUDIT.md` for the critical assessment, formal assumptions, unresolved problems, and minimum experimental standard.

## 1. Direction-conditioned attention

Each block begins with ordinary scaled dot-product attention. For batch b, head h, query token i, and key token j:

    base_score[b,h,i,j] = q[b,h,i] dot k[b,h,j] / sqrt(d_head)

Each key token is also mapped to a learned unit 3-vector `a[b,h,j]`. Given the model's live unit direction `d`, the final score is:

    score[b,h,i,j] = base_score[b,h,i,j]
                       + direction_scale[h] * a[b,h,j] dot d

The directional term varies over key index j, which is the softmax normalisation axis. This is essential. Adding one constant to an entire attention row would have no effect because softmax is translation-invariant.

The merged attention context is returned internally so the local update for the output projection uses the tensor that actually enters that projection.

## 2. Local forward adaptation

When `forward(..., learn=True)` is called, each block applies two updates under `torch.no_grad()`.

### Attention output projection

For the output projection `y = W x`, the code uses a bounded multi-output Oja rule:

    Delta W = E[y x^T] - diag(E[y^2]) W

The Oja term opposes unconstrained Hebbian norm growth. The resulting parameter displacement is clipped to a configured fraction of the current weight norm.

### MLP output bias

Let `y_bar` be the current mean output and `m` its exponential moving target. The local additive-bias update is:

    b_next = b - eta * (y_bar - m)
    m_next = rho * m + (1 - rho) * y_bar

The subtraction sign follows gradient descent on local squared prediction error.

These are local adaptation rules. They are not a substitute for a task-level objective, and the repository does not presently contain a convergence theorem for the coupled dynamics.

## 3. Sphere-valued state

Each transformer stores a unit direction `d` on `S^2`.

A projection frame `P` with shape `(d_model, 3)` maps pooled hidden features to a target direction:

    target = normalise(mean(features) P)

The state update is:

    d_next = normalise(momentum * d + (1 - momentum) * target)

Transformers in the same `FlowRegistry` and with the same hidden width receive the same orthonormal projection frame. This removes the obvious error of comparing coordinates produced by independently rotated 3D frames.

A shared projection frame does not guarantee semantic comparability when hidden feature spaces themselves are unaligned. The `orthogonal_procrustes` utility can fit a proper 3D rotation from shared anchor observations, but users must still supply and validate those anchors.

## 4. Peer discovery and null calibration

`FlowRegistry.query_aligned` returns peers satisfying:

    d_source dot d_target >= threshold

For independent uniform directions on `S^2`:

    P(random match) = (1 - threshold) / 2

With N registry members, the expected random out-degree is:

    (N - 1) * (1 - threshold) / 2

The registry exposes both this null expectation and its inverse threshold calculation. A reported threshold is scientifically incomplete unless registry size and the implied random degree are also reported.

Connections are currently records only. They do not transmit activations, aggregate messages, or alter predictions.

## 5. Counter-Flow snapshots

Before a local update, the model stores:

- step number
- previous direction
- cloned state dictionary
- connection identifiers
- metadata
- a clean frozen architecture clone for distillation queries

The frozen clone is created without a registry and without historical Counter-Flows. This prevents recursive copying of the swarm and its snapshot history.

Retaining K full snapshots of a P-parameter model still costs O(KP) storage. Delta checkpoints or low-rank parameter differences are required for scale.

Counter-Flows are not yet included in an explicit regularisation term. At present they provide auditability and a callable frozen target, not a complete dual-flow optimisation law.

## 6. Module map

```text
heraclitus/
├── core.py              DualFlowTransformer and update lifecycle
├── attention.py         key-dependent direction-conditioned attention
├── forward_learner.py   bounded Oja and local prediction-error updates
├── direction.py         stable S^2 geometry
├── mathematics.py       null model and Procrustes frame alignment
├── counter_flow.py      frozen snapshots
├── network.py           registry, shared frame, calibrated search
├── connections.py       directed edge records
└── utils.py             identifiers
```

## 7. Update lifecycle

1. `net(x, learn=True)` is called.
2. A clean frozen Counter-Flow snapshot is created from the pre-update state.
3. Each transformer block computes attention and MLP outputs.
4. The attention output projection receives the bounded Oja update using its true input and output activities.
5. The MLP output bias descends its local prediction error.
6. The final hidden features update the unit direction through the shared frame.
7. The step counter increments.
8. A caller may scan the registry and create similarity-edge records using a calibrated threshold.

## 8. Non-claims

The current code does not establish that:

- three dimensions preserve useful model-state information
- direction cosine measures hypothesis similarity
- local updates improve any task
- Counter-Flows prevent forgetting
- graph connections enable cooperation
- the coupled dynamics converge

Those questions require the experiments and baselines specified in `MATHEMATICAL_AUDIT.md`.
