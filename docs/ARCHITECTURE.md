# Heraclitus architecture

Heraclitus is an LLM parameter subsystem. It receives a transformer hidden stream, applies a causal state-conditioned low-rank residual, and returns a hidden stream of the same shape together with explicit continuation state and training diagnostics.

## Components

### HeraclitusConfig

`HeraclitusConfig` fixes hidden width, state width, state time constants, norm bounds, residual gain, gate temperature, opposition strength, dropout, and numerical epsilon. Construction validates every domain constraint.

### HeraclitusParameter

`HeraclitusParameter` owns all trainable tensors:

```text
projection:           D x R
reconstruction:       R x D
state_seed:           R
gate_bias:            1
residual_scale_logit: 1
opposition_logit:     1
```

The module transforms `(B, T, D)` to `(B, T, D)`. It can be inserted at any residual boundary in a transformer block.

### HeraclitusState

`HeraclitusState` contains:

```text
live:    B x R
counter: B x R
steps:   B
```

State is external to the module. It can be detached, cloned, moved, selected, reordered, checkpointed, and restored. This keeps sequence history isolated from trainable model parameters and gives host inference engines direct control over batching and beam order.

### HeraclitusOutput

`HeraclitusOutput` returns:

- transformed hidden states
- continuation state
- regularisation terms
- diagnostics

Its `regularization_loss` method combines the auxiliary terms with explicit weights.

## Data path

For each token:

1. RMS-normalise the hidden vector in float32.
2. Project it into the state space through a norm-bounded matrix.
3. Convert the projected vector to a unit observation.
4. Form a dual-flow direction from live and counter states.
5. Compute a bounded scalar gate from observation alignment.
6. Separate and amplify the latent component aligned with the dual flow.
7. Reconstruct a low-rank residual through a second norm-bounded matrix.
8. Add the residual to the original hidden vector.
9. Update counter state from the prior live state.
10. Update live state from the current observation.

Output computation precedes the state update. The token therefore cannot use information from any future token.

## Training semantics

Trainable tensors are changed only by the host optimiser. Forward execution performs no in-place parameter update. The live and counter states are tensors in the autograd graph during the current call; returned state is detached by default to bound graph lifetime across generation chunks.

The module supplies four auxiliary losses:

- projection-frame orthogonality
- live/counter consistency
- state drift
- residual energy

These combine with the host language-model objective through `regularization_loss`.

## Inference semantics

Each active sequence owns one state row. Chunked decoding passes returned state into the next call. Dynamic batching and beam search reorder state through `HeraclitusState.index_select` using the same indices applied to the host model cache.

Masked tokens are exact identity operations for Heraclitus and do not advance state.

## Numerical semantics

Geometry and low-rank products execute in float32. The residual is cast back to the hidden-stream dtype before addition. Effective matrices are differentiably projected into configured norm balls on every call. Live state, counter state, and observations use deterministic zero-vector fallbacks.

## Package map

```text
heraclitus/
├── __init__.py       public API and version
├── config.py         validated immutable configuration
├── mathematics.py    bounded linear algebra and spherical geometry
├── parameter.py      LLM parameter, output, and diagnostics
├── state.py          explicit continuation state
└── py.typed           typing marker
```

## Release contract

The release contract is enforced by tests for:

- output shape and dtype
- parameter-count exactness
- parameter immutability during forward execution
- gradient flow
- strict causality
- full-sequence and chunk equivalence
- batch isolation
- mask semantics
- state reordering and checkpointing
- state-dictionary round trips
- float16 execution
- finite nonnegative auxiliary losses
- unit-sphere state transitions
- effective matrix bounds
- per-token residual bounds
