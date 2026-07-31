# Heraclitus 3 architecture

Heraclitus 3 is one specific system: a bounded recurrent slot-memory adapter. This document is normative for the 3.0.0 implementation.

## Components

`HeraclitusConfig` defines hidden width, state width, slot count, attention heads, retention limits, sparse-write controls, usage bounds, dropout, and the strict residual ratio.

`HeraclitusAdapter` owns the read, write, erase, candidate, and output projections. It transforms `(B, T, D)` hidden states into the same shape.

`HeraclitusState` contains a float state bank `(B, M, R)`, usage `(B, M)`, and valid-token counts `(B,)`.

`HeraclitusOutput` contains transformed hidden states, continuation state, regularisation terms, and diagnostics.

## Causal order

At token `t`, the adapter reads incoming state, computes its output, then writes the current observation. A token cannot read its own write. Future tokens cannot affect past outputs.

## Read path

The normalised host hidden state is projected to one query per head. Each memory slot is projected to one key and value per head. Scaled dot-product attention retrieves a state-sized vector.

## Residual path

The retrieved vector is projected to hidden width. A learned scalar gate and dropout are applied. The resulting vector is then projected onto the norm ball

```text
norm(delta_t) <= max_residual_ratio * norm(h_t).
```

The projection is downstream of all trainable matrices, so the bound cannot be defeated by growing projection weights.

## Write path

The candidate and erase controller both receive `[normalised_hidden, retrieved_memory]`. This allows recalled state to participate directly in the next state transition.

Allocation combines cosine novelty with decayed usage pressure. Temperature controls sharpness. Only the largest `write_topk` logits remain active before softmax, giving exact sparse slot selection.

Each selected slot receives a gated erase-and-add update. Slot values are clamped to `[-1, 1]`. Usage decays with a factor strictly below one and is clamped to `max_usage`.

## Numerical policy

Internal projections use float32 functional linear operations, even when module parameters are float16 or bfloat16. Outputs are cast back to the host hidden dtype. Continuation state is normally retained in float32.

## Complexity

For batch `B`, sequence `T`, hidden width `D`, state width `R`, and slots `M`, the dominant reference cost is approximately:

```text
O(B T (D R + M R^2 + M R))
```

State memory is `O(B M R)`. The Python reference is sequential across tokens. One-token decoding is its natural execution mode; long prefill should be benchmarked and may require a compiled or fused implementation.

## Guarantees and non-guarantees

Tests enforce causality, chunk equivalence in deterministic mode, exact initial output identity, masking, batch isolation, serialisation, sparse writes, finite gradients, reduced-precision execution, bounded slot values, bounded usage, and a strict residual norm ratio.

The architecture does not guarantee improved language modelling, lossless history, indefinite retention, or competitive throughput. Those require empirical evaluation.
