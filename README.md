# Heraclitus

Heraclitus is a causal, state-conditioned, low-rank parameter for transformer language models. It adds a bounded adaptive residual to an existing hidden stream while preserving standard gradient training, explicit sequence state, chunked generation, masking, mixed precision, and model serialisation.

## Install

```bash
pip install -e .
```

Heraclitus requires Python 3.9 or later and PyTorch 2.0 or later.

## LLM integration

```python
import torch
from heraclitus import HeraclitusConfig, HeraclitusParameter

hidden_size = 4096
heraclitus = HeraclitusParameter(
    HeraclitusConfig(
        hidden_size=hidden_size,
        state_size=64,
        max_residual_scale=0.10,
    )
)

hidden_states = torch.randn(2, 128, hidden_size)
attention_mask = torch.ones(2, 128, dtype=torch.bool)

result = heraclitus.forward_with_state(
    hidden_states,
    attention_mask=attention_mask,
)

hidden_states = result.hidden_states
auxiliary_loss = result.regularization_loss()
```

Place one `HeraclitusParameter` after the attention residual, after the MLP residual, or once at the end of each transformer block. Its input and output both have shape `(batch, sequence, hidden_size)`.

## Stateful generation

```python
state = None
outputs = []

for hidden_chunk in hidden_chunks:
    result = heraclitus.forward_with_state(hidden_chunk, state=state)
    outputs.append(result.hidden_states)
    state = result.state
```

The continuation state contains one live flow, one counter-flow, and one valid-token counter per sequence. It can be detached, cloned, moved between devices, converted to a tensor dictionary, saved, and restored for continued generation.

## Mathematical contract

For hidden width `D` and state width `R`, Heraclitus uses exactly:

```text
2 * D * R + R + 3
```

trainable scalar parameters. Runtime state uses `2 * B * R + B` scalars for batch size `B`.

The parameter satisfies these invariants:

- Strict causality: output token `t` depends only on tokens `0` through `t` and state accumulated before `t`.
- Batch isolation: one sequence cannot alter another sequence's state.
- Unit state: live flow and counter-flow remain on the unit sphere.
- Bounded adaptation: effective projection and reconstruction matrices are constrained to norm balls and residual gain is bounded.
- Autograd safety: forward execution never mutates trainable parameters.
- Chunk equivalence: processing a sequence in chunks with the returned state matches processing it in one call when dropout is disabled.
- Mask correctness: masked tokens are unchanged and do not advance state.

The full equations, bounds, and proof sketches are in `docs/MATHEMATICS.md`. Integration patterns are in `docs/INTEGRATION.md`.

## Testing

```bash
pytest -q
```

The test suite verifies causality, chunk equivalence, batch isolation, gradients, masking, mixed precision, serialisation, state geometry, bounded matrices, and mathematical utilities.

## License

MIT.
