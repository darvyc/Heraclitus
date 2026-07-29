# Heraclitus

Heraclitus is a compact recurrent memory adapter for transformer hidden states. It preserves a bounded bank of learned memory slots across chunks, retrieves relevant state before each token, and writes new information through novelty-aware gated updates.

The module is causal, batch-isolated, serialisable, reorderable for beam search, safe to initialise as an exact no-op, and suitable for streaming inference or context-window extension experiments.

## Install

```bash
pip install -e .
```

Python 3.9 or later and PyTorch 2.0 or later are required.

## Basic use

```python
import torch
from heraclitus import HeraclitusAdapter, HeraclitusConfig

adapter = HeraclitusAdapter(
    HeraclitusConfig(
        hidden_size=4096,
        state_size=128,
        memory_slots=8,
        num_heads=4,
    )
)

hidden = torch.randn(2, 256, 4096)
result = adapter.forward_with_state(hidden)
adapted = result.hidden_states
state = result.state

next_hidden = torch.randn(2, 128, 4096)
continued = adapter.forward_with_state(next_hidden, state=state)
loss = task_loss + continued.regularization_loss()
```

## Architecture

For each valid token, Heraclitus:

1. normalises the host hidden state;
2. projects it into a compact query;
3. retrieves from persistent memory by content attention;
4. writes a bounded residual correction to the host stream;
5. forms a candidate memory value from the current token;
6. identifies underused and novel memory locations;
7. applies gated erase-and-add updates under learned retention;
8. returns explicit continuation state and diagnostics.

The output projection is zero-initialised, so inserting the adapter does not alter the host model before training.

## State contract

`HeraclitusState` contains:

- `memory`: `(batch, memory_slots, state_size)`;
- `usage`: `(batch, memory_slots)`;
- `steps`: `(batch,)` valid-token counts.

State supports validation, detachment, cloning, device and dtype transfer, dictionary serialisation, and batch reordering.

## Operational properties

- Causal token processing.
- Exact chunk continuation for identical token order.
- Masked tokens are identity operations and do not advance state.
- Independent state for every batch element.
- Bounded write rate and residual scale.
- Exact no-op initialisation for safe attachment to pretrained models.
- No dense covariance operations or per-token matrix decompositions.

## Diagnostics

The output reports read entropy, write entropy, mean write rate, residual ratio, memory RMS, and effective slot utilisation. Auxiliary losses encourage balanced slot use and bounded write and residual energy.

## Validation

```bash
ruff check .
mypy heraclitus
pytest --cov=heraclitus --cov-report=term-missing
python -m build
python -m twine check dist/*
```

## Integration guidance

Attach one adapter after a transformer block or after the final block. Keep state separate per sequence or user session. Reset state at document or trust-boundary changes unless persistent memory is deliberately required. During beam search, call `state.index_select(indices)` whenever beams are reordered.

## License

MIT.
