# Heraclitus 

Heraclitus is a bounded recurrent slot-memory adapter for transformer hidden states. It carries a small explicit state across chunks, retrieves from that state through multi-head content attention, and writes sparse updates selected by novelty and recent usage.

## Status

Heraclitus is a research beta. Its causal, numerical, state-management, and residual-bound contracts are tested. 

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
        write_topk=2,
    )
)

first_hidden = torch.randn(2, 256, 4096)
first = adapter.forward_with_state(first_hidden)

next_hidden = torch.randn(2, 128, 4096)
continued = adapter.forward_with_state(next_hidden, state=first.state)

loss = task_loss + continued.regularization_loss()
```

## Token transition

For each valid token, Heraclitus:

1. RMS-normalises the host hidden state in float32.
2. Retrieves a multi-head content-addressed summary from persistent memory.
3. Forms a residual and projects it onto a strict per-token norm ball.
4. Forms a new memory candidate from both the current token and retrieved memory.
5. Selects at most `write_topk` slots using novelty and decayed usage.
6. Applies a bounded erase-and-add update.
7. Returns the adapted hidden state and explicit continuation state.

The residual contract is:

```text
norm(delta_t) <= max_residual_ratio * norm(h_t)
```

This is enforced after the output projection, so it remains true even if projection weights grow during training.

## State contract

`HeraclitusState` contains:

- `memory`: `(batch, memory_slots, state_size)`
- `usage`: `(batch, memory_slots)`
- `steps`: `(batch,)`

Memory values are clamped to `[-1, 1]`. Usage is both exponentially decayed and clamped to `max_usage`; `usage_decay=1` is rejected. State can be detached, cloned, moved, serialised, and reordered for beam search.

```python
payload = state.as_dict()
restored = HeraclitusState.from_dict(payload)
reordered = restored.index_select(beam_indices)
```

Keep one state per independent sequence, session, or generation beam. Reset state at user, document, permission, or trust-boundary changes unless persistence is deliberate.

## Precision

Heraclitus performs its internal geometry in float32 through explicit functional projections. The host hidden dtype is preserved at the output. This supports float32, float16, and bfloat16 host streams, including when the adapter parameters have been converted to reduced precision.

## Training semantics

The output projection is zero-initialised, so attaching a new adapter is an exact output no-op. Returned state is detached by default. Pass `detach_state=False` only when gradients must cross the call boundary and retained graph memory is acceptable.

All auxiliary losses exclude masked positions. Padding values therefore cannot change write-energy or residual-energy regularisation.

## Generation

For one-token decoding, use `forward_step`:

```python
result = adapter.forward_step(hidden_t, state=session_state)
hidden_t = result.hidden_states[:, 0]
session_state = result.state
```

For chunked prefill, use `forward_with_state`. The current reference implementation uses an explicit causal token loop. Benchmark it on the intended host and workload before deployment.

## Diagnostics

Each pass reports:

- read and write entropy
- mean write rate
- aggregate residual ratio
- maximum per-token residual ratio
- memory RMS and maximum slot norm
- effective slot utilisation
- maximum usage value

These are observability signals, not quality scores.

## Validation

```bash
ruff check .
mypy heraclitus
pytest --cov=heraclitus --cov-report=term-missing
python examples/basic_usage.py
python examples/benchmark_adapter.py --sequence-length 32 --iterations 2
python -m build
python -m twine check dist/*
```

The suite covers exact no-op initialisation, complete active gradient flow, causality, chunk equivalence, masking, padding-invariant auxiliary losses, batch isolation, state round trips, beam reordering, sparse writes, retrieved-state-conditioned candidates, strict residual bounds, reduced-precision modules, bounded long-horizon memory and usage, diagnostics, and invalid inputs.

## Evaluation standard

Judge Heraclitus against matched-cost baselines, including no memory, a single recurrent vector, GRU, LSTM, and recurrent memory-token adapters. Report task accuracy or perplexity, state bytes, throughput, peak accelerator memory, long-length generalisation, and ablations for slot count, top-k allocation, retention, usage pressure, and residual bounds.

Suitable tasks include delayed copy, associative recall, passkey retrieval after source tokens leave the attention window, streaming language modelling, and interference tests. A runnable delayed-copy harness is included:

```bash
python examples/evaluate_delayed_copy.py --steps 500 --delay 32
```

The script reports adapter, no-memory, and chance accuracy. Its output is a local diagnostic, not a repository-level performance claim.

## Limits

Heraclitus is learned fixed-capacity compression. It cannot losslessly retain arbitrary history, guarantee exact quotations, or replace external retrieval for exhaustive evidence. Its token-sequential reference implementation may also be unsuitable for high-throughput prefill without compilation or a fused scan.

## Security and privacy

Treat recurrent state as derived user data. Isolate it by sequence and tenant, encrypt it in transit and at rest, expire it with the session, and reset it across trust boundaries. A learned erase operation is not guaranteed deletion.

## License

MIT.
