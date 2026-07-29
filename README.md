# Heraclitus

Heraclitus is a bounded recurrent memory adapter for transformer hidden states. It compresses prior activations into a small, explicit slot bank that continues across chunks, retrieves relevant state through multi-head content attention, and writes new information through novelty and usage-aware gated updates.

Heraclitus is designed for streaming inference, recurrent context compression, stateful agents, and controlled experiments in context extension. It is causal, batch-isolated, serialisable, reorderable for beam search, and safe to attach to a pretrained model as an exact no-op.

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
first = adapter.forward_with_state(hidden)

next_hidden = torch.randn(2, 128, 4096)
continued = adapter.forward_with_state(next_hidden, state=first.state)

loss = task_loss + continued.regularization_loss()
```

## Memory contract

For each valid token, Heraclitus performs five operations:

1. normalise the host hidden state;
2. retrieve a multi-head content-addressed summary from persistent slots;
3. add a bounded residual correction to the host stream;
4. allocate a bounded write using novelty and decayed slot usage;
5. return the adapted hidden state and explicit continuation state.

Let `h_t` be the current hidden state, `M_t` the slot bank, and `u_t` slot usage.

```text
q_t = W_q RMSNorm(h_t)
K_t = W_k M_t
V_t = W_v M_t
A_t = softmax(q_t K_t^T / sqrt(head_size))
r_t = concat_heads(A_t V_t)

h'_t = h_t + alpha W_o r_t

c_t = tanh(W_c RMSNorm(h_t))
novelty_i = 1 - cosine(c_t, M_ti)
allocation = softmax(novelty - usage_penalty * normalised_usage)
write_rate = max_write_rate * sigmoid(W_g [h_t, r_t])

M_(t+1) = clamp(
    retention * M_t * (1 - write_amount * erase)
    + write_amount * c_t,
    -1,
    1
)
```

The slot values are hard-clamped to `[-1, 1]`. Write rate and residual scale have explicit configuration ceilings. This gives a bounded recurrent state transition for finite inputs.

## State

`HeraclitusState` contains:

- `memory`: `(batch, memory_slots, state_size)`;
- `usage`: `(batch, memory_slots)`;
- `steps`: `(batch,)` valid-token counts.

State supports validation, detachment, cloning, device and dtype transfer, dictionary serialisation, and batch reordering.

```python
payload = state.as_dict()
restored = HeraclitusState.from_dict(payload)
reordered = restored.index_select(beam_indices)
```

Keep one state per independent sequence, user session, or generation beam. Reset state at document, user, permission, or trust-boundary changes unless persistence is explicitly intended.

## Operational guarantees

The release contract covers:

- causal token processing;
- exact chunk continuation in evaluation mode for identical token order;
- exact identity output at initialisation;
- masked-token identity and no state advancement;
- independent state for every batch element;
- explicit state serialisation and beam reordering;
- finite gradients across every active trainable parameter;
- bounded slot values under long recurrent execution;
- dtype preservation for host hidden states;
- genuine multi-head memory retrieval.

Dropout is stochastic in training mode, so exact chunk equivalence is only guaranteed when dropout is disabled or the adapter is in evaluation mode.

## Diagnostics

Each forward pass reports:

- `read_entropy`: concentration of memory retrieval;
- `write_entropy`: concentration of slot allocation;
- `write_rate`: mean gated write magnitude;
- `residual_ratio`: adapter correction relative to host activation norm;
- `memory_rms`: recurrent-state energy;
- `effective_slots`: entropy-derived slot utilisation;
- `maximum_slot_norm`: largest slot norm.

These values are observability signals, not task-quality scores. Interpret them alongside validation loss, recall accuracy, latency, and ablation results.

## Training

The output projection is zero-initialised, so attaching Heraclitus does not alter the host model before training. The output projection learns first; the complete memory path receives task gradients once it becomes active. Auxiliary losses provide bounded pressure on slot balance, write energy, and residual energy.

```python
result = adapter.forward_with_state(hidden, detach_state=False)
loss = task_loss + result.regularization_loss(
    usage_weight=1e-4,
    write_weight=1e-4,
    residual_weight=1e-4,
)
loss.backward()
```

For truncated backpropagation through time, pass the returned detached state into the next chunk. Set `detach_state=False` only when gradients must cross the chunk boundary and the retained graph fits the available memory.

## Transformer integration

Attach Heraclitus after a transformer block or after the final block:

```python
hidden = transformer_block(hidden, attention_mask=attention_mask)
result = adapter.forward_with_state(
    hidden,
    state=session_state,
    attention_mask=attention_mask,
)
hidden = result.hidden_states
session_state = result.state
```

For generation:

1. maintain state beside the model KV cache;
2. reorder state whenever beams are reordered;
3. discard or reset state when the sequence ownership or trust boundary changes;
4. persist state only through an authenticated, encrypted session store.

Heraclitus complements a transformer context window by carrying a learned bounded summary after earlier tokens are removed. It is not lossless token storage and should not replace exact retrieval when verbatim evidence is required.

## Security and privacy

A recurrent state may encode sensitive or adversarial information even when it is not human-readable.

Treat every state object as derived user data:

- isolate it by user and sequence;
- encrypt it in transit and at rest;
- expire it with the session;
- never reuse it across tenants;
- reset it after untrusted-document or permission-boundary transitions;
- do not treat a learned erase operation as a guaranteed deletion mechanism.

## Validation

```bash
ruff check .
mypy heraclitus
pytest --cov=heraclitus --cov-report=term-missing
python -m build
python -m twine check dist/*
```

The test suite exercises no-op initialisation, parameter immutability, complete active gradient flow, causality, chunk equivalence, masking, all-masked transitions, batch isolation, state round trips, beam reordering, multi-head execution, half precision, long-horizon boundedness, diagnostics, parameter counting, and invalid inputs.

## Evaluation standard

A memory adapter should be judged against matched-cost baselines, not against its own diagnostics. Suitable evaluations include delayed copy, associative recall, passkey retrieval after source tokens leave the attention window, streaming language modelling, interference tests, and latency measurement against a GRU, LSTM, single-vector recurrence, and no-memory adapter.

Report at minimum:

- task accuracy or perplexity;
- recurrent state bytes per sequence;
- training and inference tokens per second;
- peak accelerator memory;
- performance beyond the training sequence length;
- ablations for novelty, usage, retention, slot count, and regularisation.

## Scope

Heraclitus is learned recurrent compression. Its fixed-size state can preserve task-relevant sufficient statistics, but it cannot losslessly retain an arbitrary history. Use external retrieval or retained token context for exact quotations, exhaustive provenance, or unbounded collections of independent facts.

## License

MIT.
