# Heraclitus 3 integration

## Placement

Attach Heraclitus after a transformer block or another residual boundary:

```python
hidden = transformer_block(hidden, attention_mask=attention_mask)
result = heraclitus.forward_with_state(
    hidden,
    state=session_state,
    attention_mask=attention_mask,
)
hidden = result.hidden_states
session_state = result.state
```

## Training

Use task loss plus the optional auxiliary objective:

```python
result = heraclitus.forward_with_state(hidden, detach_state=False)
loss = language_model_loss + result.regularization_loss()
loss.backward()
```

The output path is initially zero, so a new adapter is an exact output no-op. State still records the stream, which allows auxiliary losses to shape memory before the output projection becomes active.

## Truncated recurrence

Returned state is detached by default. Pass it into the next chunk for truncated backpropagation. Set `detach_state=False` only when the graph must cross the call boundary.

## Decoding

For one token per active sequence:

```python
result = heraclitus.forward_step(hidden_t, state=session_state)
hidden_t = result.hidden_states[:, 0]
session_state = result.state
```

Reorder `session_state` whenever the host KV cache or generation beams are reordered.

## Precision

The adapter supports float32, float16, and bfloat16 host streams and module parameters. Internal computation is float32 and output dtype matches the host stream.

## Padding

Pass a boolean or zero-one mask of shape `(batch, sequence)`. Masked positions receive no residual, do not update state, do not increment steps, and do not contribute to write or residual regularisation.

## Lifecycle and security

Create or reset state at the start of a logical sequence. Never reuse it across tenants. Reset it across document, permission, or trust boundaries unless persistence has been explicitly authorised. Encrypt persisted state and expire it with the session.

## Performance

The 3.0.0 package is a reference implementation with a causal Python token loop. Use `examples/benchmark_adapter.py` on the intended device, dtype, batch size, and sequence length. Do not infer prefill throughput from one-token decoding measurements.
