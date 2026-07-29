# LLM integration

## Placement

A transformer block can contain one or more Heraclitus parameters:

```python
hidden = hidden + attention(norm_1(hidden), attention_mask=attention_mask)
hidden = heraclitus_after_attention(hidden, attention_mask=attention_mask)
hidden = hidden + mlp(norm_2(hidden))
hidden = heraclitus_after_mlp(hidden, attention_mask=attention_mask)
```

A single parameter after the complete block is also valid:

```python
hidden = transformer_block(hidden, attention_mask=attention_mask)
hidden = heraclitus(hidden, attention_mask=attention_mask)
```

## Training

Use the ordinary task loss and add the auxiliary objective returned by `forward_with_state`:

```python
result = heraclitus.forward_with_state(hidden, attention_mask=attention_mask)
hidden = result.hidden_states
loss = language_model_loss + result.regularization_loss()
```

No parameter update occurs inside the forward call. Optimisers, gradient accumulation, distributed data parallel training, activation checkpointing, and automatic mixed precision retain their standard semantics.

## Generation

Keep one `HeraclitusState` per active sequence:

```python
state = heraclitus.initial_state(batch_size=batch_size, device=device)

for hidden_chunk in hidden_chunks:
    result = heraclitus.forward_with_state(hidden_chunk, state=state)
    hidden_chunk = result.hidden_states
    state = result.state
```

State is detached by default at the call boundary. This gives bounded graph lifetime during long generation runs.

## Padding and packed batches

Pass a boolean or zero-one mask with shape `(batch, sequence)`. Masked positions:

- receive no Heraclitus residual
- do not update live flow
- do not update counter-flow
- do not increment the valid-token count

## Request lifecycle

Create or reset state at the beginning of a new request. Preserve it only while continuing the same logical sequence. Reorder the state rows whenever the host generation engine reorders beams or batch entries.

## Serialisation

Model parameters are included in the normal PyTorch module state dictionary:

```python
torch.save(heraclitus.state_dict(), "heraclitus.pt")
heraclitus.load_state_dict(torch.load("heraclitus.pt", weights_only=True))
```

Continuation state can be saved directly:

```python
torch.save(state.as_dict(), "continuation-state.pt")
state = HeraclitusState.from_dict(
    torch.load("continuation-state.pt", weights_only=True)
)
```

## Precision

Input and output dtypes match. State transitions and low-rank geometry are evaluated in float32, then the residual is cast back to the model hidden dtype. This provides stable operation for float32, float16, and bfloat16 hidden streams.

## Configuration

For a hidden width of 4096:

```python
HeraclitusConfig(hidden_size=4096, state_size=32)
HeraclitusConfig(hidden_size=4096, state_size=64)
HeraclitusConfig(hidden_size=4096, state_size=128)
```

State width controls capacity and cost linearly. A width of 64 adds 524,355 trainable scalars to a 4096-wide residual stream.
