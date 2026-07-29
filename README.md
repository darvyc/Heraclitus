# Heraclitus 1.0

Heraclitus is an experimental causal, predictive, low-rank state adapter for transformer language models. It estimates a compact latent sequence state, preserves several probabilistic alternatives to its next prediction, and writes only prediction error back into the residual stream.

It is a research hypothesis, not a demonstrated replacement for attention, a long-context guarantee, or a production-proven improvement. Its value must be established by compute-matched experiments.

## Core idea

For each hidden token, Heraclitus:

1. projects the RMS-normalised hidden vector into a low-dimensional latent observation;
2. predicts the next latent state with a stable diagonal transition;
3. constructs K Gaussian predictive shadows around that forecast;
4. scores each shadow by the observation likelihood;
5. retains the full posterior distribution, including the best and next-best latent alternatives;
6. performs a diagonal Kalman-style correction;
7. reconstructs a bounded, surprise-gated innovation into the transformer stream.

The shadows do not claim to store literal alternative text answers. They store alternative latent predictions that can be trained and scored against subsequent hidden observations.

## Install

```bash
pip install -e .
```

Requires Python 3.9 or later and PyTorch 2.0 or later.

## Usage

```python
import torch
from heraclitus import HeraclitusConfig, HeraclitusParameter

module = HeraclitusParameter(
    HeraclitusConfig(
        hidden_size=4096,
        state_size=64,
        num_shadows=4,
        max_residual_scale=0.10,
    )
)

hidden = torch.randn(2, 128, 4096)
mask = torch.ones(2, 128, dtype=torch.bool)
result = module.forward_with_state(hidden, attention_mask=mask)

hidden = result.hidden_states
state = result.state
loss = language_model_loss + result.regularization_loss()
```

## Stateful generation

```python
state = None
outputs = []

for hidden_chunk in hidden_chunks:
    result = module.forward_with_state(hidden_chunk, state=state)
    outputs.append(result.hidden_states)
    state = result.state
```

The continuation state contains:

- posterior latent mean;
- diagonal posterior variance;
- posterior log-probabilities for all Gaussian shadows;
- valid-token count.

It can be detached, cloned, checkpointed, moved across devices, and reordered for beam search or dynamic batching.

## Mathematical contract

For observation z_t and previous posterior state (m_(t-1), v_(t-1)):

```text
a = bounded learned retention in (0, 1)
m_prior = a * m_previous
v_prior = a^2 * v_previous + q

shadow_k = m_prior + shadow_scale * sqrt(v_prior) * offset_k
log_weight_k = previous_log_weight_k + log N(z_t | shadow_k, v_prior + r)
weights = softmax(log_weight)

m_mix = sum_k weights_k * shadow_k
innovation = z_t - m_mix
gain = v_prior / (v_prior + r)

m_t = m_mix + gain * innovation
v_t = (1 - gain) * v_prior + weighted_shadow_spread

delta_t = bounded_scale * surprise_gate * innovation * reconstruction
y_t = x_t + delta_t
```

The output at token t depends only on the current token, the state accumulated before it, and learned parameters. Masked tokens are unchanged and do not advance state.

## Why Gaussian shadows

A single mean collapses uncertainty into one guess. Gaussian shadows retain multiple locally plausible latent futures. Their likelihoods are updated by evidence, so the system can preserve a runner-up hypothesis rather than irreversibly discarding it.

This only earns the phrase next-best prediction when three conditions hold:

- shadows are diverse rather than collapsed;
- likelihoods are calibrated;
- future-token predictive loss demonstrates that the runner-up mode contains useful information.

## Training objective

The auxiliary objective combines:

- predictive Gaussian-mixture negative log likelihood;
- shadow anti-collapse penalty;
- projection orthogonality penalty;
- residual-energy budget.

The task loss remains primary.

## Required empirical validation

A credible evaluation must compare against:

- a parameter-matched MLP adapter;
- an ordinary low-rank adapter;
- a single-state innovation filter;
- a GRU-style adapter;
- a small state-space layer;
- Heraclitus without shadows, uncertainty, surprise gating, or predictive loss.

Report perplexity, downstream quality, long-context performance, training throughput, generation latency, memory, calibration, shadow utilisation, and statistical uncertainty.

## Limitations

- The current recurrence is sequential over tokens.
- Diagonal covariance cannot represent correlated latent uncertainty.
- Gaussian shadows are local latent hypotheses, not discrete semantic plans.
- More shadows increase cost linearly.
- Stable mathematics does not establish useful language modelling.
- Speculative decoding requires state rollback for rejected tokens.

## Testing

```bash
pytest -q
```

The tests cover shape and dtype preservation, causality, chunk equivalence, batch isolation, masking, gradients, serialisation, state reordering, probabilistic diagnostics, and bounded effective matrices.

## License

MIT.
