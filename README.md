# Heraclitus 1.0

Heraclitus is a causal predictive-state parameter for transformer language models. It maintains compact uncertainty-aware sequence memory, preserves multiple plausible latent futures, and writes only statistically normalised prediction error into the residual stream.

The implementation is designed for language-model research, stateful generation, long-running assistants, streaming inference, document processing, and any workload where continuation state must remain explicit, serialisable, batch-isolated, and inexpensive relative to the host model.

## Architecture

For each valid hidden token, Heraclitus:

1. RMS-normalises and projects the hidden vector into a low-dimensional observation;
2. advances a contractive diagonal latent transition;
3. constructs context-conditioned Gaussian predictive shadows;
4. scores each shadow under a diagonal predictive likelihood;
5. preserves posterior mass for both leading and alternative hypotheses;
6. applies an uncertainty-weighted innovation correction;
7. reconstructs whitened surprise through a bounded low-rank map;
8. returns explicit continuation state and operational diagnostics.

The shadows are latent forecasts, not text sequences. Their usefulness is measured by held-out predictive likelihood, calibration, component utilisation, and downstream task performance.

## Install

```bash
pip install -e .
```

Python 3.9 or later and PyTorch 2.0 or later are required.

## Basic use

```python
import torch
from heraclitus import HeraclitusConfig, HeraclitusParameter

module = HeraclitusParameter(
    HeraclitusConfig(
        hidden_size=4096,
        state_size=64,
        num_shadows=4,
    )
)

hidden = torch.randn(2, 128, 4096)
mask = torch.ones(2, 128, dtype=torch.bool)
result = module.forward_with_state(hidden, attention_mask=mask)

adapted_hidden = result.hidden_states
continuation_state = result.state
auxiliary_loss = result.regularization_loss()
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

The state contains:

- posterior latent mean;
- diagonal posterior variance;
- posterior log-probabilities for every shadow;
- valid-token count.

It supports detachment, cloning, checkpointing, device transfer, and batch reordering for beam search or dynamic batching. A cloned state can be retained before speculative decoding and restored when proposed tokens are rejected.

## One-step forecast

```python
shadow_means, diagonal_variance, probabilities = module.predictive_distribution(state)
```

This API exposes the next latent predictive distribution without consuming a token. It is useful for diagnostics, uncertainty routing, speculative systems, confidence-aware orchestration, and training analysis.

## Mathematical contract

For projected observation `z_t` and previous state `(m, v, log_w)`:

```text
a = bounded learned retention in (0, 1)
prior_mean = a * m
prior_variance = a^2 * v + process_noise

context_offsets = normalise(base_offsets + context(prior_mean))
shadow_k = prior_mean + shadow_scale * sqrt(prior_variance) * context_offset_k

prior_weights = temper(log_w, learned_prior)
log_score_k = log(prior_weight_k) + log N(z_t | shadow_k, prior_variance + observation_noise)
posterior_weights = probability_floor(softmax(log_score))

mixture_mean = sum_k posterior_weight_k * shadow_k
innovation = z_t - mixture_mean
gain = prior_variance / (prior_variance + observation_noise)

posterior_mean = mixture_mean + gain * innovation
posterior_variance = filtered_variance + between_shadow_variance

whitened_innovation = innovation / sqrt(prior_variance + observation_noise)
delta = bounded_scale * surprise_gate * whitened_innovation * reconstruction
y_t = x_t + delta
```

The output at token `t` depends only on the current token, preceding valid state, and learned parameters. Masked tokens are identity operations and do not advance state.

## Design properties

### Contractive memory

Every latent coordinate has a learned retention factor strictly below one. This bounds propagation of old-state perturbations while supporting a bank of short and long timescales.

### Context-conditioned alternatives

Shadow geometry changes with the predicted state. Alternative latent futures therefore follow the current context rather than remaining fixed global directions.

### Persistent but recoverable hypotheses

Posterior shadow weights carry evidence across tokens. Tempering prevents irreversible lock-in, while a probability floor prevents unused components from losing all gradient and becoming permanently dead.

### Uncertainty-aware correction

Process noise, observation noise, posterior variance, and between-shadow disagreement remain separate. New evidence receives more weight when the state is uncertain and less weight when the observation channel is noisy.

### Innovation-only write-back

The residual contains whitened prediction error rather than a duplicate projection of the token. This gives the parameter a defined role: communicate information not already captured by its compact predictive state.

### Explicit operational state

No trainable parameter is mutated during forward execution. Runtime state belongs to each sequence and can be safely stored, reordered, rolled back, or resumed.

## Training

Use the host model's task loss as the primary objective:

```python
loss = task_loss + result.regularization_loss()
```

The auxiliary objective contains:

- Gaussian-mixture predictive negative log likelihood;
- weak shadow anti-collapse pressure;
- projection orthogonality pressure;
- residual-energy budgeting.

Track these diagnostics during training:

- `innovation_rms`;
- `surprise_mean`;
- `posterior_variance`;
- `shadow_entropy`;
- `effective_shadows`;
- leading and runner-up shadow probabilities;
- residual ratio.

Healthy training requires predictive loss to improve without shadow collapse, permanent uniformity, exploding posterior variance, or excessive residual energy.

## Integration guidance

Insert Heraclitus after attention or after the feed-forward sublayer, before the residual stream is consumed by the next normalisation boundary. Begin with one parameter in the upper half of the network and establish an ablation before adding more layers.

Recommended initial settings:

```text
state_size = 32 to 128
num_shadows = 4
max_residual_scale = 0.02 to 0.10
shadow_scale = 0.10 to 0.35
shadow_weight_memory = 0.90 to 0.99
```

For mixed-precision hosts, Heraclitus performs state estimation in float32 and returns hidden states in the input dtype. Continuation states should remain float32 unless memory constraints have been empirically shown to justify another representation.

## Benchmarking

Run the included operational benchmark:

```bash
python examples/benchmark_adapter.py \
  --hidden-size 4096 \
  --state-size 64 \
  --num-shadows 4 \
  --sequence-length 512
```

A model-level evaluation should compare equal training compute and report:

- validation perplexity and held-out negative log likelihood;
- downstream generation quality;
- long-context retrieval and topic-shift adaptation;
- training and generation throughput;
- peak memory and latency;
- shadow calibration and utilisation;
- mean and tail residual ratios;
- uncertainty under distribution shift;
- confidence intervals across seeds.

Required ablations include a parameter-matched MLP, ordinary low-rank adapter, single-state innovation filter, GRU-style parameter, compact state-space layer, one-shadow Heraclitus, and removal of contextual shadows, uncertainty, probability flooring, weight tempering, whitening, and predictive loss.

## Validation

```bash
ruff check .
mypy heraclitus
pytest --cov=heraclitus --cov-report=term-missing
python -m build
python -m twine check dist/*
```

Continuous integration runs the supported Python matrix, static analysis, tests, coverage, bytecode compilation, package construction, and distribution validation.

## Scope

Heraclitus supplies a rigorously defined recurrent parameter and the instrumentation needed to evaluate it. Performance superiority is an empirical result, not a property conferred by architecture alone; release decisions should be based on compute-matched measurements from the target model and workload.

## License

MIT.
