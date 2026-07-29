# Heraclitus 2.0

Heraclitus is an experimental multimodal predictive-state parameter for transformer language models. It maintains several persistent latent hypotheses, propagates correlated uncertainty, and writes a bounded reliability-gated prediction error into the host residual stream.

Version 2.0 is a research implementation, not evidence of superiority over attention, LoRA, recurrent adapters or state-space layers. Model-level claims require compute-matched experiments.

## What changed

Heraclitus 1.x collapsed every candidate back into one shared posterior mean and recreated local offsets at the next token. Version 2.0 instead gives each mode its own recurrent mean, diagonal variance and low-rank covariance factor.

The transition is a product of Householder reflections followed by context-dependent contraction. This introduces stable cross-coordinate dynamics rather than independent scalar decay. Gaussian likelihoods use diagonal-plus-low-rank covariance and the Woodbury identity, so correlated uncertainty does not require an `r by r` matrix inverse.

Process noise and observation noise are context-dependent. Residual intervention combines novelty with reliability: surprise can increase the write, while high predictive uncertainty suppresses it. A latent-information floor discourages the learned projection from becoming trivially predictable and uninformative.

See `docs/MATHEMATICS_2_0.md` for the full mathematical contract and its approximations.

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
        num_modes=4,
        covariance_rank=4,
        transition_reflections=4,
    )
)

hidden = torch.randn(2, 128, 4096)
result = module.forward_with_state(hidden)
adapted_hidden = result.hidden_states
continuation_state = result.state
loss = task_loss + result.regularization_loss()
```

The continuation state can be detached, cloned, checkpointed, moved across devices and reordered for beam search or dynamic batching. `num_shadows` remains accepted as a compatibility alias for `num_modes`.

## State

For each sequence and mode, the state contains:

- a persistent latent mean;
- a positive diagonal covariance;
- a low-rank covariance factor;
- a posterior mode probability;
- a valid-token count.

The mixture mean and total marginal variance remain available through `state.mean` and `state.variance`.

## Diagnostics

Track predictive negative log likelihood, calibration error, latent standard deviation, mode separation, effective modes, posterior variance, residual ratio and held-out downstream quality. A healthy internal diagnostic is not a substitute for an external language-model result.

## Required evaluation

Any performance claim must compare equal training compute against:

- a parameter-matched bottleneck MLP;
- LoRA or another ordinary low-rank adapter;
- a GRU-style recurrent adapter;
- a compact state-space layer;
- a one-mode Heraclitus ablation;
- removal of low-rank covariance, contextual noise, information preservation and reliability gating.

Report validation perplexity, downstream generation quality, retrieval and topic-shift behaviour, training throughput, decoding latency, peak memory, calibration, mode utilisation and confidence intervals across multiple seeds.

## Computational scope

The implementation remains recurrent across tokens. That cost is explicit rather than hidden. Its most natural present use is stateful or streaming inference, where recurrence already occurs token by token. Training-time value must be demonstrated against parallel alternatives; the repository does not claim that architecture alone compensates for sequential execution.

## Validation

```bash
ruff check .
mypy heraclitus
pytest --cov=heraclitus --cov-report=term-missing
python -m build
python -m twine check dist/*
```

The 2.0 release-contract tests cover persistent mode identity, low-rank state shape, probability normalisation, chunk equivalence, masking, norm preservation under Householder dynamics, gradients and 1.x configuration compatibility.

## License

MIT.
