# Heraclitus Release Edition

Heraclitus is a multimodal predictive-state parameter for transformer language models. It maintains persistent latent hypotheses, propagates structured uncertainty through stable higher-dimensional dynamics, and writes bounded reliability-gated innovation into the host residual stream.

The Release Edition is intended for language-model research, stateful generation, streaming inference, long-running assistants and document-processing systems where continuation state must remain explicit, serialisable, batch-isolated and mathematically inspectable.

## Architecture

For each valid hidden token, Heraclitus:

1. RMS-normalises and projects the hidden representation into a compact latent observation;
2. evolves every latent mode independently through a stable orthogonal-contractive operator;
3. propagates diagonal-plus-low-rank covariance;
4. evaluates correlated Gaussian evidence using Woodbury solves and determinant identities;
5. updates mode probabilities with a tempered Markov prior;
6. performs reliability-aware Bayesian innovation correction;
7. computes mixture uncertainty by the law of total covariance;
8. writes a bounded, whitened and uncertainty-gated residual correction;
9. returns explicit continuation state and diagnostics.

See `docs/MATHEMATICS.md` for the mathematical contract, stability conditions, covariance geometry, information objectives and complexity bounds.

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

## State

For each sequence and latent mode, the continuation state contains:

- a persistent latent mean;
- a strictly positive diagonal covariance;
- a low-rank covariance factor;
- a posterior mode probability;
- a valid-token count.

The aggregate mixture mean and marginal variance are available through `state.mean` and `state.variance`. State objects support detachment, cloning, checkpointing, device transfer and batch reordering.

## Mathematical guarantees

Under the configured retention bounds, the deterministic transition is contractive. Covariance matrices remain positive definite because every diagonal term is floored above zero and every low-rank contribution is positive semidefinite. Mode probabilities remain normalised and bounded away from numerical extinction. Masked tokens are identity operations and do not advance state.

These are implementation-level guarantees. Language-model quality remains an empirical property and must be established through compute-matched evaluation.

## Diagnostics

Track predictive negative log likelihood, calibration error, innovation energy, latent standard deviation, mode separation, effective mode count, posterior variance, residual ratio and downstream held-out quality.

## Evaluation contract

Any performance claim must compare equal training compute against a parameter-matched bottleneck MLP, an ordinary low-rank adapter, a gated recurrent adapter, a compact state-space layer and a one-mode ablation. Report validation perplexity, downstream generation quality, retrieval behaviour, topic-shift adaptation, throughput, decoding latency, peak memory, calibration, mode utilisation and confidence intervals across multiple seeds.

## Computational scope

Heraclitus is recurrent across tokens. The Release Edition therefore treats streaming and stateful inference as the primary operational regime. Training-time value must be established against parallel alternatives rather than inferred from architectural complexity.

## Validation

```bash
ruff check .
mypy heraclitus
pytest --cov=heraclitus --cov-report=term-missing
python -m build
python -m twine check dist/*
```

The release-contract suite covers persistent mode identity, covariance shape and positivity, probability normalisation, chunk equivalence, masking, Householder norm preservation, gradient flow and state serialisation.

## License

MIT.
