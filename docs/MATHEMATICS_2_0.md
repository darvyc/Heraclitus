# Heraclitus 2.0 mathematical specification

Heraclitus 2.0 represents sequence state as a finite mixture of persistent modes. For mode k at token t, the state is

```text
(mu[t,k], d[t,k], U[t,k], pi[t,k])
```

where `mu` is the mode mean in R^r, `d` is a positive diagonal covariance, `U` is an r by c low-rank covariance factor, and `pi` is the posterior mode probability. The covariance is

```text
P[t,k] = diag(d[t,k]) + U[t,k] U[t,k]^T.
```

## Stable cross-coordinate dynamics

The transition uses a product of Householder reflections

```text
Q = H_j ... H_2 H_1,
H_i = I - 2 v_i v_i^T / (v_i^T v_i).
```

Every `H_i` is orthogonal, hence `Q` is orthogonal and preserves Euclidean norm. A context-dependent retention vector `a[t,k]` lies strictly inside `(0, 1)^r`. The predicted mean is

```text
mu-[t,k] = a[t,k] elementwise_multiply (Q mu[t-1,k]).
```

This supplies cross-coordinate rotation without allowing an unstable spectral radius. It is richer than independent scalar decay while remaining auditable.

## Correlated predictive uncertainty

The diagonal and low-rank factors propagate as

```text
d-[t,k] = a[t,k]^2 elementwise_multiply d[t-1,k] + q[t,k]
U-[t,k] = diag(a[t,k]) Q U[t-1,k] + B[k].
```

Both process noise `q[t,k]` and observation noise `r[t]` are positive and context-dependent. The predictive observation covariance is

```text
S[t,k] = diag(d-[t,k] + r[t]) + U-[t,k] U-[t,k]^T.
```

Likelihood evaluation and linear solves use the Woodbury identity:

```text
(D + U U^T)^-1
= D^-1 - D^-1 U (I + U^T D^-1 U)^-1 U^T D^-1.
```

The expensive inverse is therefore only `c by c`, not `r by r`.

## Persistent alternatives

Each mode has its own recurrent mean and covariance. Modes are not regenerated as offsets around one shared state. Their posterior probabilities are updated by Bayes' rule:

```text
log pi[t,k] proportional_to log pi-[t,k] + log Normal(z[t] | mu-[t,k], S[t,k]).
```

A small probability floor prevents irreversible numerical death, while mode-separation and calibration diagnostics reveal whether the mixture is merely cosmetic.

## Exact mean correction and assumed-density covariance update

For innovation `e[t,k] = z[t] - mu-[t,k]`, compute

```text
s[t,k] = S[t,k]^-1 e[t,k]
mu[t,k] = mu-[t,k] + P-[t,k] s[t,k].
```

This is the exact linear-Gaussian posterior mean. The covariance is compressed back into diagonal plus low-rank form after each update. This is an assumed-density approximation and is stated as such rather than presented as an exact full-covariance Kalman filter.

## Information and intervention controls

The learned observation projection is protected against trivial predictable collapse by a latent-variance floor. The residual write is whitened by mixture uncertainty and gated by two terms:

```text
gate = novelty(surprise) * reliability(predictive_variance).
```

Novel observations can increase intervention, but high model uncertainty suppresses it. This prevents the earlier failure mode in which the least trustworthy predictions automatically produced the strongest writes.

## Falsifiability

The architecture does not imply language-model improvement. A release claim requires compute-matched comparisons against a bottleneck MLP, LoRA, GRU-style adapter, compact state-space block and one-mode ablation, with multiple seeds and reporting of perplexity, quality, throughput, latency, memory, calibration and mode utilisation.
