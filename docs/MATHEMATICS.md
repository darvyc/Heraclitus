# Mathematical specification

## 1. Objects

Let:

```text
X in R^(B x T x D)
P in R^(D x R)
U in R^(R x D)
live_t in S^(R-1)
counter_t in S^(R-1)
```

`B` is batch size, `T` is sequence length, `D` is LLM hidden width, and `R` is Heraclitus state width.

The trainable parameter set is:

```text
Theta = {P, U, seed, gate_bias, scale_logit, opposition_logit}
```

Its exact size is:

```text
|Theta| = 2 D R + R + 3
```

## 2. Effective bounded matrices

Raw trainable matrices are projected differentiably into Frobenius norm balls:

```text
P_hat = P * min(1, C_P / max(||P||_F, epsilon))
U_hat = U * min(1, C_U / max(||U||_F, epsilon))
```

Therefore:

```text
||P_hat||_2 <= ||P_hat||_F <= C_P
||U_hat||_2 <= ||U_hat||_F <= C_U
```

## 3. Token normalisation and observation

For token vector `x_t`:

```text
rms(x_t) = sqrt(mean_j x_t[j]^2 + epsilon)
x_bar_t = x_t / rms(x_t)
z_t = x_bar_t P_hat
obs_t = unit(z_t)
```

`unit(v)` returns `v / ||v||` for nonzero `v` and a deterministic unit pole for a zero vector.

Because RMS normalisation gives `||x_bar_t||_2 <= sqrt(D)`:

```text
||z_t||_2 <= C_P sqrt(D)
```

## 4. Dual-flow state

The live flow tracks the current sequence direction. The counter-flow is a slower trace of prior live states.

For valid token `t`:

```text
counter_(t+1) = unit(kappa counter_t + (1 - kappa) live_t)
live_(t+1) = unit(beta live_t + (1 - beta) obs_t)
```

where:

```text
0 <= beta < 1
0 <= kappa < 1
```

The direction used to modulate token `t` is computed before observing token `t`:

```text
lambda = sigmoid(opposition_logit)
dual_t = unit((1 + lambda) live_t - lambda counter_t)
```

This ordering gives a causal recurrence. Masked tokens leave both states unchanged.

## 5. State-conditioned residual

Token alignment and gate are:

```text
score_t = dot(obs_t, dual_t)
gate_t = sigmoid(score_t / temperature + gate_bias)
```

The aligned latent component is:

```text
aligned_t = dot(z_t, dual_t) dual_t
adapted_t = z_t + lambda aligned_t
```

The bounded residual gain is:

```text
alpha = alpha_max sigmoid(scale_logit)
```

The residual is:

```text
delta_t = alpha gate_t adapted_t U_hat
y_t = x_t + delta_t
```

Dropout, when configured, is applied to `delta_t` during training.

## 6. Residual bound

Since `dual_t` is unit length:

```text
||aligned_t||_2 <= ||z_t||_2
||adapted_t||_2 <= (1 + lambda) ||z_t||_2
```

Because `0 <= gate_t <= 1`, `0 <= lambda <= 1`, and `0 <= alpha <= alpha_max`:

```text
||delta_t||_2
<= alpha_max C_U (1 + lambda) C_P sqrt(D)
<= 2 alpha_max C_U C_P sqrt(D)
```

With dropout probability `p`, the training-time bound is multiplied by `1 / (1 - p)` for retained coordinates. Evaluation uses the bound above directly.

## 7. Causality proof

Base case: token `0` uses only `x_0` and the learned initial state.

Inductive step: assume `live_t` and `counter_t` depend only on tokens before `t`. Output `y_t` uses `x_t`, `live_t`, and `counter_t`, so it depends only on tokens through `t`. The update to state `t+1` uses only state `t` and observation `t`. Therefore future tokens cannot affect past outputs.

## 8. Chunk equivalence

The recurrence is Markovian in:

```text
state_t = {live_t, counter_t, steps_t}
```

Passing the final state from one chunk as the initial state of the next chunk reproduces the same recurrence as a single call. With deterministic evaluation settings, concatenated chunk outputs equal full-sequence outputs.

## 9. Batch isolation

Every state tensor has shape `(B, R)` and every transition is row-wise. No reduction is performed across the batch dimension. Therefore sequence `b` cannot influence sequence `b'` for `b != b'`.

## 10. Auxiliary objective

Training uses the task objective plus weighted regularisers:

```text
L_total = L_task
        + w_orth L_orth
        + w_counter L_counter
        + w_drift L_drift
        + w_residual L_residual
```

The terms are:

```text
L_orth = mean((column_normalise(P)^T column_normalise(P) - I)^2)
L_counter = mean((1 - dot(live_t, counter_t)) / 2)
L_drift = mean((1 - dot(live_t, live_(t+1))) / 2)
L_residual = sum(delta^2) / max(sum(X^2), epsilon)
```

All four terms are finite and nonnegative.

## 11. Complexity

For each token, the dominant operations are the two low-rank matrix products:

```text
x_bar_t P_hat
adapted_t U_hat
```

Time complexity:

```text
O(B T D R)
```

Trainable parameter memory:

```text
O(D R)
```

Runtime continuation-state memory:

```text
O(B R)
```
