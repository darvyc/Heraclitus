# Heraclitus 3 mathematical contract

Let `h_t` be the current hidden state, `M_t` the incoming slot bank, and `u_t` slot usage.

## Read

```text
x_t = RMSNorm(h_t)
q_t = W_q x_t
K_t = W_k M_t
V_t = W_v M_t
A_t = softmax(q_t K_t^T / sqrt(head_size))
r_t = concat_heads(A_t V_t)
```

## Strictly bounded residual

```text
z_t = sigmoid(g) * dropout(W_o r_t)
b_t = rho_max * norm(h_t)
delta_t = z_t * min(1, b_t / max(norm(z_t), epsilon))
h'_t = h_t + delta_t
```

Therefore, for every finite token,

```text
norm(delta_t) <= rho_max * norm(h_t).
```

This remains true for arbitrary finite projection weights.

## Retrieved-state-conditioned candidate

```text
c_t = tanh(W_c [x_t, r_t])
e_t = sigmoid(W_e [x_t, r_t])
g_t = w_max * sigmoid(W_g [x_t, r_t])
```

## Sparse allocation

```text
novelty_i = 1 - cosine(c_t, M_ti)
relative_usage_i = u_ti / max(mean(u_t), epsilon)
logit_i = (novelty_i - lambda_u * relative_usage_i) / temperature
```

Only the largest `k=write_topk` logits remain finite. Softmax over those logits gives allocation `a_t` with at most `k` nonzero entries.

## State update

```text
amount_ti = g_t * a_ti
retained_ti = retention_i * M_ti
M_(t+1)i = clamp(
    retained_ti * (1 - amount_ti * e_t)
    + amount_ti * c_t,
    -1,
    1
)

u_(t+1)i = clamp(
    usage_decay * u_ti + g_t * a_ti,
    0,
    max_usage
)
```

Masked tokens satisfy:

```text
h'_t = h_t
M_(t+1) = M_t
u_(t+1) = u_t
steps_(t+1) = steps_t
```

## Causality and chunk equivalence

Output is computed from the current token and incoming state before the current write. The recurrence is Markovian in explicit state. Under deterministic settings, processing adjacent chunks with exact state hand-off reproduces full-sequence processing.

## Capacity statement

The state contains `M * R` floating-point values per sequence. It can learn bounded task-relevant summaries, but cannot injectively encode an arbitrary unbounded token history. Exact retrieval claims are therefore outside the mathematical contract.
