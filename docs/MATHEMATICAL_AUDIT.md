# Heraclitus: Mathematical Audit and Minimum Specification

## Verdict

Before this patch, Heraclitus was a suggestive software sketch, not a mathematically coherent learning architecture. Its headline mechanism did not affect the computation, its cross-model geometry was undefined, one update moved in the wrong optimisation direction, and its tests checked shapes and bookkeeping rather than the stated scientific claims.

The patch repairs the immediately falsifiable defects and supplies a minimum mathematical specification. It does not turn the project into a validated research result. That requires controlled experiments, baselines, ablations, uncertainty estimates, and a task-level objective.

## 1. Defects found in the original implementation

### 1.1 The directional attention term was exactly inert

The original code added one scalar per head to every element of that head's attention-logit matrix.

For any row vector z and scalar c:

    softmax(z + c * 1) = softmax(z)

Therefore the original direction vector could not change attention probabilities. The central architectural claim was false at code level.

The corrected mechanism assigns a direction-dependent bias to each key token:

    score[b,h,i,j] = q[b,h,i] dot k[b,h,j] / sqrt(d_head)
                       + alpha[h] * a[b,h,j] dot d

where:

    d is a unit vector in S^2
    a[b,h,j] is a learned unit 3-vector for key token j
    alpha[h] is a learned per-head scale

The added term varies with j, the softmax normalisation axis, so it can alter attention.

### 1.2 The predictive update used the ascent sign

Let the local prediction error be:

    L(b) = 0.5 * ||y(b) - m||^2

For an additive output bias b, dy/db = I. Hence:

    grad_b L = y - m

Gradient descent requires:

    b_next = b - eta * (y - m)

The original implementation added this residual and therefore increased the local squared error in the additive approximation. The sign is now corrected.

### 1.3 The claimed Hebbian update did not use the synapse's input

For a linear map y = W x, a local correlation update for W must use x as the pre-synaptic activity and y as the post-synaptic activity. The original update modified the attention output projection but used the transformer's block input as `pre`, not the tensor entering that projection. The dimensions happened to match; the causal locality did not.

The attention module now exposes the merged attention context that actually enters the output projection.

### 1.4 Unnormalised Hebbian growth had no stationary scale

Pure Hebbian correlation can increase weight norms without bound. Normalising every update to unit Frobenius norm, as the original code did, is not a decay law and destroys information about signal magnitude.

The corrected multi-output Oja-style update is:

    Delta W = E[y x^T] - diag(E[y^2]) W

The second term opposes runaway norm growth. The final parameter displacement is also bounded relative to ||W||.

This is a stability mechanism, not a convergence theorem.

### 1.5 Cross-model direction comparison lacked a common coordinate frame

A cosine is meaningful only when both vectors are expressed in the same basis. The original project gave each transformer an independently initialised feature-to-3D projection, then compared the resulting 3D coordinates as though their axes had shared meaning.

That is a gauge error. Rotating one model's private 3D frame changes every cosine without changing that model's internal computation.

The registry now supplies one shared orthonormal projection frame for each hidden width. This removes the most obvious arbitrary rotation between registered members.

Important limitation: a shared projection matrix is necessary but not sufficient when the models' hidden feature bases themselves have diverged. Semantic comparison then requires anchor data and explicit alignment. The repository now includes a proper-rotation orthogonal Procrustes solver for that calibration step.

Given paired anchor directions A and B, it solves:

    R_star = argmin over R in SO(3) of ||A R - B||_F

using the singular value decomposition of A^T B.

### 1.6 The connection threshold had no null model

For independent directions uniformly distributed on S^2, the cosine is uniform on [-1, 1]. Therefore:

    P(cosine >= tau) = (1 - tau) / 2

At tau = 0.7, the random-match probability is 0.15. In a registry with N = 101 members, the expected random out-degree is:

    (N - 1) * 0.15 = 15

Thus 0.7 is not a selective threshold in a large swarm. The code now exposes the exact S^2 null probability, expected random degree, and inverse threshold calibration:

    tau = 1 - 2k / (N - 1)

where k is the desired expected number of random matches.

This null model assumes independent uniform directions. Empirical directions will generally be correlated, so a permutation or bootstrap null is still required for serious experiments.

### 1.7 The sphere utilities were undefined at important edge cases

Normalising the zero vector does not produce a point on S^2. The old implementation returned the zero vector while describing it as sphere-valued.

The new convention maps a numerically zero vector to a fixed pole. This is explicit and unit norm, although applications should count how often this fallback occurs because frequent zero projections indicate a failed representation.

The original spherical interpolation also divided by sin(pi) for antipodal vectors. The corrected implementation chooses a deterministic orthogonal great-circle direction in that case.

### 1.8 Counter-Flow copying had recursive memory risk

Deep-copying the live transformer also copied its registry and prior Counter-Flow objects. Repeating this operation could copy historical copies inside newer historical copies and, in a registry, copy peer models as well.

The corrected snapshot constructs a clean, unregistered architecture clone, loads only the current state dictionary, clears history, and freezes its parameters.

Even after this fix, retaining K full snapshots of a P-parameter model costs O(KP) parameter storage. A production design should use delta checkpoints, low-rank parameter differences, periodic keyframes, or external immutable storage.

## 2. Minimum mathematical object now implemented

Let model i at update step t have parameters W_i(t), hidden output H_i(t), and direction d_i(t) in S^2.

### Direction observation

A shared projection frame P in R^(d_model x 3) produces:

    u_i(t) = mean_tokens_and_batch(H_i(t)) P
    target_i(t) = normalise(u_i(t))

### Direction dynamics

    d_i(t + 1) = normalise(m d_i(t) + (1 - m) target_i(t))

with momentum m in [0, 1].

This is an extrinsic normalised average, not the intrinsic Frechet mean on S^2. For small angular steps it is a reasonable approximation. For large or antipodal revisions, a geodesic update should be used instead.

### Local attention-projection adaptation

For the true input x to the output projection and its output y:

    W_next = W + clipped_eta * (E[y x^T] - diag(E[y^2]) W)

### Local predictive-bias adaptation

For output mean y_bar and exponential moving target m_t:

    b_next = b - eta * (y_bar - m_t)
    m_next = rho m_t + (1 - rho) y_bar

### Graph formation

A directed edge i -> j is eligible when:

    d_i dot d_j >= tau

A threshold must be reported together with registry size and its random-null expected degree. Reporting tau alone is statistically incomplete.

## 3. What remains mathematically absent

### 3.1 No task-level objective

There is no scalar objective connecting local updates, directions, graph formation, and useful prediction. The system adapts, but "adaptation" is not synonymous with learning. A publishable formulation needs an explicit objective or a clearly specified online-regret criterion.

### 3.2 No convergence or boundedness result

Oja stabilisation controls one source of norm growth, but the coupled system includes residual blocks, bias adaptation, moving directions, graph rewiring, and optional external gradient training. No theorem or empirical bound shows that the joint dynamics remain stable.

At minimum, experiments must report:

- parameter-norm trajectories
- activation norms by layer
- angular velocity of d
- update-to-weight norm ratios
- loss before and after each local update
- sensitivity to eta, rho, and direction momentum

### 3.3 No evidence that three dimensions are sufficient

S^2 has only two intrinsic degrees of freedom. Compressing an entire transformer's current behaviour to one point on S^2 is an extreme bottleneck. The repository offers no distortion analysis, neighbourhood preservation score, mutual-information estimate, or comparison against larger spheres.

Required ablation:

    direction dimension in {2, 3, 8, 32, 128}

with held-out task utility, neighbour precision, and stability reported.

### 3.4 No semantic identifiability result

Even with a shared projection frame, independently trained hidden spaces may undergo rotations, permutations, scalings, and nonlinear reparameterisations. Cosine similarity between compressed hidden means is not automatically a similarity between hypotheses.

Required validation should compare direction cosine against behavioural similarity on a shared probe set, such as output-distribution Jensen-Shannon divergence or agreement on labelled examples.

### 3.5 Counter-Flow is stored but not part of the learning law

The repository describes distillation and regularisation, but the forward update never uses a Counter-Flow target. Until an explicit term such as

    lambda * D(current_output, frozen_previous_output)

is implemented and tested, Counter-Flow is checkpointing with philosophical branding, not a dual-flow optimisation mechanism.

### 3.6 Connections do not perform computation

The graph is metadata. A forged connection neither exchanges activations nor changes a prediction. The project therefore does not yet implement a network of cooperating transformers; it implements a registry of similarity records.

A complete model must define a message function, aggregation rule, timing model, normalisation, and protection against positive-feedback collapse.

### 3.7 No statistical testing or uncertainty

There are no repeated seeds, confidence intervals, hypothesis tests, or correction for multiple peer comparisons. Selecting the maximum cosine from many peers creates an extreme-value effect even under randomness.

### 3.8 No credible baselines

At minimum, compare against:

- the same transformer with no forward adaptation
- backpropagation on the same data budget
- test-time training or entropy minimisation
- Hebbian-only and predictive-only variants
- random graph rewiring with matched degree
- cosine matching in the full hidden space
- larger learned embeddings with nearest-neighbour search
- exponential moving average checkpoints without Counter-Flow terminology

### 3.9 No benchmark defining success

Shape tests cannot establish learning. The project needs at least one online non-stationary task where the desired properties are measurable: adaptation speed, retained performance, forgetting, calibration, and compute cost.

## 4. Minimum experimental acceptance criteria

The architecture should not be described as validated until all of the following hold:

1. Direction modulation produces a reproducible task-level gain over an otherwise identical attention block.
2. Local updates reduce a declared local or global objective more often than chance and remain numerically bounded.
3. Direction cosine predicts behavioural similarity on held-out probes.
4. Alignment thresholds outperform the calibrated random null after correction for the number of peers searched.
5. Counter-Flow regularisation improves the adaptation-retention trade-off against ordinary checkpoint or EMA baselines.
6. Graph connections perform a defined computation and beat a degree-matched random graph.
7. Results hold across multiple seeds with uncertainty intervals and ablations.
8. Memory and runtime are reported as functions of model size, sequence length, swarm size, and retained history.

## 5. Current scientific status

After this patch, Heraclitus is a more internally consistent research prototype with falsifiable mechanisms and basic mathematical diagnostics. It is not yet evidence of a new transformer paradigm, a successful continual-learning method, or a meaningful distributed intelligence system. Those are hypotheses to test, not conclusions supported by the present repository.
