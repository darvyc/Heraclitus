# Heraclitus Release Edition — Mathematical Specification

## 1. State space

Let the transformer hidden width be `d`, latent dimension `r`, number of persistent modes `K`, and covariance rank `c`. For token `t` and mode `k`,

\[
\mathcal S_{t,k}=(\mu_{t,k},D_{t,k},U_{t,k},\pi_{t,k}),
\]

where

\[
\mu_{t,k}\in\mathbb R^r,
\quad D_{t,k}=\operatorname{diag}(d_{t,k})\succ0,
\quad U_{t,k}\in\mathbb R^{r\times c},
\quad \pi_{t,k}\ge0,
\quad \sum_k\pi_{t,k}=1.
\]

The mode covariance is

\[
P_{t,k}=D_{t,k}+U_{t,k}U_{t,k}^{\top}\succ0.
\]

The latent posterior is the Gaussian mixture

\[
p(s_t\mid z_{1:t})=\sum_{k=1}^{K}\pi_{t,k}\mathcal N(s_t;\mu_{t,k},P_{t,k}).
\]

The projected observation is

\[
z_t=W_p^{\top}\operatorname{RMSNorm}(x_t).
\]

## 2. Mixture moments

The aggregate mean is

\[
\bar\mu_t=\sum_k\pi_{t,k}\mu_{t,k}.
\]

By the law of total covariance,

\[
\bar P_t=\sum_k\pi_{t,k}\left[P_{t,k}+(\mu_{t,k}-\bar\mu_t)(\mu_{t,k}-\bar\mu_t)^{\top}\right].
\]

This decomposes uncertainty into within-mode covariance and between-mode disagreement.

## 3. Stable higher-dimensional dynamics

For learned vectors `v_j`, define Householder reflections

\[
H_j=I-2\frac{v_jv_j^{\top}}{v_j^{\top}v_j},
\qquad H_j^{\top}H_j=I.
\]

The orthogonal transport is

\[
Q=H_mH_{m-1}\cdots H_1,
\qquad Q^{\top}Q=I.
\]

For each mode,

\[
a_{t,k}=a_{\min}+(a_{\max}-a_{\min})\sigma(\alpha_k+C\bar\mu_{t-1}),
\]

with

\[
0\le a_{\min}<a_{\max}<1.
\]

The transition operator is

\[
A_{t,k}=\operatorname{diag}(a_{t,k})Q.
\]

Since `Q` is orthogonal,

\[
\|A_{t,k}\|_2\le a_{\max}<1,
\]

so

\[
\|A_{t,k}u-A_{t,k}v\|_2\le a_{\max}\|u-v\|_2.
\]

The prior mean is

\[
\mu^-_{t,k}=A_{t,k}\mu_{t-1,k}.
\]

## 4. Covariance propagation

The exact linear prediction is

\[
P^-_{t,k}=A_{t,k}P_{t-1,k}A_{t,k}^{\top}+Q_{t,k}.
\]

The structured representation is propagated as

\[
D^-_{t,k}=\operatorname{diag}(a_{t,k}^{\odot2}\odot d_{t-1,k}+q_{t,k}),
\]

\[
U^-_{t,k}=\operatorname{diag}(a_{t,k})QU_{t-1,k}+B_k,
\]

with positive context-dependent process noise

\[
q_{t,k}=q_{\min}+\operatorname{softplus}(\beta_k+G_q\bar\mu^-_t).
\]

Thus

\[
P^-_{t,k}=D^-_{t,k}+U^-_{t,k}(U^-_{t,k})^{\top}.
\]

## 5. Observation model

The latent observation model is

\[
z_t=s_t+\varepsilon_t,
\qquad \varepsilon_t\sim\mathcal N(0,R_t),
\]

where

\[
R_t=\operatorname{diag}(r_t),
\qquad
r_t=r_{\min}+\operatorname{softplus}(\gamma+G_r\bar\mu^-_t).
\]

The predictive observation covariance is

\[
S_{t,k}=P^-_{t,k}+R_t
=\Delta_{t,k}+U^-_{t,k}(U^-_{t,k})^{\top},
\]

with

\[
\Delta_{t,k}=D^-_{t,k}+R_t.
\]

## 6. Woodbury inverse and determinant lemma

Define

\[
M_{t,k}=I_c+(U^-_{t,k})^{\top}\Delta_{t,k}^{-1}U^-_{t,k}.
\]

Then

\[
S_{t,k}^{-1}=\Delta_{t,k}^{-1}-\Delta_{t,k}^{-1}U^-_{t,k}M_{t,k}^{-1}(U^-_{t,k})^{\top}\Delta_{t,k}^{-1}.
\]

The determinant lemma gives

\[
\log\det S_{t,k}=\log\det\Delta_{t,k}+\log\det M_{t,k}.
\]

For innovation

\[
e_{t,k}=z_t-\mu^-_{t,k},
\]

the Gaussian log likelihood is

\[
\ell_{t,k}=-\frac12\left[e_{t,k}^{\top}S_{t,k}^{-1}e_{t,k}+\log\det S_{t,k}+r\log(2\pi)\right].
\]

Only a `c x c` matrix is inverted.

## 7. Markov mode dynamics

Let `T_t` be row-stochastic:

\[
T_t=\operatorname{softmax}_{\mathrm{row}}(\Theta+\mathcal C(\bar\mu_{t-1})).
\]

The prior mode probability is

\[
\pi^-_{t,k}=\sum_j\pi_{t-1,j}T_{t,jk}.
\]

Bayesian evidence gives

\[
\pi_{t,k}=\frac{\pi^-_{t,k}\exp(\ell_{t,k})}{\sum_j\pi^-_{t,j}\exp(\ell_{t,j})}.
\]

A probability floor is applied before renormalisation to prevent numerical extinction.

## 8. Posterior correction

The gain is

\[
K_{t,k}=P^-_{t,k}S_{t,k}^{-1}.
\]

The posterior mean is

\[
\mu_{t,k}=\mu^-_{t,k}+K_{t,k}e_{t,k}.
\]

The Joseph covariance form is

\[
P_{t,k}=(I-K_{t,k})P^-_{t,k}(I-K_{t,k})^{\top}+K_{t,k}R_tK_{t,k}^{\top}.
\]

This preserves positive semidefiniteness under finite precision. The resulting covariance is projected back onto

\[
\mathcal M_c=\{D+UU^{\top}:D\succ0,\ U\in\mathbb R^{r\times c}\}.
\]

If

\[
P_{t,k}=V\Lambda V^{\top},
\]

retain the leading `c` eigenpairs:

\[
U_{t,k}=V_{1:c}\Lambda_{1:c}^{1/2},
\]

and assign the remaining marginal energy to

\[
d_{t,k}=\operatorname{diag}(P_{t,k}-U_{t,k}U_{t,k}^{\top})+\epsilon.
\]

This is an assumed-density projection, not an exact dense-covariance filter.

## 9. Information geometry

The Fisher metric for the Gaussian mean is

\[
\mathcal I_{\mu}=S_{t,k}^{-1}.
\]

The whitened innovation

\[
w_{t,k}=S_{t,k}^{-1/2}e_{t,k}
\]

has squared norm

\[
\|w_{t,k}\|_2^2=e_{t,k}^{\top}S_{t,k}^{-1}e_{t,k},
\]

which is Mahalanobis surprise and is invariant under invertible linear reparameterisations. The update

\[
P^-_{t,k}S_{t,k}^{-1}e_{t,k}
\]

is a covariance-preconditioned natural-gradient step in latent space.

## 10. Reliability-gated residual write

The mixture innovation is

\[
\bar e_t=z_t-\bar\mu^-_t.
\]

Let the predictive mixture covariance be

\[
\bar S_t=\sum_k\pi^-_{t,k}\left[S_{t,k}+(\mu^-_{t,k}-\bar\mu^-_t)(\mu^-_{t,k}-\bar\mu^-_t)^{\top}\right].
\]

The normalised innovation is

\[
\hat e_t=\bar S_t^{-1/2}\bar e_t.
\]

Novelty and reliability are

\[
g_{\mathrm{nov},t}=\sigma(a_n\|\hat e_t\|_2^2+b_n),
\]

\[
g_{\mathrm{rel},t}=\exp\left(-a_r\frac{\operatorname{tr}(\bar S_t)}{r}\right).
\]

The residual correction is

\[
\delta_t=\rho_{\max}\sigma(\rho)g_{\mathrm{nov},t}g_{\mathrm{rel},t}W_r^{\top}\hat e_t,
\qquad y_t=x_t+\delta_t.
\]

This separates novelty from trustworthiness and bounds the intervention.

## 11. Information-preservation objectives

Let `C_z` be the batch covariance of projected observations. A variance floor is

\[
\mathcal L_{\mathrm{var}}=\frac1r\sum_i\max(0,\tau-\sqrt{(C_z)_{ii}+\epsilon})^2.
\]

Projection redundancy is penalised by

\[
\mathcal L_{\mathrm{decor}}=\frac1{r(r-1)}\sum_{i\ne j}(C_z)_{ij}^2.
\]

Mode distinctness uses symmetric KL separation:

\[
\mathcal L_{\mathrm{mode}}=\frac{2}{K(K-1)}\sum_{i<j}\max\left(0,\tau_{\mathrm{KL}}-\frac12[D_{\mathrm{KL}}(i\|j)+D_{\mathrm{KL}}(j\|i)]\right).
\]

Its strength is weighted by posterior ambiguity

\[
w_t=\frac{H(\pi_t)}{\log K},
\]

so separation pressure is strongest only when several modes remain plausible.

## 12. Calibration

Define the posterior-weighted squared Mahalanobis statistic

\[
q_t=\sum_k\pi^-_{t,k}e_{t,k}^{\top}S_{t,k}^{-1}e_{t,k}.
\]

For a calibrated `r`-dimensional Gaussian model,

\[
\mathbb E[q_t]\approx r,
\qquad
\operatorname{Var}(q_t)\approx2r.
\]

A moment calibration penalty is

\[
\mathcal L_{\mathrm{cal}}=(\mathbb E[q_t]-r)^2+\lambda_q(\operatorname{Var}(q_t)-2r)^2.
\]

## 13. Training objective

The auxiliary objective is

\[
\mathcal L_{\mathrm{aux}}=
\lambda_{\mathrm{nll}}\mathcal L_{\mathrm{nll}}+
\lambda_{\mathrm{mode}}\mathcal L_{\mathrm{mode}}+
\lambda_{\mathrm{var}}\mathcal L_{\mathrm{var}}+
\lambda_{\mathrm{decor}}\mathcal L_{\mathrm{decor}}+
\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{cal}}+
\lambda_{\mathrm{orth}}\|W_p^{\top}W_p-I\|_F^2+
\lambda_{\mathrm{energy}}\mathcal L_{\mathrm{residual}}.
\]

The complete objective is

\[
\mathcal L=\mathcal L_{\mathrm{task}}+\mathcal L_{\mathrm{aux}}.
\]

## 14. Stochastic stability bound

Assume

\[
\sup_{t,k}\|A_{t,k}\|_2\le\alpha<1,
\qquad
\sup_{t,k}\operatorname{tr}(Q_{t,k})\le q_{\max}.
\]

Then

\[
\mathbb E\|s_t\|_2^2\le\alpha^2\mathbb E\|s_{t-1}\|_2^2+q_{\max}.
\]

Iterating,

\[
\mathbb E\|s_t\|_2^2\le\alpha^{2t}\mathbb E\|s_0\|_2^2+\frac{q_{\max}(1-\alpha^{2t})}{1-\alpha^2}.
\]

Therefore the latent second moment is uniformly bounded.

## 15. Causality and chunk equivalence

At token `t`, output depends only on `x_t`, the incoming state and learned parameters. The outgoing state is a deterministic function of those quantities. By induction, no future token affects a past output. Because the recurrence is Markovian in the explicit continuation state, processing adjacent chunks with state hand-off reproduces full-sequence evaluation under deterministic settings.

## 16. Complexity

For batch `B`, sequence length `T`, modes `K`, latent width `r`, rank `c`, and hidden width `d`:

- projection and reconstruction: `O(BTdr)`;
- mode propagation: `O(BTKrc)`;
- Woodbury likelihoods: `O(BTK(rc^2+c^3))`;
- continuation-state memory: `O(BK(r+rc))`.

When `c << r`, correlated uncertainty is substantially cheaper than dense `r x r` covariance algebra.

## 17. Falsifiability

The mathematical contract guarantees causality, explicit state, probability normalisation, covariance positivity under the stated parameterisation, contractive deterministic transport, bounded latent second moments under bounded process noise and bounded residual intervention. It does not guarantee improved language modelling. That claim requires compute-matched experiments, ablations, confidence intervals and external task measurements.
