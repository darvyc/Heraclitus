"""Forward-pass learning rules.

Two complementary, gradient-free updates are applied during `forward()`:

1. **Hebbian pre/post correlation** on the value-projection of each attention
   block. This nudges the projection toward its empirical input/output
   covariance, scaled by a small learning rate.

2. **Predictive-coding residual** that compares the block's output with a
   short EMA of past outputs. The discrepancy (the 'surprise') is used to
   adjust an additive bias on the output projection, so high-surprise tokens
   exert more pull than low-surprise ones.

Both rules use only locally available statistics — no global loss, no .backward().
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ForwardLearner:
    """Stateless container of forward-pass update rules."""

    def __init__(self, lr: float = 1e-3, ema_decay: float = 0.95):
        self.lr = lr
        self.ema_decay = ema_decay

    @torch.no_grad()
    def hebbian_update(self, linear: nn.Linear, pre: Tensor, post: Tensor) -> None:
        """Δ W ∝ <post · preᵀ> — vanilla Hebb with a unit-norm decay."""
        # pre:  (..., in_features)
        # post: (..., out_features)
        pre_flat = pre.reshape(-1, pre.shape[-1])          # (N, in)
        post_flat = post.reshape(-1, post.shape[-1])       # (N, out)
        n = max(pre_flat.shape[0], 1)
        # outer product mean: (out, in)
        delta = (post_flat.t() @ pre_flat) / n
        # normalise so a single update can't blow up the weight scale.
        delta = delta / (delta.norm() + 1e-8)
        linear.weight.add_(self.lr * delta)

    @torch.no_grad()
    def predictive_update(
        self,
        bias: nn.Parameter,
        output: Tensor,
        ema_buffer: Tensor,
    ) -> Tensor:
        """Update an additive bias toward reducing recent prediction error.

        Returns the updated EMA buffer so the caller can persist it.
        """
        flat = output.reshape(-1, output.shape[-1]).mean(dim=0)   # (out,)
        surprise = flat - ema_buffer                              # (out,)
        bias.add_(self.lr * surprise)
        new_ema = self.ema_decay * ema_buffer + (1 - self.ema_decay) * flat
        return new_ema
