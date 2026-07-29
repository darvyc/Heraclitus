"""Local, gradient-free adaptation rules with explicit stability controls."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ForwardLearner:
    def __init__(
        self,
        lr: float = 1e-3,
        ema_decay: float = 0.95,
        max_relative_update: float = 1e-3,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if max_relative_update <= 0:
            raise ValueError("max_relative_update must be positive")
        self.lr = lr
        self.ema_decay = ema_decay
        self.max_relative_update = max_relative_update

    @torch.no_grad()
    def hebbian_update(self, linear: nn.Linear, pre: Tensor, post: Tensor) -> None:
        """Apply a multi-output Oja update to the actual synaptic activities.

        delta = E[post pre^T] - diag(E[post^2]) W.
        The second term prevents unconstrained Hebbian norm growth. The final
        parameter displacement is clipped relative to the current weight norm.
        """
        pre_flat = pre.reshape(-1, pre.shape[-1])
        post_flat = post.reshape(-1, post.shape[-1])
        if pre_flat.shape[0] != post_flat.shape[0]:
            raise ValueError("pre and post must have the same sample count")
        if pre_flat.shape[1] != linear.in_features:
            raise ValueError("pre does not match linear.in_features")
        if post_flat.shape[1] != linear.out_features:
            raise ValueError("post does not match linear.out_features")

        covariance = post_flat.transpose(0, 1) @ pre_flat / max(pre_flat.shape[0], 1)
        post_power = post_flat.square().mean(dim=0, keepdim=True).transpose(0, 1)
        delta = covariance - post_power * linear.weight
        proposed = self.lr * delta

        weight_norm = linear.weight.norm().clamp(min=1e-8)
        max_update_norm = self.max_relative_update * weight_norm
        scale = torch.clamp(max_update_norm / proposed.norm().clamp(min=1e-8), max=1.0)
        linear.weight.add_(proposed * scale)

    @torch.no_grad()
    def predictive_update(
        self,
        bias: nn.Parameter,
        output: Tensor,
        ema_buffer: Tensor,
    ) -> Tensor:
        """Descend the local squared prediction error of an additive bias."""
        mean_output = output.reshape(-1, output.shape[-1]).mean(dim=0)
        surprise = mean_output - ema_buffer
        # For L = 0.5 ||output - ema||^2 and d(output)/d(bias) = I,
        # gradient descent requires subtraction. Addition amplifies surprise.
        bias.sub_(self.lr * surprise)
        return self.ema_decay * ema_buffer + (1.0 - self.ema_decay) * mean_output
