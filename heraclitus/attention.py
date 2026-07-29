"""Direction-conditioned multi-head self-attention.

The directional term must vary along the key-token axis. A scalar added to an
entire attention row is annihilated by softmax translation invariance and has
no effect on the output.
"""
from __future__ import annotations

import math
from typing import Tuple, Union

import torch
from torch import Tensor, nn


class DirectionModulatedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Each key token receives a learned 3D axis per head. Dotting these axes
        # with the live direction creates a key-dependent logit bias, which does
        # survive softmax because it varies across the normalisation dimension.
        self.direction_keys = nn.Linear(d_model, n_heads * 3, bias=False)
        self.direction_scale = nn.Parameter(torch.full((n_heads,), 0.1))

    def forward(
        self,
        x: Tensor,
        direction: Tensor,
        return_context: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # x: (B, T, d_model), direction: (3,)
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        logits = (q @ k.transpose(-2, -1)) * scale

        unit_direction = direction / direction.norm().clamp(min=1e-8)
        token_axes = self.direction_keys(x).reshape(b, t, self.n_heads, 3)
        token_axes = token_axes.transpose(1, 2)
        token_axes = token_axes / token_axes.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        key_bias = torch.einsum("bhtc,c->bht", token_axes, unit_direction)
        key_bias = key_bias * self.direction_scale.view(1, self.n_heads, 1)
        logits = logits + key_bias.unsqueeze(-2)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        context = attn @ v
        merged_context = context.transpose(1, 2).reshape(b, t, self.d_model)
        output = self.out(merged_context)
        if return_context:
            return output, merged_context
        return output
