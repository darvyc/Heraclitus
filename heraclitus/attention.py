"""Direction-modulated multi-head self-attention.

A standard scaled-dot-product attention block, except that the per-head logits
are biased by a 3D-direction-derived term. This lets the transformer's current
S^2 orientation continuously shape what it attends to, without having to
re-allocate any structural capacity.
"""
from __future__ import annotations

import math
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

        # Per-head 3-vector that gets dotted with the module's live direction.
        # The resulting scalar bias gently warps each head's logits.
        self.head_axes = nn.Parameter(torch.randn(n_heads, 3) * 0.1)

    def forward(self, x: Tensor, direction: Tensor) -> Tensor:
        # x: (B, T, d_model)  direction: (3,)
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                   # (B, T, H, Dh) each
        q = q.transpose(1, 2)                         # (B, H, T, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        logits = (q @ k.transpose(-2, -1)) * scale    # (B, H, T, T)

        # Direction bias: one scalar per head, broadcast over the logit grid.
        head_bias = (self.head_axes @ direction)      # (H,)
        logits = logits + head_bias.view(1, self.n_heads, 1, 1)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(b, t, self.d_model)
        return self.out(out)
