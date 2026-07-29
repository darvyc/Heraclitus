"""Minimal insertion of Heraclitus into an LLM residual stream."""
import torch
from torch import nn

from heraclitus import HeraclitusConfig, HeraclitusParameter


class ExampleTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, state_size: int):
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, n_heads, batch_first=True
        )
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.heraclitus = HeraclitusParameter(
            HeraclitusConfig(hidden_size=hidden_size, state_size=state_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        normalised = self.norm_1(hidden_states)
        attended, _ = self.attention(
            normalised,
            normalised,
            normalised,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return self.heraclitus(hidden_states)


block = ExampleTransformerBlock(hidden_size=256, n_heads=8, state_size=32)
x = torch.randn(2, 16, 256)
y = block(x)
print(y.shape)
