"""Stateful insertion of Heraclitus 3 into a transformer residual stream."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from heraclitus import HeraclitusAdapter, HeraclitusConfig, HeraclitusState


class ExampleTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, state_size: int) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.heraclitus = HeraclitusAdapter(
            HeraclitusConfig(
                hidden_size=hidden_size,
                state_size=state_size,
                memory_slots=8,
                num_heads=4,
                write_topk=2,
            )
        )

    def forward(
        self,
        hidden_states: Tensor,
        state: Optional[HeraclitusState] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, HeraclitusState]:
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
            key_padding_mask=None if attention_mask is None else ~attention_mask.bool(),
            need_weights=False,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        result = self.heraclitus.forward_with_state(
            hidden_states,
            state=state,
            attention_mask=attention_mask,
        )
        return result.hidden_states, result.state


def main() -> None:
    block = ExampleTransformerBlock(hidden_size=256, n_heads=8, state_size=32).eval()
    first, state = block(torch.randn(2, 16, 256))
    second, state = block(torch.randn(2, 4, 256), state=state)
    print(first.shape, second.shape, state.steps.tolist())


if __name__ == "__main__":
    main()
