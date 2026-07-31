"""Executable Heraclitus 3 usage example."""
import torch

from heraclitus import HeraclitusAdapter, HeraclitusConfig


def main() -> None:
    adapter = HeraclitusAdapter(
        HeraclitusConfig(
            hidden_size=64,
            state_size=16,
            memory_slots=8,
            num_heads=4,
            write_topk=2,
        )
    ).eval()
    first = adapter.forward_with_state(torch.randn(2, 12, 64))
    second = adapter.forward_with_state(torch.randn(2, 5, 64), state=first.state)
    print(tuple(second.hidden_states.shape))
    print(second.state.steps.tolist())
    print(float(second.diagnostics.maximum_residual_ratio.detach()))


if __name__ == "__main__":
    main()
