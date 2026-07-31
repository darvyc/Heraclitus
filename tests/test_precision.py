import pytest
import torch

from heraclitus import HeraclitusAdapter, HeraclitusConfig


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_returned_reduced_precision_residual_is_strictly_bounded(
    dtype: torch.dtype,
) -> None:
    maximum_ratio = 0.10
    module = HeraclitusAdapter(
        HeraclitusConfig(
            hidden_size=12,
            state_size=4,
            memory_slots=3,
            num_heads=2,
            write_topk=1,
            max_residual_ratio=maximum_ratio,
        )
    ).to(dtype=dtype).eval()
    with torch.no_grad():
        module.output.weight.fill_(1000.0)

    hidden = torch.ones(2, 5, 12, dtype=dtype)
    result = module.forward_with_state(hidden)
    actual_delta = result.hidden_states.float() - hidden.float()
    actual_ratio = actual_delta.norm(dim=-1) / hidden.float().norm(dim=-1)

    assert float(actual_ratio.max().detach()) <= maximum_ratio
    assert float(result.diagnostics.maximum_residual_ratio.detach()) <= maximum_ratio
