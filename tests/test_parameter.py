import copy

import pytest
import torch

from heraclitus import HeraclitusConfig, HeraclitusParameter, HeraclitusState


def make_parameter() -> HeraclitusParameter:
    torch.manual_seed(7)
    return HeraclitusParameter(
        HeraclitusConfig(
            hidden_size=12,
            state_size=4,
            memory_slots=3,
            num_heads=1,
            dropout=0.0,
        )
    )


def test_shape_dtype_and_noop_initialisation():
    module = make_parameter().eval()
    x = torch.randn(2, 5, 12)
    y = module(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.equal(y, x)


def test_gradients_are_finite():
    module = make_parameter()
    x = torch.randn(2, 5, 12, requires_grad=True)
    result = module.forward_with_state(x, detach_state=False)
    loss = result.hidden_states.square().mean() + result.regularization_loss()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_causality():
    module = make_parameter().eval()
    with torch.no_grad():
        module.output.weight.normal_(std=0.01)
    prefix = torch.randn(2, 4, 12)
    a = module(torch.cat([prefix, torch.randn(2, 3, 12)], dim=1))
    b = module(torch.cat([prefix, torch.randn(2, 3, 12) * 20], dim=1))
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6, rtol=1e-6)


def test_chunk_equivalence():
    module = make_parameter().eval()
    with torch.no_grad():
        module.output.weight.normal_(std=0.01)
    x = torch.randn(2, 9, 12)
    full = module.forward_with_state(x)
    first = module.forward_with_state(x[:, :4])
    second = module.forward_with_state(x[:, 4:], state=first.state)
    assert torch.allclose(full.hidden_states, torch.cat([first.hidden_states, second.hidden_states], 1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(full.state.memory, second.state.memory, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full.state.usage, second.state.usage, atol=1e-6, rtol=1e-6)
    assert torch.equal(full.state.steps, second.state.steps)


def test_masking_is_identity_and_does_not_advance_state():
    module = make_parameter().eval()
    with torch.no_grad():
        module.output.weight.normal_(std=0.01)
    x = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 1]], dtype=torch.bool)
    result = module.forward_with_state(x, attention_mask=mask)
    assert torch.equal(result.state.steps, mask.sum(dim=1))
    assert torch.equal(result.hidden_states[~mask], x[~mask])


def test_batch_isolation():
    module = make_parameter().eval()
    with torch.no_grad():
        module.output.weight.normal_(std=0.01)
    x = torch.randn(2, 6, 12)
    together = module(x)
    separate = torch.cat([module(x[i:i + 1]) for i in range(2)], dim=0)
    assert torch.allclose(together, separate, atol=1e-6, rtol=1e-6)


def test_state_round_trip_and_reordering():
    module = make_parameter().eval()
    state = module.forward_with_state(torch.randn(3, 4, 12)).state
    indices = torch.tensor([2, 0, 2])
    reordered = state.index_select(indices)
    restored = HeraclitusState.from_dict(reordered.as_dict())
    assert torch.equal(restored.steps, state.steps.index_select(0, indices))
    assert torch.allclose(restored.memory, state.memory.index_select(0, indices))
    assert torch.allclose(restored.usage, state.usage.index_select(0, indices))


def test_state_dict_round_trip():
    module = make_parameter().eval()
    clone = make_parameter().eval()
    clone.load_state_dict(copy.deepcopy(module.state_dict()))
    x = torch.randn(2, 5, 12)
    assert torch.equal(module(x), clone(x))


def test_half_precision_input_preserves_dtype():
    module = make_parameter().eval()
    x = torch.randn(2, 5, 12, dtype=torch.float16)
    y = module(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()


def test_diagnostics_and_regularisation_are_finite():
    result = make_parameter().forward_with_state(torch.randn(2, 5, 12))
    for value in result.regularization.values():
        assert torch.isfinite(value)
        assert float(value.detach()) >= 0.0
    for value in vars(result.diagnostics).values():
        assert torch.isfinite(value)
    assert 1.0 <= float(result.diagnostics.effective_slots) <= 3.0 + 1e-5


def test_invalid_inputs_are_rejected():
    module = make_parameter()
    with pytest.raises(ValueError):
        module(torch.randn(2, 12))
    with pytest.raises(ValueError):
        module(torch.randn(2, 5, 11))
    with pytest.raises(ValueError):
        module(torch.randn(2, 5, 12), attention_mask=torch.ones(2, 4))
    with pytest.raises(ValueError):
        HeraclitusConfig(hidden_size=12, state_size=4, memory_slots=0)
