import copy

import pytest
import torch

from heraclitus import HeraclitusAdapter, HeraclitusConfig, HeraclitusState


def make_adapter(num_heads: int = 2) -> HeraclitusAdapter:
    torch.manual_seed(7)
    return HeraclitusAdapter(
        HeraclitusConfig(
            hidden_size=12,
            state_size=4,
            memory_slots=3,
            num_heads=num_heads,
            dropout=0.0,
        )
    )


def activate_output(module: HeraclitusAdapter) -> None:
    with torch.no_grad():
        module.output.weight.normal_(std=0.01)


def test_shape_dtype_and_exact_noop_initialisation():
    module = make_adapter().eval()
    x = torch.randn(2, 5, 12)
    y = module(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.equal(y, x)


def test_all_parameters_receive_finite_gradients_when_active():
    module = make_adapter()
    activate_output(module)
    x = torch.randn(2, 5, 12, requires_grad=True)
    result = module.forward_with_state(x, detach_state=False)
    loss = result.hidden_states.square().mean() + result.regularization_loss()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_forward_does_not_mutate_parameters():
    module = make_adapter().eval()
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    module(torch.randn(2, 5, 12))
    for name, value in module.named_parameters():
        assert torch.equal(value, before[name])


def test_parameter_count_is_exact():
    module = make_adapter()
    expected = HeraclitusAdapter.parameter_count(12, 4, memory_slots=3, num_heads=2)
    assert sum(parameter.numel() for parameter in module.parameters()) == expected


def test_causality():
    module = make_adapter().eval()
    activate_output(module)
    prefix = torch.randn(2, 4, 12)
    a = module(torch.cat([prefix, torch.randn(2, 3, 12)], dim=1))
    b = module(torch.cat([prefix, torch.randn(2, 3, 12) * 20], dim=1))
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6, rtol=1e-6)


def test_chunk_equivalence():
    module = make_adapter().eval()
    activate_output(module)
    x = torch.randn(2, 9, 12)
    full = module.forward_with_state(x)
    first = module.forward_with_state(x[:, :4])
    second = module.forward_with_state(x[:, 4:], state=first.state)
    chunked = torch.cat([first.hidden_states, second.hidden_states], dim=1)
    assert torch.allclose(full.hidden_states, chunked, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full.state.memory, second.state.memory, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full.state.usage, second.state.usage, atol=1e-6, rtol=1e-6)
    assert torch.equal(full.state.steps, second.state.steps)


def test_masking_is_identity_and_does_not_advance_state():
    module = make_adapter().eval()
    activate_output(module)
    x = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 1]], dtype=torch.bool)
    result = module.forward_with_state(x, attention_mask=mask)
    assert torch.equal(result.state.steps, mask.sum(dim=1))
    assert torch.equal(result.hidden_states[~mask], x[~mask])


def test_all_masked_sequence_is_an_identity_state_transition():
    module = make_adapter().eval()
    activate_output(module)
    x = torch.randn(2, 5, 12)
    initial = module.initial_state(2)
    result = module.forward_with_state(x, state=initial, attention_mask=torch.zeros(2, 5))
    assert torch.equal(result.hidden_states, x)
    assert torch.equal(result.state.memory, initial.memory)
    assert torch.equal(result.state.usage, initial.usage)
    assert torch.equal(result.state.steps, initial.steps)


def test_batch_isolation():
    module = make_adapter().eval()
    activate_output(module)
    x = torch.randn(2, 6, 12)
    together = module(x)
    separate = torch.cat([module(x[index : index + 1]) for index in range(2)], dim=0)
    assert torch.allclose(together, separate, atol=1e-6, rtol=1e-6)


def test_state_round_trip_and_reordering():
    module = make_adapter().eval()
    state = module.forward_with_state(torch.randn(3, 4, 12)).state
    indices = torch.tensor([2, 0, 2])
    reordered = state.index_select(indices)
    restored = HeraclitusState.from_dict(reordered.as_dict())
    assert torch.equal(restored.steps, state.steps.index_select(0, indices))
    assert torch.allclose(restored.memory, state.memory.index_select(0, indices))
    assert torch.allclose(restored.usage, state.usage.index_select(0, indices))


def test_state_dict_round_trip():
    module = make_adapter().eval()
    clone = make_adapter().eval()
    clone.load_state_dict(copy.deepcopy(module.state_dict()))
    x = torch.randn(2, 5, 12)
    assert torch.equal(module(x), clone(x))


def test_multi_head_path_is_operational():
    one_head = make_adapter(num_heads=1).eval()
    two_heads = make_adapter(num_heads=2).eval()
    activate_output(one_head)
    activate_output(two_heads)
    x = torch.randn(2, 5, 12)
    assert one_head(x).shape == two_heads(x).shape == x.shape
    assert not torch.equal(one_head(x), two_heads(x))


def test_half_precision_input_preserves_dtype():
    module = make_adapter().eval()
    x = torch.randn(2, 5, 12, dtype=torch.float16)
    y = module(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()


def test_long_horizon_state_remains_finite_and_bounded():
    module = make_adapter().eval()
    activate_output(module)
    state = None
    for _ in range(20):
        result = module.forward_with_state(torch.randn(2, 64, 12), state=state)
        state = result.state
    assert state is not None
    assert torch.isfinite(state.memory).all()
    assert torch.isfinite(state.usage).all()
    assert float(state.memory.abs().amax()) <= 1.0
    assert torch.equal(state.steps, torch.full((2,), 1280, dtype=torch.long))


def test_diagnostics_and_regularisation_are_finite():
    result = make_adapter().forward_with_state(torch.randn(2, 5, 12))
    for value in result.regularization.values():
        assert torch.isfinite(value)
        assert float(value.detach()) >= 0.0
    for value in vars(result.diagnostics).values():
        assert torch.isfinite(value)
    assert 1.0 <= float(result.diagnostics.effective_slots) <= 3.0 + 1e-5


def test_invalid_inputs_are_rejected():
    module = make_adapter()
    with pytest.raises(ValueError):
        module(torch.randn(2, 12))
    with pytest.raises(ValueError):
        module(torch.randn(2, 5, 11))
    with pytest.raises(ValueError):
        module(torch.randn(2, 5, 12), attention_mask=torch.ones(2, 4))
    with pytest.raises(ValueError):
        module(torch.full((2, 5, 12), float("nan")))
    with pytest.raises(ValueError):
        HeraclitusConfig(hidden_size=12, state_size=4, memory_slots=0)
    with pytest.raises(ValueError):
        HeraclitusConfig(hidden_size=12, state_size=4, num_heads=3)
    with pytest.raises(ValueError):
        result = module.forward_with_state(torch.randn(2, 5, 12))
        result.regularization_loss(usage_weight=-1.0)
