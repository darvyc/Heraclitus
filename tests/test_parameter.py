import copy

import pytest
import torch

from heraclitus import HeraclitusConfig, HeraclitusParameter


def make_parameter(hidden_size: int = 12, state_size: int = 4) -> HeraclitusParameter:
    torch.manual_seed(7)
    return HeraclitusParameter(
        HeraclitusConfig(
            hidden_size=hidden_size,
            state_size=state_size,
            dropout=0.0,
        )
    )


def test_shape_dtype_and_finite_values():
    module = make_parameter()
    x = torch.randn(2, 5, 12, dtype=torch.float32)
    y = module(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_parameter_count_is_exact():
    module = make_parameter(hidden_size=12, state_size=4)
    actual = sum(parameter.numel() for parameter in module.parameters())
    assert actual == HeraclitusParameter.parameter_count(12, 4)


def test_forward_does_not_mutate_parameters():
    module = make_parameter()
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    module(torch.randn(2, 5, 12))
    after = dict(module.named_parameters())
    for name, value in before.items():
        assert torch.equal(value, after[name])


def test_gradients_flow_to_all_parameter_groups():
    module = make_parameter()
    x = torch.randn(2, 5, 12, requires_grad=True)
    result = module.forward_with_state(x, detach_state=False)
    loss = result.hidden_states.square().mean() + result.regularization_loss()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_causality_future_tokens_do_not_change_past_outputs():
    module = make_parameter().eval()
    prefix = torch.randn(2, 4, 12)
    future_a = torch.randn(2, 3, 12)
    future_b = torch.randn(2, 3, 12) * 20.0
    output_a = module(torch.cat([prefix, future_a], dim=1))
    output_b = module(torch.cat([prefix, future_b], dim=1))
    assert torch.allclose(output_a[:, :4], output_b[:, :4], atol=1e-6, rtol=1e-6)


def test_chunked_state_matches_single_pass():
    module = make_parameter().eval()
    x = torch.randn(2, 9, 12)
    full = module.forward_with_state(x)
    first = module.forward_with_state(x[:, :4])
    second = module.forward_with_state(x[:, 4:], state=first.state)
    chunked = torch.cat([first.hidden_states, second.hidden_states], dim=1)
    assert torch.allclose(full.hidden_states, chunked, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full.state.live, second.state.live, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full.state.counter, second.state.counter, atol=1e-6, rtol=1e-6)
    assert torch.equal(full.state.steps, second.state.steps)


def test_batch_isolation():
    module = make_parameter().eval()
    x = torch.randn(2, 6, 12)
    together = module(x)
    separate = torch.cat([module(x[index : index + 1]) for index in range(2)], dim=0)
    assert torch.allclose(together, separate, atol=1e-6, rtol=1e-6)


def test_masked_tokens_are_identity_and_do_not_advance_state():
    module = make_parameter().eval()
    x = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 1]], dtype=torch.bool)
    result = module.forward_with_state(x, attention_mask=mask)
    assert torch.equal(result.state.steps, mask.sum(dim=1))
    assert torch.allclose(result.hidden_states[~mask], x[~mask], atol=0.0, rtol=0.0)


def test_state_dict_round_trip():
    module = make_parameter().eval()
    clone = make_parameter().eval()
    clone.load_state_dict(copy.deepcopy(module.state_dict()))
    x = torch.randn(2, 5, 12)
    assert torch.allclose(module(x), clone(x), atol=0.0, rtol=0.0)


def test_half_precision_input_preserves_dtype():
    module = make_parameter().eval()
    x = torch.randn(2, 5, 12, dtype=torch.float16)
    y = module(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()


def test_regularization_is_finite_and_nonnegative():
    module = make_parameter()
    result = module.forward_with_state(torch.randn(2, 5, 12))
    for value in result.regularization.values():
        assert torch.isfinite(value)
        assert float(value.detach()) >= 0.0
    loss = result.regularization_loss()
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0.0


def test_invalid_shapes_are_rejected():
    module = make_parameter()
    with pytest.raises(ValueError):
        module(torch.randn(2, 12))
    with pytest.raises(ValueError):
        module(torch.randn(2, 5, 11))
    with pytest.raises(ValueError):
        module(torch.randn(2, 5, 12), attention_mask=torch.ones(2, 4))


def test_runtime_state_is_reorderable_and_checkpointable():
    module = make_parameter().eval()
    result = module.forward_with_state(torch.randn(3, 4, 12))
    indices = torch.tensor([2, 0, 2], dtype=torch.long)
    reordered = result.state.index_select(indices)
    restored = type(reordered).from_dict(reordered.as_dict())
    assert torch.equal(restored.steps, result.state.steps.index_select(0, indices))
    assert torch.allclose(restored.live, result.state.live.index_select(0, indices))
    assert torch.allclose(restored.counter, result.state.counter.index_select(0, indices))


def test_effective_matrices_and_residual_obey_declared_bounds():
    module = make_parameter().eval()
    with torch.no_grad():
        module.projection.mul_(1000.0)
        module.reconstruction.mul_(1000.0)
    assert float(module.effective_projection().norm(p="fro").detach()) <= 1.0 + 1e-5
    assert float(module.effective_reconstruction().norm(p="fro").detach()) <= 1.0 + 1e-5
    x = torch.randn(2, 6, 12)
    delta = module(x) - x
    per_token = delta.float().norm(dim=-1)
    assert float(per_token.max().detach()) <= module.residual_bound(training=False) + 1e-5
