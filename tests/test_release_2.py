import torch

from heraclitus import HeraclitusConfig, HeraclitusParameter, HeraclitusState


def make_module() -> HeraclitusParameter:
    torch.manual_seed(7)
    return HeraclitusParameter(
        HeraclitusConfig(
            hidden_size=16,
            state_size=8,
            num_modes=3,
            covariance_rank=2,
            transition_reflections=3,
        )
    ).eval()


def test_shapes_probabilities_and_finite_values() -> None:
    module = make_module()
    hidden = torch.randn(2, 5, 16)
    result = module.forward_with_state(hidden, detach_state=False)
    assert result.hidden_states.shape == hidden.shape
    assert result.state.mode_means.shape == (2, 3, 8)
    assert result.state.covariance_factors.shape == (2, 3, 8, 2)
    assert torch.allclose(result.state.mode_log_weights.exp().sum(-1), torch.ones(2), atol=1e-5)
    assert torch.isfinite(result.hidden_states).all()
    assert torch.isfinite(result.regularization_loss())


def test_modes_are_persistent_not_offsets_around_one_mean() -> None:
    module = make_module()
    state = module.initial_state(1)
    state = HeraclitusState(
        mode_means=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        ),
        mode_variances=state.mode_variances,
        covariance_factors=state.covariance_factors,
        mode_log_weights=state.mode_log_weights,
        steps=state.steps,
    )
    means, _, _ = module.predictive_distribution(state)
    assert not torch.allclose(means[:, 0], means[:, 1])
    assert not torch.allclose(means[:, 1], means[:, 2])


def test_chunk_equivalence() -> None:
    module = make_module()
    hidden = torch.randn(2, 7, 16)
    whole = module.forward_with_state(hidden, detach_state=False)
    first = module.forward_with_state(hidden[:, :3], detach_state=False)
    second = module.forward_with_state(hidden[:, 3:], state=first.state, detach_state=False)
    joined = torch.cat([first.hidden_states, second.hidden_states], dim=1)
    assert torch.allclose(whole.hidden_states, joined, atol=1e-5, rtol=1e-5)
    assert torch.allclose(whole.state.mode_means, second.state.mode_means, atol=1e-5, rtol=1e-5)


def test_masked_tokens_are_identity_and_do_not_advance_state() -> None:
    module = make_module()
    hidden = torch.randn(1, 4, 16)
    mask = torch.tensor([[True, True, False, False]])
    result = module.forward_with_state(hidden, attention_mask=mask)
    assert torch.equal(result.hidden_states[:, 2:], hidden[:, 2:])
    assert result.state.steps.item() == 2


def test_householder_transition_preserves_norm_before_contraction() -> None:
    module = make_module()
    values = torch.randn(2, 3, 8)
    rotated = module._apply_orthogonal_transition(values)
    assert torch.allclose(values.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_gradients_reach_projection_transition_and_noise() -> None:
    module = make_module().train()
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    result = module.forward_with_state(hidden, detach_state=False)
    loss = result.hidden_states.square().mean() + result.regularization_loss()
    loss.backward()
    assert module.projection.grad is not None
    assert module.transition_vectors.grad is not None
    assert module.process_noise_logits.grad is not None
    assert torch.isfinite(module.transition_vectors.grad).all()


def test_legacy_num_shadows_alias() -> None:
    config = HeraclitusConfig(hidden_size=16, state_size=8, num_shadows=5)
    assert config.num_modes == 5
    assert config.num_shadows == 5
