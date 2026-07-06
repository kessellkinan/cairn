from __future__ import annotations

import torch

from cairn.headwall.field_ode import ODEField
from cairn.kit.wiring import ExperimentConfig


def _field(config: ExperimentConfig, use_physics: bool) -> ODEField:
    return ODEField(
        config.model.latent_dim,
        config.model.dynamics,
        config.model.growth,
        use_physics=use_physics,
        immune_coupling=True,
    )


def test_rollout_returns_expected_length(smoke: ExperimentConfig) -> None:
    field = _field(smoke, use_physics=True)
    z0 = torch.randn(3, smoke.model.latent_dim)
    out = field.rollout(z0, n_steps=4, step_days=90.0)
    assert out.states.shape == (5, 3, smoke.model.latent_dim)
    assert out.times.shape == (5,)


def test_physics_derivative_only_touches_first_k(smoke: ExperimentConfig) -> None:
    field = _field(smoke, use_physics=True)
    z = torch.randn(3, smoke.model.latent_dim).abs() + 0.1
    deriv = field.physics_derivative(z)
    k = smoke.model.dynamics.physics_dims
    assert torch.count_nonzero(deriv[:, k:]) == 0


def test_disabled_physics_is_zero(smoke: ExperimentConfig) -> None:
    field = _field(smoke, use_physics=False)
    z = torch.randn(3, smoke.model.latent_dim)
    assert torch.allclose(field.physics_derivative(z), torch.zeros_like(z))
