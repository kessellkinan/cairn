from __future__ import annotations

import torch

from cairn.headwall.constraint import physics_loss
from cairn.headwall.field_ode import ODEField
from cairn.kit.wiring import ExperimentConfig


def _field(config: ExperimentConfig) -> ODEField:
    return ODEField(
        config.model.latent_dim,
        config.model.dynamics,
        config.model.growth,
        use_physics=True,
        immune_coupling=True,
    )


def test_bound_penalty_grows_with_violation(smoke: ExperimentConfig) -> None:
    field = _field(smoke)
    for p in field.residual.parameters():
        torch.nn.init.zeros_(p)
    valid = torch.ones(4, 1, smoke.model.latent_dim)
    violating = valid.clone()
    violating[..., 0] = -5.0
    loss_valid = physics_loss(field, valid, smoke.model.dynamics)
    loss_bad = physics_loss(field, violating, smoke.model.dynamics)
    assert loss_bad > loss_valid


def test_loss_nonnegative(smoke: ExperimentConfig) -> None:
    field = _field(smoke)
    states = torch.randn(6, 1, smoke.model.latent_dim)
    assert physics_loss(field, states, smoke.model.dynamics) >= 0.0
