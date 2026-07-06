from __future__ import annotations

import torch

from cairn.headwall.growth import GrowthODE
from cairn.kit.wiring import ExperimentConfig


def test_pure_gompertz_when_immune_disabled(smoke: ExperimentConfig) -> None:
    ode = GrowthODE(smoke.model.growth, immune_coupling=False)
    state = torch.tensor([[2.0, 1.0], [4.0, 0.5]])
    deriv = ode(state)
    assert torch.allclose(deriv[:, 1], torch.zeros(2))
    alpha = torch.nn.functional.softplus(ode.alpha) + 1e-4
    capacity = torch.nn.functional.softplus(ode.carrying_capacity) + 1e-4
    expected = alpha * state[:, 0] * torch.log(capacity / state[:, 0])
    assert torch.allclose(deriv[:, 0], expected, atol=1e-5)


def test_immune_term_changes_volume_dynamics(smoke: ExperimentConfig) -> None:
    with_immune = GrowthODE(smoke.model.growth, immune_coupling=True)
    state = torch.tensor([[3.0, 2.0]])
    deriv = with_immune(state)
    assert deriv[:, 1].abs().item() > 0.0


def test_parameters_receive_gradients(smoke: ExperimentConfig) -> None:
    ode = GrowthODE(smoke.model.growth, immune_coupling=True)
    state = torch.tensor([[1.5, 0.8]], requires_grad=True)
    ode(state).sum().backward()
    assert ode.alpha.grad is not None
