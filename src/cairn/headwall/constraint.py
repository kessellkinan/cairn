"""Physics constraint loss.

Ref: Methods Sec. 4.5, Eq. (6). The matching term pins the learned dynamics to
the ODE-derived coordinates; the bound term enforces biological plausibility
(V >= 0, I >= 0, V <= V_max).
"""

from __future__ import annotations

import torch

from cairn.headwall.field_ode import ODEField
from cairn.kit.wiring import DynamicsConfig


def physics_loss(
    field: ODEField, states: torch.Tensor, dynamics: DynamicsConfig
) -> torch.Tensor:
    if not field.uses_physics:
        return states.new_zeros(())
    k = dynamics.physics_dims
    flat = states.reshape(-1, states.shape[-1])
    full = field.forward(states.new_zeros(()), flat)
    physics = field.physics_derivative(flat)
    matching = ((full[..., :k] - physics[..., :k]) ** 2).sum(dim=-1).mean()
    volume = flat[..., 0]
    immune = flat[..., 1]
    bound = (
        torch.relu(-volume)
        + torch.relu(-immune)
        + torch.relu(volume - dynamics.volume_max)
        + torch.relu(immune - dynamics.immune_max)
    )
    return matching + dynamics.bound_weight * bound.mean()
