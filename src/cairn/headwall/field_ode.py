"""Physics-constrained Neural-ODE latent field and imagination rollout.

Ref: Methods Sec. 4.5-4.6, Eq. (5) hybrid field and Eq. (7) rollout integral.
The physics term acts on the first k latent coordinates; the residual network
acts on the whole latent vector.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torchdiffeq import odeint

from cairn.headwall.growth import GrowthODE
from cairn.kit.wiring import DynamicsConfig, GrowthConfig


class Rollout(NamedTuple):
    states: torch.Tensor
    times: torch.Tensor


class ODEField(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        dynamics: DynamicsConfig,
        growth: GrowthConfig,
        use_physics: bool,
        immune_coupling: bool,
    ) -> None:
        super().__init__()
        self._k = dynamics.physics_dims
        self._use_physics = use_physics
        self._rtol = dynamics.rtol
        self._atol = dynamics.atol
        self._solver = dynamics.solver
        self.growth = GrowthODE(growth, immune_coupling)
        layers: list[nn.Module] = []
        width = latent_dim
        for _ in range(dynamics.residual_layers):
            layers += [nn.Linear(width, dynamics.residual_hidden), nn.SiLU()]
            width = dynamics.residual_hidden
        layers.append(nn.Linear(width, latent_dim))
        self.residual = nn.Sequential(*layers)

    @property
    def uses_physics(self) -> bool:
        return self._use_physics

    def physics_derivative(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(z)
        if self._use_physics:
            out[..., : self._k] = self.growth(z[..., : self._k])
        return out

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.physics_derivative(z) + self.residual(z)

    def integrate(self, z0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return odeint(
            self.forward, z0, times, rtol=self._rtol, atol=self._atol, method=self._solver
        )

    def rollout(self, z0: torch.Tensor, n_steps: int, step_days: float) -> Rollout:
        scale = step_days / 365.0
        times = torch.arange(0, n_steps + 1, device=z0.device, dtype=z0.dtype) * scale
        states = self.integrate(z0, times)
        return Rollout(states=states, times=times)
