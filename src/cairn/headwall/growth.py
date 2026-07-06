"""Gompertz-logistic tumour growth with immune coupling.

Ref: Methods Sec. 4.5, Eq. (3)-(4). State coordinates are (V, I): tumour
volume and immune mass. Parameters are kept positive through softplus.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cairn.kit.wiring import GrowthConfig

_EPS = 1e-4


def _inverse_softplus(value: float) -> float:
    clipped = max(value, _EPS)
    return float(torch.log(torch.expm1(torch.tensor(clipped))).item())


class GrowthODE(nn.Module):
    def __init__(self, config: GrowthConfig, immune_coupling: bool) -> None:
        super().__init__()
        self._immune = immune_coupling
        names = [
            "alpha",
            "carrying_capacity",
            "kill_rate",
            "saturation",
            "basal_influx",
            "stimulation",
            "half_max",
            "immune_death",
            "rho_immune",
            "eta_immune",
        ]
        for name in names:
            raw = _inverse_softplus(float(getattr(config, name)))
            self.register_parameter(name, nn.Parameter(torch.tensor(raw)))

    def _p(self, name: str) -> torch.Tensor:
        return F.softplus(getattr(self, name)) + _EPS

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        volume = state[..., 0].clamp_min(_EPS)
        immune = state[..., 1].clamp_min(0.0)
        capacity = self._p("carrying_capacity")
        gompertz = self._p("alpha") * volume * torch.log(capacity / volume)
        if self._immune:
            kill = (
                self._p("kill_rate")
                * immune
                * volume
                / (1.0 + self._p("saturation") * volume)
            )
            dv = gompertz - kill
            stim = (
                self._p("stimulation") * volume**2 / (self._p("half_max") ** 2 + volume**2)
            )
            exhaust = (
                self._p("rho_immune")
                * immune
                * volume
                / (1.0 + self._p("eta_immune") * volume)
            )
            di = self._p("basal_influx") + stim - self._p("immune_death") * immune - exhaust
        else:
            dv = gompertz
            di = torch.zeros_like(immune)
        return torch.stack([dv, di], dim=-1)
