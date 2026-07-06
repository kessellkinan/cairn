"""Latent world model: representation, transition, observation, reward, risk.

Ref: Methods Sec. 4.6. The representation aggregates longitudinal observations
with a learned exponential time-decay (adaptive feature integration); the
transition is the physics-constrained Neural-ODE; the risk read-out maps the
latent to a cumulative transformation probability.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from cairn.cordage.fusion import EncoderBank
from cairn.field.collate import Batch
from cairn.field.modalities import ORDER
from cairn.headwall.field_ode import ODEField, Rollout
from cairn.kit.wiring import ExperimentConfig


class Representation(NamedTuple):
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


class WorldModel(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        model = config.model
        self._names = [m.name for m in ORDER]
        self._step_days = config.rl.step_days
        self._latent_dim = model.latent_dim
        self.encoder = EncoderBank(config.data, model, config.ablation)
        self.field = ODEField(
            model.latent_dim,
            model.dynamics,
            model.growth,
            use_physics=config.ablation.physics,
            immune_coupling=config.ablation.immune_coupling,
        )
        self.log_decay = nn.Parameter(torch.tensor(model.decay_init))
        self.risk_head = nn.Sequential(
            nn.Linear(model.latent_dim, model.reward_hidden),
            nn.SiLU(),
            nn.Linear(model.reward_hidden, 1),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(model.latent_dim, model.reward_hidden),
            nn.SiLU(),
            nn.Linear(model.reward_hidden, 1),
        )

    def represent(self, batch: Batch) -> Representation:
        steps = int(batch.obs_mask.shape[1])
        decay = F.softplus(self.log_decay)
        times = torch.cumsum(batch.obs_dt, dim=1)
        last = (times * batch.obs_mask.to(times.dtype)).amax(dim=1, keepdim=True)
        age = (last - times).clamp_min(0.0) / 365.0
        weights = torch.exp(-decay * age) * batch.obs_mask.to(times.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mean = torch.zeros(batch.size, self._latent_dim, device=times.device)
        logvar = torch.zeros_like(mean)
        recon = torch.zeros(batch.size, device=times.device)
        kl = torch.zeros(batch.size, device=times.device)
        for t in range(steps):
            slice_features = {m: batch.features[m][:, t, :] for m in self._names}
            out = self.encoder(slice_features, batch.present)
            w = weights[:, t : t + 1]
            mean = mean + w * out.mean
            logvar = logvar + w * out.logvar
            step_mask = batch.obs_mask[:, t].to(times.dtype)
            recon = recon + out.reconstruction_loss * step_mask
            kl = kl + out.kl * step_mask
        denom = batch.obs_mask.to(times.dtype).sum(dim=1).clamp_min(1.0)
        noise = torch.randn_like(mean) if self.training else torch.zeros_like(mean)
        latent = mean + noise * torch.exp(0.5 * logvar)
        return Representation(latent, mean, logvar, recon / denom, kl / denom)

    def risk(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.risk_head(latent)).squeeze(-1)

    def reward_signal(self, latent: torch.Tensor) -> torch.Tensor:
        return self.reward_head(latent).squeeze(-1)

    def transition(self, latent: torch.Tensor, n_steps: int) -> Rollout:
        return self.field.rollout(latent, n_steps, self._step_days)

    def trajectory_risk(self, latent: torch.Tensor, n_steps: int) -> torch.Tensor:
        states = self.transition(latent, n_steps).states
        return self.risk(states)
