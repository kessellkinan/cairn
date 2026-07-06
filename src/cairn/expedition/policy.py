"""Monitoring policy network, reward, and hard safety constraints.

Ref: Methods Sec. 4.7, Eq. (8) reward and the safety overrides (365-day
monitoring floor; 30-day override above the critical risk threshold).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from cairn.kit.wiring import RLConfig


class PolicyHeads(NamedTuple):
    logits: torch.Tensor
    value: torch.Tensor


class MonitoringPolicy(nn.Module):
    def __init__(self, latent_dim: int, config: RLConfig) -> None:
        super().__init__()
        h1, h2 = config.policy_hidden
        self.actions = config.actions
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, len(config.actions)),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )
        self._threshold = config.risk_threshold
        self._override_idx = config.actions.index(config.override_days)

    def forward(self, latent: torch.Tensor) -> PolicyHeads:
        return PolicyHeads(self.actor(latent), self.critic(latent).squeeze(-1))

    def act(self, latent: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        logits = self.actor(latent)
        choice = torch.argmax(logits, dim=-1)
        override = torch.full_like(choice, self._override_idx)
        return torch.where(risk > self._threshold, override, choice)


def step_reward(
    risk: torch.Tensor, action_idx: torch.Tensor, config: RLConfig
) -> torch.Tensor:
    days = torch.tensor(config.actions, dtype=torch.float32, device=risk.device)
    day = days[action_idx]
    span = float(max(config.actions))
    floor = float(min(config.actions))
    detection = risk * torch.exp(-day / span)
    cost = floor / day
    missed = risk * (day / span)
    return (
        config.lambda_detection * detection
        - config.lambda_cost * cost
        - config.lambda_missed * missed
    )
