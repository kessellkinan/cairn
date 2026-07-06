"""Composite training objective.

Ref: Methods Sec. 4.8, Eq. (9): L = L_recon + l1 L_physics + l2 L_RL + l3 L_KL.
The risk read-out is supervised with a Cox partial likelihood on the
ODE-derived risk score (see docs/deviations for the rationale).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from cairn.expedition.imagination import imagine_policy_losses
from cairn.expedition.policy import MonitoringPolicy
from cairn.expedition.worldmodel import WorldModel
from cairn.field.collate import Batch
from cairn.headwall.constraint import physics_loss
from cairn.kit.wiring import ExperimentConfig


class WorldLosses(NamedTuple):
    total: torch.Tensor
    reconstruction: torch.Tensor
    physics: torch.Tensor
    kl: torch.Tensor
    risk: torch.Tensor


def cox_partial_likelihood(
    scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    if events.sum() == 0:
        return scores.new_zeros(())
    order = torch.argsort(times, descending=True)
    ordered_scores = scores[order]
    ordered_events = events[order]
    log_cumsum = torch.logcumsumexp(ordered_scores, dim=0)
    contributions = (ordered_scores - log_cumsum) * ordered_events
    return -contributions.sum() / events.sum().clamp_min(1.0)


def world_objective(
    world: WorldModel, batch: Batch, config: ExperimentConfig
) -> WorldLosses:
    rep = world.represent(batch)
    steps = config.rl.rollout_steps
    states = world.transition(rep.latent, steps).states
    physics = physics_loss(world.field, states, config.model.dynamics)
    recon = rep.reconstruction.mean()
    kl = rep.kl.mean()
    score = world.risk_head(rep.latent).squeeze(-1)
    risk = cox_partial_likelihood(score, batch.event_time, batch.event)
    weights = config.loss_weights
    total = recon + weights.physics * physics + weights.kl * kl + risk
    return WorldLosses(total, recon, physics, kl, risk)


def policy_objective(
    world: WorldModel, policy: MonitoringPolicy, batch: Batch, config: ExperimentConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    rep = world.represent(batch)
    losses = imagine_policy_losses(world, policy, rep.latent.detach(), config)
    total = config.loss_weights.rl * (losses.actor + losses.critic)
    return total, losses.mean_reward
