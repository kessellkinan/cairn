"""Imagination rollout and actor-critic objective for the monitoring policy.

Ref: Methods Sec. 4.7-4.8. The policy is optimized only on trajectories
imagined by the world model (Eq. 7); the advantage follows
A = R + gamma * V(z_{k+1}) - V(z_k).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.distributions import Categorical

from cairn.expedition.policy import MonitoringPolicy, step_reward
from cairn.expedition.worldmodel import WorldModel
from cairn.kit.wiring import ExperimentConfig


class PolicyLoss(NamedTuple):
    actor: torch.Tensor
    critic: torch.Tensor
    mean_reward: torch.Tensor


def imagine_policy_losses(
    world: WorldModel,
    policy: MonitoringPolicy,
    latent: torch.Tensor,
    config: ExperimentConfig,
) -> PolicyLoss:
    rl = config.rl
    if not config.ablation.drl:
        zero = latent.new_zeros(())
        return PolicyLoss(zero, zero, zero)
    steps = rl.rollout_steps if config.ablation.imagination else 1
    states = world.transition(latent, steps).states.detach()
    actor_loss = latent.new_zeros(())
    critic_loss = latent.new_zeros(())
    reward_total = latent.new_zeros(())
    discount = 1.0
    for k in range(steps):
        heads = policy(states[k])
        dist = Categorical(logits=heads.logits)
        action = dist.sample()
        risk = world.risk(states[k]).detach()
        reward = step_reward(risk, action, rl)
        next_value = policy.critic(states[k + 1]).squeeze(-1).detach()
        target = reward + rl.gamma * next_value
        advantage = target - heads.value
        actor_loss = (
            actor_loss - discount * (dist.log_prob(action) * advantage.detach()).mean()
        )
        critic_loss = critic_loss + advantage.pow(2).mean()
        reward_total = reward_total + reward.mean()
        discount *= rl.gamma
    return PolicyLoss(actor_loss, critic_loss / steps, reward_total / steps)
