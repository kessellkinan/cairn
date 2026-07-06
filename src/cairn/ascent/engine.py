"""Epoch loops for the world model and the monitoring policy.

Gradient accumulation reaches the reported effective batch on the documented
hardware; clipping and optional mixed precision are configured per route.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional

import torch
from torch.utils.data import DataLoader

from cairn.expedition.policy import MonitoringPolicy
from cairn.expedition.worldmodel import WorldModel
from cairn.kit.wiring import ExperimentConfig
from cairn.provisions.objective import policy_objective, world_objective


@contextmanager
def _autocast(device: torch.device, precision: str) -> Iterator[None]:
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def world_epoch(
    world: WorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> List[float]:
    world.train()
    accum = config.schedule.grad_accum
    losses: List[float] = []
    optimizer.zero_grad(set_to_none=True)
    steps = 0
    for index, batch in enumerate(loader):
        moved = batch.to(device)
        with _autocast(device, config.schedule.precision):
            result = world_objective(world, moved, config)
            loss = result.total / accum
        loss.backward()
        if (index + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(world.parameters(), config.optim.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(result.total.detach()))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
    return losses


def policy_epoch(
    world: WorldModel,
    policy: MonitoringPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> List[float]:
    world.eval()
    policy.train()
    losses: List[float] = []
    for steps, batch in enumerate(loader):
        moved = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy_objective(world, policy, moved, config)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.optim.grad_clip)
            optimizer.step()
        losses.append(float(loss.detach()))
        if max_steps is not None and steps + 1 >= max_steps:
            break
    return losses
