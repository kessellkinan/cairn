"""Two-phase training: world model first, then the imagination-trained policy.

Ref: Methods Sec. 4.8. Phase one fits the encoder, dynamics and decoders;
phase two trains the policy on imagined trajectories with a frozen world model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from cairn.altimeter.concordance import harrell_cindex
from cairn.ascent.engine import policy_epoch, world_epoch
from cairn.ascent.optim import build_optimizer, build_scheduler
from cairn.expedition.policy import MonitoringPolicy
from cairn.expedition.worldmodel import WorldModel
from cairn.field.cohorts import synthesize
from cairn.field.collate import CohortDataset, make_loader
from cairn.kit.checkpoints import save_checkpoint
from cairn.kit.logbook import get_logger
from cairn.kit.seeding import rng_snapshot, set_seed
from cairn.kit.wiring import ExperimentConfig

LOGGER = get_logger(__name__)


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_models(
    config: ExperimentConfig, device: torch.device
) -> Tuple[WorldModel, MonitoringPolicy]:
    world = WorldModel(config).to(device)
    policy = MonitoringPolicy(config.model.latent_dim, config.rl).to(device)
    return world, policy


def _validation_cindex(
    world: WorldModel, loader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    world.eval()
    scores: List[float] = []
    times: List[float] = []
    events: List[float] = []
    with torch.no_grad():
        for batch in loader:
            moved = batch.to(device)
            rep = world.represent(moved)
            risk = world.risk(rep.latent)
            scores.extend(risk.cpu().tolist())
            times.extend(moved.event_time.cpu().tolist())
            events.extend(moved.event.cpu().tolist())
    return harrell_cindex(np.array(scores), np.array(times), np.array(events))


def fit(
    config: ExperimentConfig,
    device: torch.device,
    workdir: Path,
    max_world_steps: Optional[int] = None,
    max_policy_steps: Optional[int] = None,
) -> Dict[str, object]:
    set_seed(config.seed)
    patients = synthesize(config.data, config.seed)
    split = max(int(0.85 * len(patients)), 1)
    train = CohortDataset(patients[:split])
    val = CohortDataset(patients[split:] or patients[:1])
    train_loader = make_loader(train, config.schedule.batch_size, True, config.seed)
    val_loader = make_loader(val, config.schedule.batch_size, False, config.seed + 1)

    world, policy = build_models(config, device)
    optimizer = build_optimizer(world.parameters(), config.optim)
    scheduler = build_scheduler(optimizer, config.optim)

    history: List[float] = []
    best = -1.0
    stale = 0
    for epoch in range(config.schedule.world_epochs):
        losses = world_epoch(
            world, train_loader, optimizer, config, device, max_world_steps
        )
        scheduler.step()
        history.extend(losses)
        metric = _validation_cindex(world, val_loader, device)
        LOGGER.info("world epoch %d loss %.4f cindex %.4f", epoch, np.mean(losses), metric)
        if metric > best:
            best = metric
            stale = 0
            _persist(world, policy, config, workdir / "best.ckpt", epoch)
        else:
            stale += 1
            if stale >= config.schedule.patience:
                LOGGER.info("early stopping at epoch %d", epoch)
                break

    policy_opt = build_optimizer(policy.parameters(), config.optim)
    policy_hist: List[float] = []
    for epoch in range(config.schedule.policy_epochs):
        losses = policy_epoch(
            world, policy, train_loader, policy_opt, config, device, max_policy_steps
        )
        policy_hist.extend(losses)
        LOGGER.info("policy epoch %d loss %.4f", epoch, float(np.mean(losses or [0.0])))

    _persist(world, policy, config, workdir / "final.ckpt", config.schedule.world_epochs)
    return {
        "world_losses": history,
        "policy_losses": policy_hist,
        "best_cindex": best,
    }


def _persist(
    world: WorldModel,
    policy: MonitoringPolicy,
    config: ExperimentConfig,
    path: Path,
    epoch: int,
) -> None:
    save_checkpoint(
        {
            "world": world.state_dict(),
            "policy": policy.state_dict(),
            "seed": config.seed,
            "rng": rng_snapshot(),
            "route": config.name,
            "epoch": epoch,
        },
        path,
    )
