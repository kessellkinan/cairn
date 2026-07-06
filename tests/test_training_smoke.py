from __future__ import annotations

from pathlib import Path

import torch

from cairn.ascent.engine import world_epoch
from cairn.ascent.optim import build_optimizer
from cairn.ascent.phases import build_models, fit
from cairn.field.cohorts import synthesize
from cairn.field.collate import CohortDataset, make_loader
from cairn.kit.seeding import set_seed
from cairn.kit.wiring import ExperimentConfig


def test_two_world_steps_reduce_loss(smoke: ExperimentConfig) -> None:
    set_seed(0)
    device = torch.device("cpu")
    world, _ = build_models(smoke, device)
    patients = synthesize(smoke.data, smoke.seed)
    loader = make_loader(
        CohortDataset(patients), smoke.schedule.batch_size, True, smoke.seed
    )
    optimizer = build_optimizer(world.parameters(), smoke.optim)
    losses = world_epoch(world, loader, optimizer, smoke, device, max_steps=2)
    assert len(losses) == 2
    assert losses[-1] <= losses[0] + 1e-3


def test_full_fit_runs(tmp_path: Path, smoke: ExperimentConfig) -> None:
    device = torch.device("cpu")
    summary = fit(smoke, device, tmp_path, max_world_steps=2, max_policy_steps=2)
    assert (tmp_path / "final.ckpt").exists()
    assert "best_cindex" in summary
