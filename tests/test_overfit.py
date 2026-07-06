from __future__ import annotations

import torch

from cairn.expedition.worldmodel import WorldModel
from cairn.field.collate import Batch
from cairn.kit.seeding import set_seed
from cairn.kit.wiring import ExperimentConfig
from cairn.provisions.objective import world_objective


def test_world_model_overfits_single_batch(
    smoke: ExperimentConfig, smoke_batch: Batch
) -> None:
    set_seed(0)
    world = WorldModel(smoke)
    optimizer = torch.optim.Adam(world.parameters(), lr=5e-3)
    first = float(world_objective(world, smoke_batch, smoke).total.detach())
    for _ in range(60):
        optimizer.zero_grad()
        loss = world_objective(world, smoke_batch, smoke).total
        loss.backward()
        optimizer.step()
    last = float(world_objective(world, smoke_batch, smoke).total.detach())
    assert last < first * 0.9
