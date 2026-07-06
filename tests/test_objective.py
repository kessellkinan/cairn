from __future__ import annotations

import torch

from cairn.expedition.worldmodel import WorldModel
from cairn.field.collate import Batch
from cairn.kit.wiring import ExperimentConfig
from cairn.provisions.objective import cox_partial_likelihood, world_objective


def test_world_objective_components_finite(
    smoke: ExperimentConfig, smoke_batch: Batch
) -> None:
    world = WorldModel(smoke)
    losses = world_objective(world, smoke_batch, smoke)
    for value in losses:
        assert torch.isfinite(value)
    assert losses.physics >= 0.0


def test_cox_orders_risk() -> None:
    times = torch.tensor([1.0, 2.0, 3.0])
    events = torch.tensor([1.0, 1.0, 0.0])
    aligned = cox_partial_likelihood(torch.tensor([2.0, 1.0, 0.0]), times, events)
    reversed_ = cox_partial_likelihood(torch.tensor([0.0, 1.0, 2.0]), times, events)
    assert aligned < reversed_


def test_cox_zero_without_events() -> None:
    out = cox_partial_likelihood(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0])
    )
    assert out.item() == 0.0
