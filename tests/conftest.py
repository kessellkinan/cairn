from __future__ import annotations

import pytest
import torch

from cairn.field.cohorts import synthesize
from cairn.field.collate import collate
from cairn.kit.wiring import ExperimentConfig, load_route


@pytest.fixture()
def smoke() -> ExperimentConfig:
    return load_route("beta/_smoke.yaml")


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def smoke_batch(smoke: ExperimentConfig):
    patients = synthesize(smoke.data, smoke.seed)
    return collate(patients[:8])
