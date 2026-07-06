from __future__ import annotations

import pytest

from cairn.kit.wiring import load_route

ABLATIONS = [
    "beta/ablation_no_physics.yaml",
    "beta/ablation_no_imagination.yaml",
    "beta/ablation_no_fusion.yaml",
    "beta/ablation_no_drl.yaml",
    "beta/ablation_no_immune.yaml",
    "beta/ablation_no_attention.yaml",
    "beta/ablation_equal_budget.yaml",
]


def test_standard_effective_batch() -> None:
    config = load_route("beta/standard.yaml")
    assert config.effective_batch == 256 * 24 * 8
    assert config.optim.lr == pytest.approx(2.0e-7)
    assert config.schedule.precision == "fp32"


def test_ablation_flags_toggle() -> None:
    physics = load_route("beta/ablation_no_physics.yaml")
    assert physics.ablation.physics is False
    assert physics.ablation.imagination is True
    budget = load_route("beta/ablation_equal_budget.yaml")
    assert budget.ablation.physics is False and budget.ablation.imagination is False


@pytest.mark.parametrize("route", ABLATIONS)
def test_all_ablations_parse(route: str) -> None:
    config = load_route(route)
    assert config.model.latent_dim % len(config.data.modalities) == 0


def test_single_modality_supplementary() -> None:
    config = load_route("beta/supplementary_transcriptomics_only.yaml")
    assert config.data.active_modalities == ["transcriptomics"]
