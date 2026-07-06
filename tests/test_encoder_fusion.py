from __future__ import annotations

import torch

from cairn.cordage.fusion import EncoderBank
from cairn.field.modalities import ORDER
from cairn.kit.wiring import AblationFlags, ExperimentConfig


def _features(config: ExperimentConfig, n: int) -> dict:
    return {
        m.name: torch.randn(n, spec.keep) for m, spec in zip(ORDER, config.data.modalities)
    }


def test_fusion_output_shape(smoke: ExperimentConfig) -> None:
    bank = EncoderBank(smoke.data, smoke.model, smoke.ablation)
    present = torch.ones(4, len(ORDER), dtype=torch.bool)
    out = bank(_features(smoke, 4), present)
    assert out.latent.shape == (4, smoke.model.latent_dim)
    assert out.reconstruction_loss.shape == (4,)


def test_missing_modality_uses_default(smoke: ExperimentConfig) -> None:
    bank = EncoderBank(smoke.data, smoke.model, smoke.ablation)
    bank.eval()
    present = torch.ones(2, len(ORDER), dtype=torch.bool)
    present[:, 1] = False
    out = bank(_features(smoke, 2), present)
    assert torch.isfinite(out.latent).all()


def test_single_encoder_ablation_runs(smoke: ExperimentConfig) -> None:
    flags = AblationFlags(True, True, False, True, True, True)
    bank = EncoderBank(smoke.data, smoke.model, flags)
    present = torch.ones(3, len(ORDER), dtype=torch.bool)
    out = bank(_features(smoke, 3), present)
    assert out.latent.shape == (3, smoke.model.latent_dim)


def test_attention_ablation_runs(smoke: ExperimentConfig) -> None:
    flags = AblationFlags(True, True, True, True, True, False)
    bank = EncoderBank(smoke.data, smoke.model, flags)
    present = torch.ones(3, len(ORDER), dtype=torch.bool)
    out = bank(_features(smoke, 3), present)
    assert out.latent.shape == (3, smoke.model.latent_dim)
