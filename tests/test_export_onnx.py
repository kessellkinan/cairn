from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from cairn.ascent.phases import build_models
from cairn.kit.wiring import ExperimentConfig
from cairn.routes.export import RiskPathway

_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


@pytest.mark.skipif(not _HAS_ORT, reason="onnxruntime not installed")
def test_onnx_parity(tmp_path: Path, smoke: ExperimentConfig) -> None:
    import onnxruntime as ort

    device = torch.device("cpu")
    world, _ = build_models(smoke, device)
    world.eval()
    pathway = RiskPathway(world, smoke.rl.step_days / 365.0)
    pathway.eval()
    latent = torch.randn(2, smoke.model.latent_dim)
    with torch.no_grad():
        ref_now, ref_next = pathway(latent)
    path = tmp_path / "pathway.onnx"
    kwargs: dict = {
        "input_names": ["latent"],
        "output_names": ["risk_now", "risk_next"],
        "opset_version": 14,
        "dynamic_axes": {"latent": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(pathway, (latent,), str(path), **kwargs)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"latent": latent.numpy()})
    assert np.allclose(outputs[0], ref_now.numpy(), atol=1e-4)
    assert np.allclose(outputs[1], ref_next.numpy(), atol=1e-4)
