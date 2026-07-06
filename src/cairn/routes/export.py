"""ONNX export of the deterministic risk pathway.

The exported graph takes a fused latent state and returns the current risk and
the risk after one fixed-step RK4 advance of the latent field (the adaptive ODE
solver is replaced by an explicit step so the graph contains only tensor ops).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Tuple

import plac
import torch
from torch import nn

from cairn.ascent.phases import build_models, resolve_device
from cairn.expedition.worldmodel import WorldModel
from cairn.kit.checkpoints import load_checkpoint
from cairn.kit.logbook import get_logger
from cairn.kit.wiring import load_route

LOGGER = get_logger(__name__)


class RiskPathway(nn.Module):
    def __init__(self, world: WorldModel, step: float) -> None:
        super().__init__()
        self.world = world
        self.step = step

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        field = self.world.field
        zero = latent.new_zeros(())
        h = self.step
        k1 = field.forward(zero, latent)
        k2 = field.forward(zero, latent + 0.5 * h * k1)
        k3 = field.forward(zero, latent + 0.5 * h * k2)
        k4 = field.forward(zero, latent + h * k3)
        advanced = latent + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.world.risk(latent), self.world.risk(advanced)


def run(
    route: str = "beta/standard.yaml",
    checkpoint: str = "runs/standard/final.ckpt",
    output: str = "runs/standard/risk_pathway.onnx",
    device: str = "cpu",
) -> None:
    config = load_route(route)
    dev = resolve_device(device)
    world, _ = build_models(config, dev)
    blob = load_checkpoint(Path(checkpoint), map_location=str(dev))
    world.load_state_dict(blob["world"])
    world.eval()
    pathway = RiskPathway(world, config.rl.step_days / 365.0).to(dev)
    dummy = torch.zeros(1, config.model.latent_dim, device=dev)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {
        "input_names": ["latent"],
        "output_names": ["risk_now", "risk_next"],
        "opset_version": 14,
        "dynamic_axes": {"latent": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(pathway, (dummy,), output, **kwargs)
    LOGGER.info("wrote %s", output)


def main() -> None:
    plac.call(run)


if __name__ == "__main__":
    main()
