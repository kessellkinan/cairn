"""Per-patient risk-trajectory forecasting entry point."""

from __future__ import annotations

from pathlib import Path

import plac
import torch

from cairn.ascent.phases import build_models, resolve_device
from cairn.field.cohorts import synthesize
from cairn.field.collate import collate
from cairn.kit.checkpoints import load_checkpoint
from cairn.kit.logbook import get_logger
from cairn.kit.wiring import load_route

LOGGER = get_logger(__name__)


def run(
    route: str = "beta/standard.yaml",
    checkpoint: str = "runs/standard/final.ckpt",
    device: str = "auto",
    patient: int = 0,
) -> None:
    config = load_route(route)
    dev = resolve_device(device)
    world, _ = build_models(config, dev)
    blob = load_checkpoint(Path(checkpoint), map_location=str(dev))
    world.load_state_dict(blob["world"])
    world.eval()
    patients = synthesize(config.data, config.seed + 11)
    index = patient % len(patients)
    batch = collate([patients[index]]).to(dev)
    with torch.no_grad():
        rep = world.represent(batch)
        curve = world.trajectory_risk(rep.latent, config.rl.rollout_steps)
    days = [int(round(k * config.rl.step_days)) for k in range(curve.shape[0])]
    for day, value in zip(days, curve.squeeze(-1).cpu().tolist()):
        LOGGER.info("day %4d risk %.4f", day, value)


def main() -> None:
    plac.call(run)


if __name__ == "__main__":
    main()
