"""Evaluation entry point."""

from __future__ import annotations

from pathlib import Path

import plac

from cairn.ascent.phases import build_models, resolve_device
from cairn.descent.report import evaluate, evaluate_policy
from cairn.field.cohorts import synthesize
from cairn.field.collate import CohortDataset, make_loader
from cairn.kit.checkpoints import load_checkpoint
from cairn.kit.logbook import get_logger
from cairn.kit.wiring import load_route

LOGGER = get_logger(__name__)


def run(
    route: str = "beta/standard.yaml",
    checkpoint: str = "runs/standard/final.ckpt",
    device: str = "auto",
) -> None:
    config = load_route(route)
    dev = resolve_device(device)
    world, policy = build_models(config, dev)
    blob = load_checkpoint(Path(checkpoint), map_location=str(dev))
    world.load_state_dict(blob["world"])
    policy.load_state_dict(blob["policy"])
    patients = synthesize(config.data, config.seed + 7)
    loader = make_loader(
        CohortDataset(patients), config.schedule.batch_size, False, config.seed
    )
    table = evaluate(world, loader, config, dev)
    for name, (value, lo, hi) in table.items():
        LOGGER.info("%s = %.4f [%.4f, %.4f]", name, value, lo, hi)
    policy_metrics = evaluate_policy(world, policy, loader, config, dev)
    for name, value in policy_metrics.items():
        LOGGER.info("%s = %.4f", name, value)


def main() -> None:
    plac.call(run)


if __name__ == "__main__":
    main()
