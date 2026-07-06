"""Training entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import plac

from cairn.ascent.phases import fit, resolve_device
from cairn.kit.logbook import get_logger
from cairn.kit.wiring import load_route

LOGGER = get_logger(__name__)


def run(
    route: str = "beta/standard.yaml",
    device: str = "auto",
    workdir: str = "runs/standard",
    world_steps: int = -1,
    policy_steps: int = -1,
) -> None:
    config = load_route(route)
    dev = resolve_device(device)
    LOGGER.info(
        "route %s effective batch %d on %s", config.name, config.effective_batch, dev
    )
    caps: Optional[int] = None if world_steps < 0 else world_steps
    pcaps: Optional[int] = None if policy_steps < 0 else policy_steps
    summary = fit(config, dev, Path(workdir), caps, pcaps)
    LOGGER.info("best cindex %.4f", cast(float, summary["best_cindex"]))


def main() -> None:
    plac.call(run)


if __name__ == "__main__":
    main()
