"""Optimizer and learning-rate schedule builders.

Ref: Methods Sec. 4.8. AdamW with cosine annealing warm restarts.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from cairn.kit.wiring import OptimConfig


def build_optimizer(parameters: Iterable[torch.nn.Parameter], config: OptimConfig) -> AdamW:
    return AdamW(
        parameters,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: AdamW, config: OptimConfig) -> CosineAnnealingWarmRestarts:
    return CosineAnnealingWarmRestarts(
        optimizer, T_0=config.restart_period, eta_min=config.restart_min_lr
    )
