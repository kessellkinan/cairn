"""Off-policy value estimators for the monitoring policy.

Ref: Methods Sec. 4.7. Weighted importance sampling and clipped weighted
per-decision importance sampling under the model-defined reward.
"""

from __future__ import annotations

import numpy as np


def _trajectory_ratios(
    behaviour: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    ratio = np.where(mask > 0, target / np.clip(behaviour, 1e-8, None), 1.0)
    return np.cumprod(ratio, axis=1)


def weighted_importance_sampling(
    behaviour: np.ndarray, target: np.ndarray, rewards: np.ndarray, mask: np.ndarray
) -> float:
    behaviour, target, rewards, mask = (
        np.asarray(behaviour, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(mask, dtype=np.float64),
    )
    cum = _trajectory_ratios(behaviour, target, mask)
    final_weight = cum[:, -1]
    returns = (rewards * mask).sum(axis=1)
    denom = final_weight.sum()
    if denom <= 0:
        return 0.0
    return float((final_weight * returns).sum() / denom)


def clipped_per_decision_wis(
    behaviour: np.ndarray,
    target: np.ndarray,
    rewards: np.ndarray,
    mask: np.ndarray,
    clip: float = 10.0,
) -> float:
    behaviour, target, rewards, mask = (
        np.asarray(behaviour, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(mask, dtype=np.float64),
    )
    cum = np.clip(_trajectory_ratios(behaviour, target, mask), 0.0, clip)
    norm = cum.sum(axis=0)
    norm = np.where(norm > 0, norm, 1.0)
    weighted = (cum * rewards * mask) / norm
    return float(weighted.sum(axis=0).sum())
