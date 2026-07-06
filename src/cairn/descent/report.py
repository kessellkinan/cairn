"""Evaluation loops producing the reported discrimination/calibration table.

Ref: Methods Sec. 4.9 and Results Tables 1-6. C-index, time-dependent
concordance at the 2-year landmark, AUC, sensitivity, specificity, Brier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from cairn.altimeter.concordance import harrell_cindex, time_dependent_concordance
from cairn.altimeter.discrimination import brier_score, roc_auc, sensitivity_specificity
from cairn.altimeter.inference_stats import bootstrap_ci
from cairn.altimeter.offpolicy import (
    clipped_per_decision_wis,
    weighted_importance_sampling,
)
from cairn.expedition.policy import MonitoringPolicy, step_reward
from cairn.expedition.worldmodel import WorldModel
from cairn.kit.wiring import ExperimentConfig

_LANDMARK_DAYS = 730.0


@dataclass
class Predictions:
    risk: np.ndarray
    landmark_risk: np.ndarray
    event_time: np.ndarray
    event: np.ndarray
    label: np.ndarray


def collect_predictions(
    world: WorldModel, loader: DataLoader, config: ExperimentConfig, device: torch.device
) -> Predictions:
    world.eval()
    landmark_step = max(int(round(_LANDMARK_DAYS / config.rl.step_days)), 1)
    risk, landmark, times, events = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            moved = batch.to(device)
            rep = world.represent(moved)
            risk.extend(world.risk(rep.latent).cpu().tolist())
            curve = world.trajectory_risk(rep.latent, landmark_step)
            idx = min(landmark_step, curve.shape[0] - 1)
            landmark.extend(curve[idx].cpu().tolist())
            times.extend(moved.event_time.cpu().tolist())
            events.extend(moved.event.cpu().tolist())
    times_arr = np.asarray(times)
    events_arr = np.asarray(events)
    label = ((events_arr == 1) & (times_arr <= _LANDMARK_DAYS)).astype(np.float64)
    return Predictions(
        risk=np.asarray(risk),
        landmark_risk=np.asarray(landmark),
        event_time=times_arr,
        event=events_arr,
        label=label,
    )


def evaluate(
    world: WorldModel, loader: DataLoader, config: ExperimentConfig, device: torch.device
) -> Dict[str, Tuple[float, float, float]]:
    pred = collect_predictions(world, loader, config, device)
    out: Dict[str, Tuple[float, float, float]] = {}

    def ci(
        value: float, fn: Callable[..., float], *arrays: np.ndarray
    ) -> Tuple[float, float, float]:
        lo, hi = bootstrap_ci(fn, arrays, iterations=200, seed=config.seed)
        return value, lo, hi

    out["c_index"] = ci(
        harrell_cindex(pred.risk, pred.event_time, pred.event),
        harrell_cindex,
        pred.risk,
        pred.event_time,
        pred.event,
    )
    out["c_td_2yr"] = (
        time_dependent_concordance(
            pred.landmark_risk, pred.event_time, pred.event, _LANDMARK_DAYS
        ),
        0.0,
        0.0,
    )
    out["auc"] = ci(
        roc_auc(pred.landmark_risk, pred.label), roc_auc, pred.landmark_risk, pred.label
    )
    sens, spec = sensitivity_specificity(pred.landmark_risk, pred.label, 0.5)
    out["sensitivity"] = (sens, 0.0, 0.0)
    out["specificity"] = (spec, 0.0, 0.0)
    out["brier"] = (brier_score(pred.landmark_risk, pred.label), 0.0, 0.0)
    return out


def evaluate_policy(
    world: WorldModel,
    policy: MonitoringPolicy,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    world.eval()
    policy.eval()
    n_actions = len(config.rl.actions)
    behaviour: List[float] = []
    target: List[float] = []
    reward_log: List[float] = []
    intervals: List[int] = []
    with torch.no_grad():
        for batch in loader:
            moved = batch.to(device)
            rep = world.represent(moved)
            probs = torch.softmax(policy(rep.latent).logits, dim=-1)
            risk = world.risk(rep.latent)
            chosen = policy.act(rep.latent, risk)
            reward = step_reward(risk, chosen, config.rl)
            picked = probs.gather(1, chosen.view(-1, 1)).squeeze(1)
            target.extend(picked.cpu().tolist())
            behaviour.extend([1.0 / n_actions] * chosen.shape[0])
            reward_log.extend(reward.cpu().tolist())
            intervals.extend([config.rl.actions[i] for i in chosen.cpu().tolist()])
    beh = np.asarray(behaviour).reshape(-1, 1)
    tgt = np.asarray(target).reshape(-1, 1)
    rew = np.asarray(reward_log).reshape(-1, 1)
    msk = np.ones_like(rew)
    return {
        "wis": weighted_importance_sampling(beh, tgt, rew, msk),
        "cwpdis": clipped_per_decision_wis(beh, tgt, rew, msk),
        "avg_interval_days": float(np.mean(intervals)) if intervals else 0.0,
    }
