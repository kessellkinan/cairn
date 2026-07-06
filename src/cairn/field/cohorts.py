"""Planted-signal cohort generator for build and test exercises.

A latent intrinsic growth rate drives both the observed multi-omics signal and
the time-to-transformation, so concordance and overfit checks have signal to
recover. No patient records are bundled; real cohorts are listed in
terrain-sources.txt and are accessed under their data-use agreements.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple

import numpy as np
import torch

from cairn.field.modalities import ORDER, Modality
from cairn.kit.wiring import DataConfig


class Patient(NamedTuple):
    features: Dict[str, torch.Tensor]
    present: torch.Tensor
    obs_dt: torch.Tensor
    obs_mask: torch.Tensor
    event_time: torch.Tensor
    event: torch.Tensor
    stage: torch.Tensor
    growth: torch.Tensor


def _modality_signal(
    rng: np.random.Generator, dim: int, growth: float, n_obs: int, times: np.ndarray
) -> np.ndarray:
    loading = rng.standard_normal(dim) * 0.3
    base = rng.standard_normal((n_obs, dim)) * 0.1
    drift = np.outer(growth * times / 365.0, loading)
    return (base + drift).astype(np.float32)


def synthesize(config: DataConfig, seed: int) -> List[Patient]:
    rng = np.random.default_rng(seed)
    dims = {m.name: spec.keep for m, spec in zip(ORDER, config.modalities)}
    active = set(config.active_modalities)
    available = torch.tensor([m.name in active for m in ORDER], dtype=torch.bool)
    step_max = max(config.horizon_days / max(config.max_observations, 1), 90.0)
    patients: List[Patient] = []
    for _ in range(config.cohort_size):
        growth = float(np.clip(rng.gamma(2.0, 0.25), 0.05, 2.0))
        n_obs = int(rng.integers(1, config.max_observations + 1))
        gaps = rng.uniform(60.0, step_max, size=n_obs)
        times = np.cumsum(gaps) - gaps[0]
        features: Dict[str, torch.Tensor] = {}
        for modality in ORDER:
            signal = _modality_signal(rng, dims[modality.name], growth, n_obs, times)
            features[modality.name] = torch.as_tensor(signal)
        present = available.clone()
        if present.sum() > 1 and rng.random() < config.censor_fraction:
            candidates = [i for i in range(len(ORDER)) if bool(present[i])]
            present[int(rng.choice(candidates))] = False
        latent_time = config.event_horizon_days / (growth + 0.1)
        noisy = latent_time * float(rng.lognormal(0.0, 0.2))
        censored = noisy > config.horizon_days or rng.random() < config.censor_fraction
        event_time = float(min(noisy, config.horizon_days))
        event = 0 if censored else 1
        stage = int(growth > 0.6)
        patients.append(
            Patient(
                features=features,
                present=present,
                obs_dt=torch.as_tensor(
                    np.diff(times, prepend=times[:1]), dtype=torch.float32
                ),
                obs_mask=torch.ones(n_obs, dtype=torch.bool),
                event_time=torch.tensor(event_time, dtype=torch.float32),
                event=torch.tensor(event, dtype=torch.float32),
                stage=torch.tensor(stage, dtype=torch.long),
                growth=torch.tensor(growth, dtype=torch.float32),
            )
        )
    return patients


def modality_input_dims(config: DataConfig) -> Dict[Modality, int]:
    return {m: spec.keep for m, spec in zip(ORDER, config.modalities)}
