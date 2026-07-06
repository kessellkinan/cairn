"""Batching with ragged observation lengths and missing-modality masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from cairn.field.cohorts import Patient
from cairn.field.modalities import ORDER


@dataclass
class Batch:
    features: Dict[str, torch.Tensor]
    present: torch.Tensor
    obs_dt: torch.Tensor
    obs_mask: torch.Tensor
    event_time: torch.Tensor
    event: torch.Tensor
    stage: torch.Tensor
    growth: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            features={k: v.to(device) for k, v in self.features.items()},
            present=self.present.to(device),
            obs_dt=self.obs_dt.to(device),
            obs_mask=self.obs_mask.to(device),
            event_time=self.event_time.to(device),
            event=self.event.to(device),
            stage=self.stage.to(device),
            growth=self.growth.to(device),
        )

    @property
    def size(self) -> int:
        return int(self.present.shape[0])


class CohortDataset(Dataset):
    def __init__(self, patients: List[Patient]) -> None:
        self._patients = patients

    def __len__(self) -> int:
        return len(self._patients)

    def __getitem__(self, index: int) -> Patient:
        return self._patients[index]


def collate(samples: List[Patient]) -> Batch:
    max_t = max(int(p.obs_mask.shape[0]) for p in samples)
    features: Dict[str, torch.Tensor] = {}
    for modality in ORDER:
        dim = samples[0].features[modality.name].shape[-1]
        stacked = torch.zeros(len(samples), max_t, dim)
        for i, p in enumerate(samples):
            t = p.features[modality.name].shape[0]
            stacked[i, :t] = p.features[modality.name]
        features[modality.name] = stacked
    present = torch.stack([p.present for p in samples])
    obs_dt = torch.zeros(len(samples), max_t)
    obs_mask = torch.zeros(len(samples), max_t, dtype=torch.bool)
    for i, p in enumerate(samples):
        t = p.obs_mask.shape[0]
        obs_dt[i, :t] = p.obs_dt
        obs_mask[i, :t] = p.obs_mask
    return Batch(
        features=features,
        present=present,
        obs_dt=obs_dt,
        obs_mask=obs_mask,
        event_time=torch.stack([p.event_time for p in samples]),
        event=torch.stack([p.event for p in samples]),
        stage=torch.stack([p.stage for p in samples]),
        growth=torch.stack([p.growth for p in samples]),
    )


def make_loader(
    dataset: CohortDataset, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
        drop_last=False,
    )
