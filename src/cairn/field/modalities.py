"""Modality registry and per-modality preprocessing.

Ref: Methods Sec. 4.4 (modality-specific pretreatments and retained dims).
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np
import torch


class Modality(str, Enum):
    TRANSCRIPTOMICS = "transcriptomics"
    METHYLATION = "methylation"
    PROTEOMICS = "proteomics"
    MIRNA = "mirna"


ORDER: Tuple[Modality, ...] = (
    Modality.TRANSCRIPTOMICS,
    Modality.METHYLATION,
    Modality.PROTEOMICS,
    Modality.MIRNA,
)

RETAINED = {
    Modality.TRANSCRIPTOMICS: 2000,
    Modality.METHYLATION: 5000,
    Modality.PROTEOMICS: 1000,
    Modality.MIRNA: 500,
}


def position(modality: Modality) -> int:
    return ORDER.index(modality)


def log2_counts(matrix: np.ndarray) -> np.ndarray:
    return np.log2(np.clip(matrix, 0.0, None) + 1.0)


def top_variance(matrix: np.ndarray, keep: int) -> Tuple[np.ndarray, np.ndarray]:
    variance = matrix.var(axis=0)
    order = np.argsort(variance)[::-1][:keep]
    idx = np.sort(order)
    return matrix[:, idx], idx


def most_differential(
    matrix: np.ndarray, group: np.ndarray, keep: int
) -> Tuple[np.ndarray, np.ndarray]:
    positive = matrix[group == 1]
    negative = matrix[group == 0]
    if positive.shape[0] == 0 or negative.shape[0] == 0:
        return top_variance(matrix, keep)
    delta = np.abs(positive.mean(axis=0) - negative.mean(axis=0))
    order = np.argsort(delta)[::-1][:keep]
    idx = np.sort(order)
    return matrix[:, idx], idx


def quantile_normalize(matrix: np.ndarray) -> np.ndarray:
    ranks = matrix.argsort(axis=0).argsort(axis=0)
    reference = np.sort(matrix, axis=0).mean(axis=1)
    fractional = ranks / max(matrix.shape[0] - 1, 1)
    grid = np.linspace(0.0, 1.0, num=matrix.shape[0])
    out = np.empty_like(matrix, dtype=np.float64)
    for col in range(matrix.shape[1]):
        out[:, col] = np.interp(fractional[:, col], grid, reference)
    return out


def to_tensor(matrix: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(matrix, dtype=torch.float32)
