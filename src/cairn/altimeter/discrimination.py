"""Discrimination and calibration metrics: AUC, sensitivity, specificity, Brier.

Ref: Methods Sec. 4.9.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    rank_sum = ranks[labels == 1].sum()
    auc = (rank_sum - positives.size * (positives.size + 1) / 2.0) / (
        positives.size * negatives.size
    )
    return float(auc)


def sensitivity_specificity(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    predicted = scores >= threshold
    true_pos = float(np.sum(predicted & (labels == 1)))
    true_neg = float(np.sum(~predicted & (labels == 0)))
    pos = float(np.sum(labels == 1))
    neg = float(np.sum(labels == 0))
    sens = true_pos / pos if pos > 0 else 0.0
    spec = true_neg / neg if neg > 0 else 0.0
    return sens, spec


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probabilities - labels) ** 2))
