"""Statistical comparison tools.

Ref: Methods Sec. 4.9. DeLong AUC comparison, paired bootstrap intervals,
Cohen's d, Bonferroni and Benjamini-Hochberg multiplicity control.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranked = x[order]
    n = x.size
    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=np.float64)
    result[order] = out
    return result


def delong_test(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    labels = np.asarray(labels)
    pos_a, neg_a = scores_a[labels == 1], scores_a[labels == 0]
    pos_b, neg_b = scores_b[labels == 1], scores_b[labels == 0]
    m, n = pos_a.size, neg_a.size
    if m == 0 or n == 0:
        return 0.0, 1.0

    def structural(
        pos: np.ndarray, neg: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        tx = _midrank(pos)
        ty = _midrank(neg)
        txy = _midrank(np.concatenate([pos, neg]))
        auc = (txy[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v01 = (txy[:m] - tx) / n
        v10 = 1.0 - (txy[m:] - ty) / m
        return auc, v01, v10

    auc_a, v01a, v10a = structural(pos_a, neg_a)
    auc_b, v01b, v10b = structural(pos_b, neg_b)
    s01 = np.cov(np.stack([v01a, v01b]))
    s10 = np.cov(np.stack([v10a, v10b]))
    cov = s01 / m + s10 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 0.0, 1.0
    z = (auc_a - auc_b) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p)


def bootstrap_ci(
    metric: Callable[..., float],
    arrays: Tuple[np.ndarray, ...],
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = arrays[0].shape[0]
    estimates: List[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        estimates.append(metric(*(a[idx] for a in arrays)))
    lo = float(np.quantile(estimates, alpha / 2.0))
    hi = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return lo, hi


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def bonferroni(pvalues: np.ndarray, comparisons: int) -> np.ndarray:
    return np.clip(np.asarray(pvalues, dtype=np.float64) * comparisons, 0.0, 1.0)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out
