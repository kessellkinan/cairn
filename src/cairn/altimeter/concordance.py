"""Concordance metrics.

Ref: Methods Sec. 4.9. Harrell concordance and the time-dependent concordance
C_td(t) = P(R_i(t) > R_j(t) | T_i < T_j, T_i <= t).
"""

from __future__ import annotations

import numpy as np


def _as_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def harrell_cindex(scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    scores, times, events = map(_as_array, (scores, times, events))
    concordant = 0.0
    permissible = 0.0
    n = times.shape[0]
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if times[j] <= times[i]:
                continue
            permissible += 1.0
            if scores[i] > scores[j]:
                concordant += 1.0
            elif scores[i] == scores[j]:
                concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else 0.5


def time_dependent_concordance(
    risk_at_t: np.ndarray, times: np.ndarray, events: np.ndarray, landmark: float
) -> float:
    risk_at_t, times, events = map(_as_array, (risk_at_t, times, events))
    concordant = 0.0
    permissible = 0.0
    n = times.shape[0]
    for i in range(n):
        if events[i] != 1 or times[i] > landmark:
            continue
        for j in range(n):
            if times[j] <= times[i]:
                continue
            permissible += 1.0
            if risk_at_t[i] > risk_at_t[j]:
                concordant += 1.0
            elif risk_at_t[i] == risk_at_t[j]:
                concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else 0.5
