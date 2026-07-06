from __future__ import annotations

import numpy as np

from cairn.altimeter.concordance import harrell_cindex, time_dependent_concordance
from cairn.altimeter.discrimination import brier_score, roc_auc, sensitivity_specificity
from cairn.altimeter.inference_stats import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci,
    cohen_d,
    delong_test,
)
from cairn.altimeter.offpolicy import (
    clipped_per_decision_wis,
    weighted_importance_sampling,
)


def test_cindex_perfect_and_reversed() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1.0, 1.0, 1.0, 1.0])
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    assert harrell_cindex(scores, times, events) == 1.0
    assert harrell_cindex(-scores, times, events) == 0.0


def test_time_dependent_respects_landmark() -> None:
    risk = np.array([0.9, 0.8, 0.1])
    times = np.array([100.0, 900.0, 1000.0])
    events = np.array([1.0, 1.0, 0.0])
    value = time_dependent_concordance(risk, times, events, landmark=500.0)
    assert 0.0 <= value <= 1.0


def test_auc_matches_rank_definition() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert abs(roc_auc(scores, labels) - 0.75) < 1e-9


def test_sensitivity_specificity_and_brier() -> None:
    scores = np.array([0.9, 0.2, 0.7, 0.1])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    sens, spec = sensitivity_specificity(scores, labels, 0.5)
    assert sens == 1.0 and spec == 1.0
    assert abs(brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0]))) < 1e-12


def test_delong_detects_difference() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([1.0] * 30 + [0.0] * 30)
    strong = labels + rng.normal(0, 0.3, size=60)
    weak = rng.normal(0, 1.0, size=60)
    _, p = delong_test(strong, weak, labels)
    assert p < 0.05


def test_bootstrap_cohen_and_multiplicity() -> None:
    a = np.array([1.0, 1.1, 0.9, 1.2, 1.0])
    b = np.array([0.1, 0.2, 0.0, 0.1, 0.05])
    lo, hi = bootstrap_ci(lambda x: float(x.mean()), (a,), iterations=100)
    assert lo <= a.mean() <= hi
    assert cohen_d(a, b) > 1.0
    adjusted = benjamini_hochberg(np.array([0.01, 0.02, 0.03]))
    assert np.all(np.diff(adjusted) >= -1e-9)
    assert np.all(bonferroni(np.array([0.01, 0.5]), 17) <= 1.0)


def test_offpolicy_estimators_finite() -> None:
    behaviour = np.full((5, 3), 0.2)
    target = np.full((5, 3), 0.4)
    rewards = np.ones((5, 3))
    mask = np.ones((5, 3))
    assert np.isfinite(weighted_importance_sampling(behaviour, target, rewards, mask))
    assert np.isfinite(clipped_per_decision_wis(behaviour, target, rewards, mask))
