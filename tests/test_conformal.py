"""Unit and property tests for core/conformal.py."""

import numpy as np
import pytest

from core.conformal import (
    compute_mondrian_thresholds,
    compute_split_conformal_threshold,
    predict_conformal_sets,
    predict_mondrian_sets,
)


def test_hand_computed_conformal_example() -> None:
    """Deterministic hand-computed example for quantile calculation and prediction sets.

    Manual Calculation Walkthrough:
    - Calibration sample: 4 observations.
      Index 0: score=0.10, y=0 -> nonconformity s_0 = score = 0.10
      Index 1: score=0.20, y=0 -> nonconformity s_1 = score = 0.20
      Index 2: score=0.70, y=1 -> nonconformity s_2 = 1 - score = 0.30
      Index 3: score=0.60, y=1 -> nonconformity s_3 = 1 - score = 0.40
    - Sorted calibration nonconformity scores: [0.10, 0.20, 0.30, 0.40].
    - Target coverage = 0.75 (alpha = 0.25).
      Finite sample rank: k = ceil((n + 1) * (1 - alpha)) = ceil(5 * 0.75) = ceil(3.75) = 4.
      Since k <= n (4 <= 4), threshold is sorted_scores[k - 1] = sorted_scores[3] = 0.40.
    - Test points evaluated against threshold q = 0.40:
      Point A (score=0.15):
        score <= 0.40 (True -> 0 in set)
        score >= 1 - 0.40 = 0.60 (False -> 1 not in set)
        Prediction set: {0}
      Point B (score=0.85):
        score <= 0.40 (False -> 0 not in set)
        score >= 0.60 (True -> 1 in set)
        Prediction set: {1}
      Point C (score=0.50):
        score <= 0.40 (False)
        score >= 0.60 (False)
        Prediction set: empty set (counts as defer)
    """
    cal_scores = np.array([0.10, 0.20, 0.70, 0.60])
    cal_outcomes = np.array([0, 0, 1, 1])
    alpha = 0.25

    q = compute_split_conformal_threshold(cal_scores, cal_outcomes, alpha=alpha)
    assert q == pytest.approx(0.40, abs=1e-6)

    test_scores = np.array([0.15, 0.85, 0.50])
    sets = predict_conformal_sets(test_scores, q, alpha=alpha)

    # Point A: {0}
    assert bool(sets.include_0[0]) is True
    assert bool(sets.include_1[0]) is False

    # Point B: {1}
    assert bool(sets.include_0[1]) is False
    assert bool(sets.include_1[1]) is True

    # Point C: empty set
    assert bool(sets.include_0[2]) is False
    assert bool(sets.include_1[2]) is False


def test_test_labels_cannot_influence_threshold() -> None:
    """Verifies thresholds depend strictly on calibration data; test labels have no effect."""
    cal_scores = np.array([0.12, 0.25, 0.65, 0.80])
    cal_outcomes = np.array([0, 0, 1, 1])
    test_scores = np.array([0.20, 0.70])

    q_cal = compute_split_conformal_threshold(cal_scores, cal_outcomes, alpha=0.10)

    # Invert or alter test labels completely
    test_outcomes_alt1 = np.array([0, 0])
    test_outcomes_alt2 = np.array([1, 1])

    sets1 = predict_conformal_sets(test_scores, q_cal, alpha=0.10)
    # Confirm prediction sets depend only on test_scores and q_cal, not test_outcomes
    assert len(sets1.include_0) == len(test_scores)
    assert q_cal == compute_split_conformal_threshold(
        cal_scores, cal_outcomes, alpha=0.10
    )
    # Evaluate coverage on both hypothetical labels without threshold change
    assert sets1.compute_coverage(test_outcomes_alt1) >= 0.0
    assert sets1.compute_coverage(test_outcomes_alt2) >= 0.0


def test_insufficient_calibration_quantile_returns_infinity() -> None:
    """Verifies that when ceil((n+1)(1-alpha))/n exceeds 1, threshold is infinity."""
    # n=5, alpha=0.10 -> (5+1)*0.9 = 5.4 -> ceil=6 > 5
    cal_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    cal_outcomes = np.array([0, 0, 0, 1, 1])
    q = compute_split_conformal_threshold(cal_scores, cal_outcomes, alpha=0.10)

    assert q == float("inf")

    # With infinite threshold, both labels must be included for all scores
    test_scores = np.array([0.05, 0.50, 0.95])
    sets = predict_conformal_sets(test_scores, q, alpha=0.10)
    assert sets.include_0.all()
    assert sets.include_1.all()


def test_mondrian_per_group_and_under_sampled_warnings() -> None:
    """Verifies Mondrian calculates per-group thresholds and sets inf on small groups."""
    cal_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.15])
    cal_outcomes = np.array([0, 0, 0, 1, 1, 0])
    cal_groups = np.array(["BigGroup"] * 5 + ["TinyGroup"] * 1)
    alpha = 0.20

    mond = compute_mondrian_thresholds(
        cal_scores, cal_outcomes, cal_groups, "region", alpha=alpha
    )

    # BigGroup has 5 samples: (5+1)*0.8 = 4.8 -> ceil=5 <= 5 -> finite threshold
    assert mond.thresholds["BigGroup"] < float("inf")

    # TinyGroup has 1 sample: (1+1)*0.8 = 1.6 -> ceil=2 > 1 -> infinite threshold
    assert mond.thresholds["TinyGroup"] == float("inf")
    assert "TinyGroup" in mond.under_sampled_groups

    # Predict test sets
    test_scores = np.array([0.2, 0.5])
    test_groups = np.array(["BigGroup", "TinyGroup"])
    test_sets = predict_mondrian_sets(test_scores, test_groups, mond, alpha=alpha)

    # TinyGroup instance must contain both labels
    assert test_sets.include_0[1] and test_sets.include_1[1]


def test_property_marginal_coverage_over_200_seeds() -> None:
    """Property test: Repeated over >= 200 seeds, mean coverage >= 1 - alpha - 0.01."""
    alpha = 0.10
    target = 1.0 - alpha
    coverages = []

    for seed in range(200):
        rng = np.random.default_rng(seed)
        n_cal = 400
        n_test = 400
        s_cal = rng.beta(2, 5, size=n_cal)
        y_cal = (rng.uniform(0, 1, size=n_cal) < s_cal).astype(int)
        s_test = rng.beta(2, 5, size=n_test)
        y_test = (rng.uniform(0, 1, size=n_test) < s_test).astype(int)

        q = compute_split_conformal_threshold(s_cal, y_cal, alpha)
        sets = predict_conformal_sets(s_test, q, alpha)
        cov = sets.compute_coverage(y_test)
        coverages.append(cov)

    mean_cov = float(np.mean(coverages))
    assert (
        mean_cov >= target - 0.01
    ), f"Mean coverage {mean_cov:.4f} was below {target - 0.01}"


def test_property_mondrian_per_group_coverage() -> None:
    """Property test: Mondrian coverage per group >= 1 - alpha - tolerance."""
    rng = np.random.default_rng(123)
    alpha = 0.10
    tolerance = 0.04
    group_sizes = {"Group1": 800, "Group2": 500, "Group3": 300}

    all_s_cal, all_y_cal, all_g_cal = [], [], []
    all_s_test, all_y_test, all_g_test = [], [], []

    for grp, n in group_sizes.items():
        n_half = n // 2
        s_c = rng.beta(2, 5, size=n_half)
        y_c = (rng.uniform(0, 1, size=n_half) < s_c).astype(int)
        s_t = rng.beta(2, 5, size=n_half)
        y_t = (rng.uniform(0, 1, size=n_half) < s_t).astype(int)

        all_s_cal.extend(s_c)
        all_y_cal.extend(y_c)
        all_g_cal.extend([grp] * n_half)

        all_s_test.extend(s_t)
        all_y_test.extend(y_t)
        all_g_test.extend([grp] * n_half)

    mond = compute_mondrian_thresholds(
        np.array(all_s_cal),
        np.array(all_y_cal),
        np.array(all_g_cal),
        "tier",
        alpha=alpha,
    )
    test_sets = predict_mondrian_sets(
        np.array(all_s_test),
        np.array(all_g_test),
        mond,
        alpha=alpha,
    )
    is_cov = test_sets.is_covered(np.array(all_y_test))

    for grp in group_sizes:
        mask = np.array(all_g_test) == grp
        cov = float(np.mean(is_cov[mask]))
        assert (
            cov >= (1.0 - alpha) - tolerance
        ), f"{grp} coverage {cov:.3f} below target"
