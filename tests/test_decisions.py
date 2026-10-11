"""Unit tests for core/decisions.py."""

import numpy as np
import pytest

from core.config import AuditLensConfig
from core.conformal import ConformalPredictionSets
from core.decisions import (
    compute_decisions,
    map_sets_to_decisions,
    search_capacity_constrained_alpha,
)
from core.errors import CapacityError


def test_map_sets_to_decisions() -> None:
    """Verifies set-to-decision mapping rules including empty sets."""
    # 4 cases: {0}, {1}, {0, 1}, empty set
    inc_0 = np.array([True, False, True, False])
    inc_1 = np.array([False, True, True, False])
    sets = ConformalPredictionSets(
        include_0=inc_0,
        include_1=inc_1,
        alpha=0.10,
        threshold_value=0.5,
    )

    decisions = map_sets_to_decisions(sets)
    assert decisions[0] == "approve"  # {0}
    assert decisions[1] == "refuse"  # {1}
    assert decisions[2] == "defer"  # {0, 1}
    assert decisions[3] == "defer"  # empty set counts as defer


def test_selective_risk_with_zero_automated_cases_returns_none() -> None:
    """Verifies selective risk returns None (not NaN or crash) when 100% of cases defer."""
    # All cases are {0, 1}
    inc_0 = np.array([True, True, True])
    inc_1 = np.array([True, True, True])
    sets = ConformalPredictionSets(
        include_0=inc_0,
        include_1=inc_1,
        alpha=0.10,
        threshold_value=0.5,
    )
    outcomes = np.array([0, 1, 0])

    res = compute_decisions(sets, outcomes)
    assert res.n_deferred == 3
    assert res.automation_rate == 0.0
    assert res.selective_risk is None


def test_selective_risk_and_expected_cost_calculation() -> None:
    """Verifies selective risk and expected cost J calculation."""
    # 4 cases:
    # 0: {0} (approve), actual=0 (repaid) -> Correct automated
    # 1: {0} (approve), actual=1 (default) -> Error: approved defaulter
    # 2: {1} (refuse), actual=0 (repaid)  -> Error: refused good applicant
    # 3: {0,1} (defer), actual=1 (default) -> Deferred
    inc_0 = np.array([True, True, False, True])
    inc_1 = np.array([False, False, True, True])
    outcomes = np.array([0, 1, 0, 1])

    sets = ConformalPredictionSets(
        include_0=inc_0,
        include_1=inc_1,
        alpha=0.10,
        threshold_value=0.5,
    )
    cfg = AuditLensConfig(cost_ratio_r=5.0, review_cost_delta=0.2)

    res = compute_decisions(sets, outcomes, config=cfg)
    assert res.n_total == 4
    assert res.n_approved == 2
    assert res.n_refused == 1
    assert res.n_deferred == 1
    assert res.automation_rate == 0.75  # 3/4
    assert res.deferral_rate == 0.25  # 1/4

    # Automated cases = 3 (cases 0, 1, 2)
    # Errors = case 1 (app&def) + case 2 (ref&rep) = 2 errors out of 3
    assert res.selective_risk == pytest.approx(2.0 / 3.0, abs=1e-5)

    # Cost J = r * P(app & def) + 1 * P(ref & rep) + delta * P(defer)
    # P(app & def) = 1/4 = 0.25
    # P(ref & rep) = 1/4 = 0.25
    # P(defer) = 1/4 = 0.25
    # J = 5.0 * 0.25 + 1.0 * 0.25 + 0.2 * 0.25 = 1.25 + 0.25 + 0.05 = 1.55
    assert res.expected_cost_j == pytest.approx(1.55, abs=1e-5)


def test_capacity_search_uses_calibration_split() -> None:
    """Verifies capacity search estimates deferral from calibration and reports on test."""
    rng = np.random.default_rng(42)
    n = 600
    s_cal = rng.beta(2, 5, size=n)
    y_cal = (rng.uniform(0, 1, size=n) < s_cal).astype(int)
    s_test = rng.beta(2, 5, size=n)
    y_test = (rng.uniform(0, 1, size=n) < s_test).astype(int)

    cfg = AuditLensConfig(review_capacity=0.25)
    res = search_capacity_constrained_alpha(s_cal, y_cal, s_test, y_test, config=cfg)

    assert 0.01 <= res.optimal_alpha <= 0.50
    assert res.implied_target_coverage == pytest.approx(
        1.0 - res.optimal_alpha, abs=1e-4
    )
    assert res.cal_deferral_rate <= 0.25
    assert 0.0 <= res.test_deferral_rate <= 1.0
    assert isinstance(res.below_target_coverage, bool)
    assert res.test_results.n_total == n
    assert len(res.grid_curve) > 0


def test_capacity_search_below_target_coverage_flag() -> None:
    """Verifies below_target_coverage is True when tight capacity forces higher alpha."""
    rng = np.random.default_rng(42)
    n = 600
    s_cal = rng.beta(2, 5, size=n)
    y_cal = (rng.uniform(0, 1, size=n) < s_cal).astype(int)
    s_test = rng.beta(2, 5, size=n)
    y_test = (rng.uniform(0, 1, size=n) < s_test).astype(int)

    # With a very tight capacity (e.g. 5%), alpha will be forced > 0.10 (so implied cov < 0.90)
    cfg = AuditLensConfig(target_coverage=0.90, review_capacity=0.08)
    res = search_capacity_constrained_alpha(s_cal, y_cal, s_test, y_test, config=cfg)

    assert res.optimal_alpha > 0.10
    assert res.implied_target_coverage < 0.90
    assert res.below_target_coverage is True


def test_capacity_search_raises_capacity_error_when_unreachable() -> None:
    """Verifies CapacityError is raised if no alpha on grid fits capacity."""
    s_cal = np.array([0.5] * 100)
    y_cal = np.array([0, 1] * 50)
    s_test = s_cal.copy()
    y_test = y_cal.copy()

    # Review capacity 0% cannot be satisfied by ambiguous 0.5 predictions
    cfg = AuditLensConfig(review_capacity=0.0)
    with pytest.raises(CapacityError, match="cannot be satisfied"):
        search_capacity_constrained_alpha(s_cal, y_cal, s_test, y_test, config=cfg)
