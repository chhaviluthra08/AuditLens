"""Decision mapping, risk evaluation, and capacity control simulation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG, AuditLensConfig
from core.conformal import (
    ConformalPredictionSets,
    compute_split_conformal_threshold,
    predict_conformal_sets,
)
from core.errors import CapacityError
from core.log_setup import logger


@dataclass
class DecisionAuditResults:
    """Outcomes and cost metrics resulting from set-to-decision mapping."""

    decisions: np.ndarray
    n_total: int
    n_approved: int
    n_refused: int
    n_deferred: int
    automation_rate: float
    deferral_rate: float
    selective_risk: Optional[float]
    expected_cost_j: float
    chosen_alpha: float
    threshold: float | Dict[str, float]


@dataclass
class CapacitySearchResult:
    """Summary of review capacity search over alpha grid."""

    optimal_alpha: float
    implied_target_coverage: float
    cal_deferral_rate: float
    test_deferral_rate: float
    below_target_coverage: bool
    test_results: DecisionAuditResults
    grid_curve: List[Dict[str, Any]]


def map_sets_to_decisions(sets: ConformalPredictionSets) -> np.ndarray:
    """Maps prediction sets to approve, refuse, or defer.

    {0} -> approve
    {1} -> refuse
    {0, 1} or empty -> defer (empty prediction sets count as defer).

    Args:
        sets: ConformalPredictionSets instance.

    Returns:
        Array of string decisions: 'approve', 'refuse', 'defer'.
    """
    n = len(sets.include_0)
    decisions = np.full(n, "defer", dtype=object)

    singleton_0 = sets.include_0 & (~sets.include_1)
    singleton_1 = (~sets.include_0) & sets.include_1

    decisions[singleton_0] = "approve"
    decisions[singleton_1] = "refuse"
    # Both included or neither included remain 'defer'
    return decisions


def compute_decisions(
    sets: ConformalPredictionSets,
    outcomes: pd.Series | np.ndarray,
    config: Optional[AuditLensConfig] = None,
) -> DecisionAuditResults:
    """Computes decision counts, automation rate, selective risk, and expected cost.

    Note: Expected cost J = r*P(approve & default) + P(refuse & repaid) + delta*P(defer).
    This formulation assumes human reviewers resolve deferred cases correctly.

    Args:
        sets: ConformalPredictionSets instance.
        outcomes: Actual ground-truth binary outcomes.
        config: AuditLensConfig. Defaults to DEFAULT_CONFIG.

    Returns:
        DecisionAuditResults instance.
    """
    cfg = config or DEFAULT_CONFIG
    y = np.asarray(outcomes)
    decisions = map_sets_to_decisions(sets)
    n = len(decisions)

    app_mask = decisions == "approve"
    ref_mask = decisions == "refuse"
    def_mask = decisions == "defer"

    n_app = int(np.sum(app_mask))
    n_ref = int(np.sum(ref_mask))
    n_def = int(np.sum(def_mask))
    n_auto = n_app + n_ref

    if n_auto == 0:
        selective_risk: Optional[float] = None
    else:
        errs = np.sum(app_mask & (y == 1)) + np.sum(ref_mask & (y == 0))
        selective_risk = float(errs / n_auto)

    p_app_def = float(np.sum(app_mask & (y == 1)) / n)
    p_ref_rep = float(np.sum(ref_mask & (y == 0)) / n)
    p_defer = float(n_def / n)

    cost_j = (
        (cfg.cost_ratio_r * p_app_def) + p_ref_rep + (cfg.review_cost_delta * p_defer)
    )

    return DecisionAuditResults(
        decisions=decisions,
        n_total=n,
        n_approved=n_app,
        n_refused=n_ref,
        n_deferred=n_def,
        automation_rate=float(n_auto / n),
        deferral_rate=p_defer,
        selective_risk=selective_risk,
        expected_cost_j=cost_j,
        chosen_alpha=sets.alpha,
        threshold=sets.threshold_value,
    )


def search_capacity_constrained_alpha(
    cal_scores: pd.Series | np.ndarray,
    cal_outcomes: pd.Series | np.ndarray,
    test_scores: pd.Series | np.ndarray,
    test_outcomes: pd.Series | np.ndarray,
    config: Optional[AuditLensConfig] = None,
    alpha_grid: Optional[np.ndarray] = None,
) -> CapacitySearchResult:
    """Finds smallest alpha whose calibration deferral rate fits review capacity.

    Estimates deferral rates strictly from the calibration split to prevent data
    leakage. Test split is evaluated only for the selected optimal alpha.
    Expected cost assumes human reviewers resolve deferred cases correctly.

    Args:
        cal_scores: Calibration scores.
        cal_outcomes: Calibration outcomes.
        test_scores: Test scores for final evaluation.
        test_outcomes: Test outcomes.
        config: AuditLensConfig. Defaults to DEFAULT_CONFIG.
        alpha_grid: Candidate alpha array. Defaults to linspace(0.01, 0.50, 50).

    Returns:
        CapacitySearchResult with optimal alpha and test split metrics.

    Raises:
        CapacityError: If no candidate alpha fits review capacity.
    """
    cfg = config or DEFAULT_CONFIG
    grid = (
        alpha_grid
        if alpha_grid is not None
        else np.linspace(0.01, 0.50, 50, dtype=float)
    )
    sorted_grid = np.sort(grid)

    curve_data: List[Dict[str, Any]] = []
    optimal_alpha: Optional[float] = None
    best_cal_deferral: Optional[float] = None

    for cand_alpha in sorted_grid:
        alpha_val = float(cand_alpha)
        q_cal = compute_split_conformal_threshold(cal_scores, cal_outcomes, alpha_val)
        cal_sets = predict_conformal_sets(cal_scores, q_cal, alpha_val)
        cal_decisions = map_sets_to_decisions(cal_sets)
        cal_def_rate = float(np.mean(cal_decisions == "defer"))

        curve_data.append(
            {
                "alpha": alpha_val,
                "target_coverage": 1.0 - alpha_val,
                "cal_deferral_rate": cal_def_rate,
                "threshold": q_cal,
            }
        )

        if optimal_alpha is None and cal_def_rate <= cfg.review_capacity:
            optimal_alpha = alpha_val
            best_cal_deferral = cal_def_rate

    if optimal_alpha is None:
        msg = (
            f"Review capacity {cfg.review_capacity:.1%} cannot be satisfied by any "
            f"alpha on grid [{sorted_grid[0]:.2f}, {sorted_grid[-1]:.2f}]."
        )
        hint = "Increase review capacity or adjust model score calibration."
        logger.error(msg)
        raise CapacityError(msg, fix_hint=hint)

    q_opt = compute_split_conformal_threshold(cal_scores, cal_outcomes, optimal_alpha)
    test_sets = predict_conformal_sets(test_scores, q_opt, optimal_alpha)
    test_res = compute_decisions(test_sets, test_outcomes, config=cfg)

    logger.info(
        f"Capacity search selected alpha={optimal_alpha:.4f} (cal_defer={best_cal_deferral:.3f}, "
        f"test_defer={test_res.deferral_rate:.3f})."
    )

    implied_cov = round(1.0 - optimal_alpha, 4)
    below_target = bool(implied_cov < cfg.target_coverage - 1e-4)

    return CapacitySearchResult(
        optimal_alpha=optimal_alpha,
        implied_target_coverage=implied_cov,
        cal_deferral_rate=best_cal_deferral or 0.0,
        test_deferral_rate=test_res.deferral_rate,
        below_target_coverage=below_target,
        test_results=test_res,
        grid_curve=curve_data,
    )
