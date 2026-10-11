"""Conformal prediction module: split conformal and Mondrian (group-aware)."""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from core.errors import ConformalFailure
from core.log_setup import logger


@dataclass
class ConformalPredictionSets:
    """Prediction sets for binary classification produced via conformal calibration."""

    include_0: np.ndarray
    include_1: np.ndarray
    alpha: float
    threshold_value: float | Dict[str, float]

    @property
    def set_sizes(self) -> np.ndarray:
        """Array of set sizes ({0}, {1} -> 1, {0,1} -> 2, empty -> 0)."""
        return self.include_0.astype(int) + self.include_1.astype(int)

    def is_covered(self, outcomes: pd.Series | np.ndarray) -> np.ndarray:
        """Returns boolean array indicating if actual outcome is in prediction set."""
        y = np.asarray(outcomes)
        return np.where(y == 0, self.include_0, self.include_1)

    def compute_coverage(self, outcomes: pd.Series | np.ndarray) -> float:
        """Computes empirical coverage on given outcomes."""
        covered = self.is_covered(outcomes)
        return float(np.mean(covered))


@dataclass
class MondrianResults:
    """Thresholds and under-sampled group metadata for group-conditional conformal."""

    group_column: str
    thresholds: Dict[str, float]
    under_sampled_groups: List[str] = field(default_factory=list)


def compute_nonconformity_scores(
    scores: pd.Series | np.ndarray,
    outcomes: pd.Series | np.ndarray,
) -> np.ndarray:
    """Computes nonconformity scores s(x, y) = 1 - p_y(x).

    Args:
        scores: Predicted probabilities of default (p_1).
        outcomes: Binary outcomes (0 = repaid, 1 = defaulted).

    Returns:
        Array of nonconformity scores.
    """
    s = np.asarray(scores)
    y = np.asarray(outcomes)
    # If y=0, p_0 = 1 - s, so s(x, 0) = 1 - (1 - s) = s
    # If y=1, p_1 = s, so s(x, 1) = 1 - s
    return np.where(y == 0, s, 1.0 - s)


def compute_split_conformal_threshold(
    cal_scores: pd.Series | np.ndarray,
    cal_outcomes: pd.Series | np.ndarray,
    alpha: float,
) -> float:
    """Computes split conformal threshold on calibration data using finite-sample quantile.

    Uses level ceil((n + 1) * (1 - alpha)) / n. If level > 1.0, returns infinity.
    Test labels must NEVER be passed to this function.

    Args:
        cal_scores: Calibration predicted default probabilities.
        cal_outcomes: Calibration actual outcomes.
        alpha: Significance level (1 - target_coverage).

    Returns:
        Conformal threshold scalar float (or float('inf')).
    """
    n = len(cal_scores)
    if n == 0:
        raise ConformalFailure("Calibration sample cannot be empty.")

    s_cal = compute_nonconformity_scores(cal_scores, cal_outcomes)
    k = math.ceil((n + 1) * (1.0 - alpha))

    if k > n:
        logger.warning(
            f"Finite-sample quantile level ceil(({n}+1)*(1-{alpha}))/{n} exceeds 1.0. "
            f"Sample size {n} too small for alpha={alpha}. Returning infinite threshold."
        )
        return float("inf")

    sorted_scores = np.sort(s_cal)
    threshold = float(sorted_scores[k - 1])
    logger.debug(f"Computed split threshold: {threshold:.4f} (rank {k}/{n})")
    return threshold


def predict_conformal_sets(
    test_scores: pd.Series | np.ndarray,
    threshold: float,
    alpha: float,
) -> ConformalPredictionSets:
    """Constructs conformal prediction sets for test scores using calibrated threshold.

    Label 0 is included if score <= threshold.
    Label 1 is included if 1 - score <= threshold (score >= 1 - threshold).

    Args:
        test_scores: Predicted probabilities for test instances.
        threshold: Calibrated split threshold.
        alpha: Significance level.

    Returns:
        ConformalPredictionSets instance.
    """
    s = np.asarray(test_scores)
    include_0 = s <= threshold
    include_1 = s >= (1.0 - threshold)

    return ConformalPredictionSets(
        include_0=include_0,
        include_1=include_1,
        alpha=alpha,
        threshold_value=threshold,
    )


def compute_mondrian_thresholds(
    cal_scores: pd.Series | np.ndarray,
    cal_outcomes: pd.Series | np.ndarray,
    cal_groups: pd.Series | np.ndarray,
    group_col_name: str,
    alpha: float,
) -> MondrianResults:
    """Computes group-specific conformal thresholds for values of one group column.

    Groups with insufficient calibration points get infinite threshold and defer.

    Args:
        cal_scores: Calibration scores.
        cal_outcomes: Calibration outcomes.
        cal_groups: Calibration group identities.
        group_col_name: Name of group attribute.
        alpha: Significance level.

    Returns:
        MondrianResults with per-group thresholds and under-sampled groups list.
    """
    scores_arr = np.asarray(cal_scores)
    outcomes_arr = np.asarray(cal_outcomes)
    groups_arr = np.asarray(cal_groups)

    unique_groups = np.unique(groups_arr)
    thresholds: Dict[str, float] = {}
    under_sampled: List[str] = []

    for grp in unique_groups:
        grp_str = str(grp)
        mask = groups_arr == grp
        n_g = int(np.sum(mask))
        k = math.ceil((n_g + 1) * (1.0 - alpha))
        if k > n_g or n_g == 0:
            logger.warning(
                f"Group '{group_col_name}={grp_str}' has {n_g} rows; rank {k} > {n_g}. "
                "Threshold set to infinity (all cases will defer)."
            )
            thresholds[grp_str] = float("inf")
            under_sampled.append(grp_str)
        else:
            s_g = compute_nonconformity_scores(scores_arr[mask], outcomes_arr[mask])
            thresholds[grp_str] = float(np.sort(s_g)[k - 1])

    return MondrianResults(
        group_column=group_col_name,
        thresholds=thresholds,
        under_sampled_groups=under_sampled,
    )


def predict_mondrian_sets(
    test_scores: pd.Series | np.ndarray,
    test_groups: pd.Series | np.ndarray,
    mondrian_res: MondrianResults,
    alpha: float,
) -> ConformalPredictionSets:
    """Generates prediction sets using per-group Mondrian thresholds.

    Unseen test groups or under-sampled groups receive infinite threshold.

    Args:
        test_scores: Predicted probabilities.
        test_groups: Group attribute for test instances.
        mondrian_res: Calibrated MondrianResults.
        alpha: Significance level.

    Returns:
        ConformalPredictionSets instance.
    """
    s = np.asarray(test_scores)
    g = np.asarray(test_groups)
    n = len(s)

    thresh_arr = np.zeros(n, dtype=float)
    for i in range(n):
        grp_str = str(g[i])
        thresh_arr[i] = mondrian_res.thresholds.get(grp_str, float("inf"))

    include_0 = s <= thresh_arr
    include_1 = s >= (1.0 - thresh_arr)

    return ConformalPredictionSets(
        include_0=include_0,
        include_1=include_1,
        alpha=alpha,
        threshold_value=mondrian_res.thresholds,
    )
