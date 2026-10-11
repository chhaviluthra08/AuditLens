"""Performance and stability evaluation metrics for credit models."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from core.config import DEFAULT_CONFIG, AuditLensConfig
from core.errors import InvalidInputError
from core.log_setup import logger
from core.splits import DatasetSplits


@dataclass
class ModelPerformanceMetrics:
    """Container holding performance and stability evaluation results."""

    auc: float
    ks_statistic: float
    ks_pvalue: float
    brier_score: float
    psi: float
    calibration_curve_true: np.ndarray
    calibration_curve_pred: np.ndarray
    n_samples: int


def compute_ks_statistic(
    scores: pd.Series | np.ndarray,
    outcomes: pd.Series | np.ndarray,
) -> Tuple[float, float]:
    """Computes Kolmogorov-Smirnov statistic between good and bad score distributions.

    Args:
        scores: Predicted probabilities.
        outcomes: Binary outcomes (0 = repaid, 1 = defaulted).

    Returns:
        Tuple of (ks_statistic, p_value).
    """
    s_arr = np.asarray(scores)
    y_arr = np.asarray(outcomes)

    scores_good = s_arr[y_arr == 0]
    scores_bad = s_arr[y_arr == 1]

    if len(scores_good) == 0 or len(scores_bad) == 0:
        raise InvalidInputError(
            "Both positive and negative outcomes required to compute KS statistic.",
            fix_hint="Ensure evaluation sample contains both 0 and 1 classes.",
        )

    res = ks_2samp(scores_good, scores_bad)
    logger.debug(f"KS statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4e}")
    return float(res.statistic), float(res.pvalue)


def compute_psi(
    reference: pd.Series | np.ndarray,
    comparison: pd.Series | np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Computes Population Stability Index (PSI) between two score distributions.

    Args:
        reference: Reference score distribution (e.g. calibration set).
        comparison: Comparison score distribution (e.g. test set).
        n_bins: Number of quantile bins.
        epsilon: Smoothing term to avoid log(0) and zero division.

    Returns:
        PSI scalar value.
    """
    ref = np.asarray(reference)
    comp = np.asarray(comparison)

    if len(ref) == 0 or len(comp) == 0:
        raise InvalidInputError("Distributions cannot be empty for PSI calculation.")

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(ref, quantiles)
    # Ensure distinct bin edges for histogram binning
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    comp_counts, _ = np.histogram(comp, bins=bin_edges)

    ref_pct = np.clip(ref_counts / len(ref), epsilon, 1.0)
    comp_pct = np.clip(comp_counts / len(comp), epsilon, 1.0)

    ref_pct /= ref_pct.sum()
    comp_pct /= comp_pct.sum()

    psi_val = float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))
    logger.debug(f"Computed PSI across {len(bin_edges)-1} bins: {psi_val:.4f}")
    return max(0.0, psi_val)


def compute_calibration_curve(
    outcomes: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes empirical calibration curve coordinates.

    Args:
        outcomes: Binary actual outcomes.
        scores: Predicted probabilities.
        n_bins: Number of probability bins.

    Returns:
        Tuple of (prob_true, prob_pred) arrays.
    """
    y_arr = np.asarray(outcomes)
    s_arr = np.asarray(scores)

    prob_true, prob_pred = calibration_curve(
        y_arr, s_arr, n_bins=n_bins, strategy="uniform"
    )
    return prob_true, prob_pred


def evaluate_performance(
    splits: DatasetSplits,
    config: Optional[AuditLensConfig] = None,
) -> ModelPerformanceMetrics:
    """Evaluates discrimination, calibration, and stability on test split vs calibration.

    Args:
        splits: DatasetSplits containing calibration and test data.
        config: Audit configuration. Defaults to DEFAULT_CONFIG.

    Returns:
        ModelPerformanceMetrics instance.
    """
    cfg = config or DEFAULT_CONFIG
    y_test = splits.test_outcome.to_numpy()
    s_test = splits.test_score.to_numpy()

    logger.info(f"Evaluating performance on {len(y_test)} test observations.")

    auc = float(roc_auc_score(y_test, s_test))
    ks_stat, ks_pval = compute_ks_statistic(s_test, y_test)
    brier = float(brier_score_loss(y_test, s_test))
    psi_val = compute_psi(
        splits.cal_score.to_numpy(),
        s_test,
        n_bins=cfg.psi_bins,
    )
    prob_true, prob_pred = compute_calibration_curve(
        y_test, s_test, n_bins=cfg.n_calibration_bins
    )

    logger.info(
        f"Performance metrics: AUC={auc:.4f}, KS={ks_stat:.4f}, Brier={brier:.4f}, PSI={psi_val:.4f}."
    )

    return ModelPerformanceMetrics(
        auc=auc,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        brier_score=brier,
        psi=psi_val,
        calibration_curve_true=prob_true,
        calibration_curve_pred=prob_pred,
        n_samples=len(y_test),
    )
