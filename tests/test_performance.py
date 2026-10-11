"""Unit tests for core/performance.py."""

import numpy as np
import pandas as pd
import pytest

from core.errors import InvalidInputError
from core.performance import (
    compute_calibration_curve,
    compute_ks_statistic,
    compute_psi,
    evaluate_performance,
)
from core.splits import create_stratified_split
from core.validation import validate_and_load


def test_ks_statistic_computation() -> None:
    """Verifies KS statistic bounds and ordering."""
    # Completely separated distributions: goods ~ 0.1, bads ~ 0.9
    goods = np.array([0.05, 0.10, 0.15, 0.20])
    bads = np.array([0.80, 0.85, 0.90, 0.95])
    scores = np.concatenate([goods, bads])
    outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    stat, pval = compute_ks_statistic(scores, outcomes)
    assert stat == pytest.approx(1.0, abs=1e-5)
    assert pval < 0.05

    # Test error if only one class provided
    with pytest.raises(InvalidInputError, match="Both positive and negative outcomes"):
        compute_ks_statistic(goods, np.zeros_like(goods))


def test_psi_identical_vs_shifted() -> None:
    """Verifies PSI near zero for similar distributions and high for drifted ones."""
    rng = np.random.default_rng(42)
    ref = rng.beta(2, 5, size=2000)
    comp_same = rng.beta(2, 5, size=2000)
    comp_shifted = rng.beta(5, 2, size=2000)  # substantial rightward drift

    psi_same = compute_psi(ref, comp_same, n_bins=10)
    psi_drift = compute_psi(ref, comp_shifted, n_bins=10)

    # Identical distribution should have PSI < 0.05
    assert psi_same < 0.05
    # Substantially shifted distribution should have high PSI (> 0.25)
    assert psi_drift > 0.50


def test_calibration_curve() -> None:
    """Verifies calibration curve produces valid coordinates."""
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    prob_true, prob_pred = compute_calibration_curve(y, scores, n_bins=2)

    assert len(prob_true) == 2
    assert len(prob_pred) == 2
    assert (prob_true >= 0.0).all() and (prob_true <= 1.0).all()
    assert (prob_pred >= 0.0).all() and (prob_pred <= 1.0).all()


def test_evaluate_performance_integration(valid_dataframe: pd.DataFrame) -> None:
    """Verifies full end-to-end performance evaluation on split data."""
    dataset = validate_and_load(valid_dataframe)
    splits = create_stratified_split(dataset)
    metrics = evaluate_performance(splits)

    assert 0.5 <= metrics.auc <= 1.0
    assert 0.0 <= metrics.ks_statistic <= 1.0
    assert 0.0 <= metrics.brier_score <= 1.0
    assert metrics.psi >= 0.0
    assert len(metrics.calibration_curve_true) > 0
    assert len(metrics.calibration_curve_pred) > 0
    assert metrics.n_samples == 600
