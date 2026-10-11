"""Reliability audit module: per-group coverage, Wilson CIs, and flagging."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from core.config import DEFAULT_CONFIG, AuditLensConfig
from core.conformal import (
    compute_mondrian_thresholds,
    compute_split_conformal_threshold,
    predict_conformal_sets,
    predict_mondrian_sets,
)
from core.decisions import map_sets_to_decisions
from core.log_setup import logger
from core.splits import DatasetSplits
from core.validation import AuditDataset


@dataclass
class GroupAuditMetrics:
    """Detailed reliability metrics for a single subgroup."""

    group_column: str
    group_value: str
    n_samples: int
    split_coverage: float
    split_wilson_lower: float
    split_wilson_upper: float
    split_deferral_rate: float
    split_selective_risk: Optional[float]
    split_flagged: bool
    mondrian_coverage: float
    mondrian_wilson_lower: float
    mondrian_wilson_upper: float
    mondrian_deferral_rate: float
    mondrian_selective_risk: Optional[float]
    mondrian_flagged: bool
    is_unreliable: bool
    status_label: str


@dataclass
class AuditReportSummary:
    """Aggregated audit results across all groups for a chosen column."""

    group_column: str
    target_coverage: float
    overall_split_coverage: float
    group_metrics: List[GroupAuditMetrics]
    flagged_groups: List[str] = field(default_factory=list)
    unreliable_groups: List[str] = field(default_factory=list)


def compute_wilson_ci(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Computes the Wilson score interval for binomial proportions.

    Args:
        k: Number of successes (e.g. covered observations).
        n: Total number of trials (subgroup sample size).
        confidence: Two-sided confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if n <= 0:
        return 0.0, 1.0

    p_hat = k / n
    z = float(norm.ppf((1.0 + confidence) / 2.0))
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (p_hat + (z2 / (2.0 * n))) / denom
    spread = (
        z * math.sqrt((p_hat * (1.0 - p_hat)) / n + (z2 / (4.0 * (n**2))))
    ) / denom

    lower = 0.0 if k <= 0 else max(0.0, float(center - spread))
    upper = 1.0 if k >= n else min(1.0, float(center + spread))

    return lower, upper


def _calculate_subgroup_stats(
    y_true: np.ndarray,
    is_covered: np.ndarray,
    decisions: np.ndarray,
    mask: np.ndarray,
    confidence: float,
) -> Tuple[float, float, float, float, Optional[float]]:
    """Calculates coverage, Wilson CI, deferral rate, and selective risk for a group."""
    n_g = int(np.sum(mask))
    if n_g == 0:
        return 0.0, 0.0, 1.0, 1.0, None

    y_g = y_true[mask]
    cov_g = is_covered[mask]
    dec_g = decisions[mask]

    k_cov = int(np.sum(cov_g))
    coverage = float(k_cov / n_g)
    ci_low, ci_high = compute_wilson_ci(k_cov, n_g, confidence=confidence)

    def_rate = float(np.mean(dec_g == "defer"))
    auto_mask = (dec_g == "approve") | (dec_g == "refuse")
    n_auto = int(np.sum(auto_mask))

    if n_auto == 0:
        selective_risk: Optional[float] = None
    else:
        errs = np.sum((dec_g == "approve") & (y_g == 1)) + np.sum(
            (dec_g == "refuse") & (y_g == 0)
        )
        selective_risk = float(errs / n_auto)

    return coverage, ci_low, ci_high, def_rate, selective_risk


def audit_group_column(
    dataset: AuditDataset,
    splits: DatasetSplits,
    group_col: str,
    config: Optional[AuditLensConfig] = None,
    n_audited_groups: Optional[int] = None,
) -> AuditReportSummary:
    """Audits marginal split vs Mondrian conformal coverage side by side.

    Flag rule: A group is flagged only if (1) not unreliable, (2) its Bonferroni-adjusted
    Wilson UPPER bound is strictly below target coverage (statistically clear under-coverage),
    AND (3) empirical point coverage <= target_coverage - coverage_tolerance.

    Args:
        dataset: Validated AuditDataset.
        splits: Partitioned DatasetSplits.
        group_col: Group column name to audit.
        config: AuditLensConfig. Defaults to DEFAULT_CONFIG.
        n_audited_groups: Total number of subgroups tested (for dynamic Bonferroni adjustment).
            Defaults to count of subgroups in group_col.

    Returns:
        AuditReportSummary containing subgroup metrics and flagged groups.
    """
    cfg = config or DEFAULT_CONFIG
    alpha = cfg.alpha

    # 1. Marginal Split Conformal
    q_split = compute_split_conformal_threshold(
        splits.cal_score, splits.cal_outcome, alpha
    )
    split_sets = predict_conformal_sets(splits.test_score, q_split, alpha)
    split_dec = map_sets_to_decisions(split_sets)
    split_cov = split_sets.is_covered(splits.test_outcome)
    overall_cov = float(np.mean(split_cov))

    # 2. Mondrian Group Conformal
    mond_res = compute_mondrian_thresholds(
        splits.cal_score,
        splits.cal_outcome,
        splits.cal_data[group_col],
        group_col,
        alpha,
    )
    mond_sets = predict_mondrian_sets(
        splits.test_score, splits.test_data[group_col], mond_res, alpha
    )
    mond_dec = map_sets_to_decisions(mond_sets)
    mond_cov = mond_sets.is_covered(splits.test_outcome)

    y_test = splits.test_outcome.to_numpy()
    test_groups = splits.test_data[group_col].to_numpy()
    unique_vals = sorted(list(set(test_groups)))

    k_total_audited = (
        n_audited_groups if n_audited_groups is not None else len(unique_vals)
    )
    bonferroni_conf = 1.0 - (0.05 / max(1, k_total_audited))
    min_point_cov = cfg.target_coverage - cfg.coverage_tolerance

    metrics: List[GroupAuditMetrics] = []
    flagged: List[str] = []
    unreliable: List[str] = []

    for val in unique_vals:
        val_str = str(val)
        mask = test_groups == val
        n_g = int(np.sum(mask))

        is_unrel = (group_col, val) in dataset.unreliable_groups or (
            n_g < cfg.small_group_threshold
        )
        if is_unrel:
            unreliable.append(val_str)

        sp_cov, sp_low, sp_high, sp_def, sp_risk = _calculate_subgroup_stats(
            y_test, split_cov, split_dec, mask, bonferroni_conf
        )
        mo_cov, mo_low, mo_high, mo_def, mo_risk = _calculate_subgroup_stats(
            y_test, mond_cov, mond_dec, mask, bonferroni_conf
        )

        sp_flagged = bool(
            (not is_unrel)
            and (sp_high < cfg.target_coverage)
            and (sp_cov <= min_point_cov)
        )
        mo_flagged = bool(
            (not is_unrel)
            and (mo_high < cfg.target_coverage)
            and (mo_cov <= min_point_cov)
        )

        if sp_flagged:
            flagged.append(val_str)
            label = "Flagged (Under-covered)"
        elif is_unrel:
            label = "Unreliable (Small sample)"
        else:
            label = "Reliable"

        metrics.append(
            GroupAuditMetrics(
                group_column=group_col,
                group_value=val_str,
                n_samples=n_g,
                split_coverage=sp_cov,
                split_wilson_lower=sp_low,
                split_wilson_upper=sp_high,
                split_deferral_rate=sp_def,
                split_selective_risk=sp_risk,
                split_flagged=sp_flagged,
                mondrian_coverage=mo_cov,
                mondrian_wilson_lower=mo_low,
                mondrian_wilson_upper=mo_high,
                mondrian_deferral_rate=mo_def,
                mondrian_selective_risk=mo_risk,
                mondrian_flagged=mo_flagged,
                is_unreliable=is_unrel,
                status_label=label,
            )
        )

    logger.info(
        f"Audited column '{group_col}': {len(metrics)} groups, {len(flagged)} flagged, "
        f"{len(unreliable)} unreliable."
    )

    return AuditReportSummary(
        group_column=group_col,
        target_coverage=cfg.target_coverage,
        overall_split_coverage=overall_cov,
        group_metrics=metrics,
        flagged_groups=flagged,
        unreliable_groups=unreliable,
    )


def audit_all_groups(
    dataset: AuditDataset,
    splits: DatasetSplits,
    config: Optional[AuditLensConfig] = None,
) -> Dict[str, AuditReportSummary]:
    """Audits all configured group columns with joint family-wise Bonferroni correction.

    Args:
        dataset: Validated AuditDataset.
        splits: Partitioned DatasetSplits.
        config: AuditLensConfig. Defaults to DEFAULT_CONFIG.

    Returns:
        Mapping of group column name to AuditReportSummary.
    """
    cfg = config or DEFAULT_CONFIG
    total_subgroups = sum(
        len(splits.test_data[c].unique()) for c in dataset.group_columns
    )
    results: Dict[str, AuditReportSummary] = {}
    for col in dataset.group_columns:
        results[col] = audit_group_column(
            dataset,
            splits,
            col,
            config=cfg,
            n_audited_groups=max(1, total_subgroups),
        )
    return results
