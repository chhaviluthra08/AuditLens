"""Configuration defaults for AuditLens."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditLensConfig:
    """Immutable application configuration and auditing parameters.

    Attributes:
        target_coverage: Target marginal / group coverage (1 - alpha).
        cost_ratio_r: Cost ratio of approving a defaulter vs refusing a good applicant.
        review_cost_delta: Cost per case deferred to human review.
        review_capacity: Max fraction of total applicants human reviewers can handle.
        small_group_threshold: Minimum sample size before flagging small-group warning.
        min_rows: Minimum total rows required in uploaded CSV.
        coverage_tolerance: Minimum coverage gap required below target (default 0.03).
            Flagging rule: A group is flagged if and only if (1) not unreliable,
            (2) Wilson upper bound < target_coverage (under Bonferroni confidence),
            AND (3) empirical coverage <= target_coverage - coverage_tolerance.
        calibration_split_ratio: Fraction of data assigned to calibration set (default 0.5).
        n_calibration_bins: Number of bins for reliability/calibration curve.
        psi_bins: Number of quantile bins for PSI computation.
        score_decision_threshold: Score cutoff for baseline classification policy (default 0.50).
        wilson_confidence_level: Baseline two-sided confidence level before Bonferroni adjustment (default 0.95).
        log_file_path: Path for output application log file.
    """

    target_coverage: float = 0.90
    cost_ratio_r: float = 5.0
    review_cost_delta: float = 0.2
    review_capacity: float = 0.20
    small_group_threshold: int = 100
    min_rows: int = 1000
    random_seed: int = 42
    coverage_tolerance: float = 0.03
    calibration_split_ratio: float = 0.50
    n_calibration_bins: int = 10
    psi_bins: int = 10
    score_decision_threshold: float = 0.50
    wilson_confidence_level: float = 0.95
    log_file_path: str = "logs/auditlens.log"

    @property
    def alpha(self) -> float:
        """Significance level alpha = 1 - target_coverage."""
        return round(1.0 - self.target_coverage, 4)


DEFAULT_CONFIG = AuditLensConfig()
