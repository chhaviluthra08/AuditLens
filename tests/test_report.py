"""Unit tests for core/report.py."""

import numpy as np

from core.audit import AuditReportSummary, GroupAuditMetrics
from core.config import DEFAULT_CONFIG
from core.decisions import CapacitySearchResult, DecisionAuditResults
from core.performance import ModelPerformanceMetrics
from core.report import AuditReportContext, render_html_report


def test_html_autoescape_xss_protection() -> None:
    """Verifies Jinja2 autoescaping prevents script injection in group labels."""
    malicious_label = "<script>alert('xss')</script>"
    metric = GroupAuditMetrics(
        group_column="malicious_col",
        group_value=malicious_label,
        n_samples=500,
        split_coverage=0.85,
        split_wilson_lower=0.80,
        split_wilson_upper=0.88,
        split_deferral_rate=0.20,
        split_selective_risk=0.05,
        split_flagged=True,
        mondrian_coverage=0.90,
        mondrian_wilson_lower=0.86,
        mondrian_wilson_upper=0.93,
        mondrian_deferral_rate=0.22,
        mondrian_selective_risk=0.04,
        mondrian_flagged=False,
        is_unreliable=False,
        status_label="Flagged (Under-covered)",
    )

    summary = AuditReportSummary(
        group_column="malicious_col",
        target_coverage=0.90,
        overall_split_coverage=0.89,
        group_metrics=[metric],
        flagged_groups=[malicious_label],
    )

    perf = ModelPerformanceMetrics(
        auc=0.75,
        ks_statistic=0.40,
        ks_pvalue=1e-10,
        brier_score=0.15,
        psi=0.01,
        calibration_curve_true=np.array([0.1, 0.5]),
        calibration_curve_pred=np.array([0.15, 0.45]),
        n_samples=1000,
    )

    ctx = AuditReportContext(
        dataset_name="XSS_Test",
        data_provenance="Unit test synthetic injection",
        n_rows=1000,
        default_rate=0.20,
        config=DEFAULT_CONFIG,
        perf_metrics=perf,
        audits_by_column={"malicious_col": summary},
    )

    html = render_html_report(ctx)

    # Must contain HTML-escaped entity, NOT unescaped <script> tag
    assert (
        "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in html
        or "&lt;script&gt;alert('xss')&lt;/script&gt;" in html
    )
    assert "<script>alert('xss')</script>" not in html


def test_report_sections_and_regulatory_mapping_integrity() -> None:
    """Verifies required sections, limitations, and regulatory topic mapping."""
    perf = ModelPerformanceMetrics(
        auc=0.78,
        ks_statistic=0.42,
        ks_pvalue=1e-15,
        brier_score=0.14,
        psi=0.02,
        calibration_curve_true=np.array([0.2]),
        calibration_curve_pred=np.array([0.2]),
        n_samples=5000,
    )

    dec_res = DecisionAuditResults(
        decisions=np.array(["approve"]),
        n_total=5000,
        n_approved=4000,
        n_refused=500,
        n_deferred=500,
        automation_rate=0.90,
        deferral_rate=0.10,
        selective_risk=0.03,
        expected_cost_j=0.45,
        chosen_alpha=0.10,
        threshold=0.75,
    )

    cap_res = CapacitySearchResult(
        optimal_alpha=0.10,
        implied_target_coverage=0.90,
        cal_deferral_rate=0.09,
        test_deferral_rate=0.10,
        below_target_coverage=False,
        test_results=dec_res,
        grid_curve=[],
    )

    ctx = AuditReportContext(
        dataset_name="Good_Model",
        data_provenance="UCI Credit Card dataset holdout",
        n_rows=5000,
        default_rate=0.22,
        config=DEFAULT_CONFIG,
        perf_metrics=perf,
        audits_by_column={},
        capacity_res=cap_res,
    )

    html = render_html_report(ctx)

    # Regulatory topics must appear by topic name only
    assert "Independent Validation" in html
    assert "Ongoing Monitoring" in html
    assert "Explainability" in html
    assert "Bias and Fairness" in html
    assert "Model Inventory" in html
    assert "Indicative mapping only; verify against the final RBI guidance." in html

    # Assert no statutory clause or paragraph citations exist
    assert "Section " not in html
    assert "Clause " not in html
    assert "Paragraph " not in html

    # Limitations content check
    assert "covers strictly the uploaded sample" in html
    assert (
        "Mondrian conformal prediction is evaluated one group column at a time" in html
    )
    assert "Taiwan UCI Credit Card dataset" in html
