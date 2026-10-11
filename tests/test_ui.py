"""Smoke tests for Streamlit UI using AppTest."""

from streamlit.testing.v1 import AppTest


def test_app_initial_load() -> None:
    """Verifies initial app launch renders title and prompt without exceptions."""
    at = AppTest.from_file("../app/main.py")
    at.run()

    assert len(at.exception) == 0
    assert any("AuditLens" in t.value for t in at.title)
    # Check that initial screen prompts user to select data
    assert any(
        "upload a CSV file or select a Demo Model" in info.value for info in at.info
    )


def test_app_loads_good_demo_model() -> None:
    """Smoke test: Selecting Good Model executes end-to-end and populates metrics."""
    at = AppTest.from_file("../app/main.py")
    at.run()

    # Select Good Model from sidebar
    at.sidebar.selectbox[0].select("Good Model (UCI LightGBM)")
    at.run()

    assert len(at.exception) == 0
    assert len(at.metric) >= 10
    # ROC AUC should be present
    assert any("ROC AUC" in m.label for m in at.metric)
    # Check that tabs rendered
    assert len(at.tabs) == 4


def test_app_loads_flawed_demo_model() -> None:
    """Smoke test: Selecting Flawed Model executes end-to-end and flags distorted group."""
    at = AppTest.from_file("../app/main.py")
    at.run()

    # Select Flawed Model from sidebar
    at.sidebar.selectbox[0].select("Flawed Model (Biased < 25)")
    at.run()

    # Select age_band for audit
    at.selectbox[0].select("age_band")
    at.run()

    assert len(at.exception) == 0
    assert len(at.metric) >= 10
    # Tab 3 should raise a flagged alert on flawed model
    assert any("under-coverage detected" in err.value for err in at.error)


def test_app_report_download_button_available() -> None:
    """Smoke test: Download report button is rendered and available in sidebar."""
    at = AppTest.from_file("../app/main.py")
    at.run()
    at.sidebar.selectbox[0].select("Good Model (UCI LightGBM)")
    at.run()

    assert len(at.sidebar.download_button) == 1
    assert "Download Audit Report" in at.sidebar.download_button[0].label


def test_dot_plot_xaxis_range_includes_coverage_78() -> None:
    """Verifies dot plot x-axis starts below min Wilson lower bound and covers 0.78 up to 1.0."""
    from app.tabs.tab_audit import plot_coverage_dot_plot
    from core.audit import AuditReportSummary, GroupAuditMetrics

    m = GroupAuditMetrics(
        group_column="test_group",
        group_value="distorted_grp",
        n_samples=500,
        split_coverage=0.78,
        split_wilson_lower=0.7416,
        split_wilson_upper=0.8141,
        split_deferral_rate=0.25,
        split_selective_risk=0.08,
        split_flagged=True,
        mondrian_coverage=0.905,
        mondrian_wilson_lower=0.875,
        mondrian_wilson_upper=0.925,
        mondrian_deferral_rate=0.22,
        mondrian_selective_risk=0.04,
        mondrian_flagged=False,
        is_unreliable=False,
        status_label="Flagged (Under-covered)",
    )
    summary = AuditReportSummary(
        group_column="test_group",
        target_coverage=0.90,
        overall_split_coverage=0.89,
        group_metrics=[m],
        flagged_groups=["distorted_grp"],
    )

    fig = plot_coverage_dot_plot(summary)
    x_range = fig.layout.xaxis.range

    # Range must reach up to 1.0
    assert x_range[1] == 1.0
    # Range start must be strictly below the Wilson lower bound (with margin)
    assert x_range[0] < m.split_wilson_lower
    # Point coverage 0.78 must be cleanly inside plotted range
    assert x_range[0] <= 0.78 <= x_range[1]
    # Target line must be shown at target_coverage (0.90)
    assert any(
        getattr(s, "x0", None) == summary.target_coverage for s in fig.layout.shapes
    )
