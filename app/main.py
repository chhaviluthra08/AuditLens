"""AuditLens: Credit-model audit web application."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from app.tabs.tab_audit import render_audit_tab
from app.tabs.tab_capacity import render_capacity_tab
from app.tabs.tab_performance import render_performance_tab
from app.tabs.tab_upload import render_upload_tab
from core.audit import audit_all_groups, audit_group_column
from core.config import DEFAULT_CONFIG, AuditLensConfig
from core.decisions import search_capacity_constrained_alpha
from core.errors import AuditLensError, CapacityError
from core.log_setup import logger
from core.performance import ModelPerformanceMetrics
from core.report import AuditReportContext, render_html_report
from core.splits import DatasetSplits, create_stratified_split
from core.validation import AuditDataset, validate_and_load


def _init_page() -> None:
    """Configures Streamlit page metadata and branding."""
    st.set_page_config(
        page_title="AuditLens — Credit Model Audit",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("🛡️ AuditLens: Credit Model Reliability & Fair Lending Audit")


def _render_sidebar_settings() -> AuditLensConfig:
    """Renders configuration settings in sidebar."""
    st.sidebar.header("Audit Configuration")
    target = st.sidebar.slider(
        "Target Coverage (1 - alpha)",
        min_value=0.70,
        max_value=0.99,
        value=DEFAULT_CONFIG.target_coverage,
        step=0.01,
    )
    r = st.sidebar.number_input(
        "Cost Ratio (r = Cost App&Def / Cost Ref&Rep)",
        min_value=1.0,
        max_value=20.0,
        value=DEFAULT_CONFIG.cost_ratio_r,
        step=0.5,
    )
    delta = st.sidebar.number_input(
        "Human Review Cost (delta)",
        min_value=0.01,
        max_value=2.0,
        value=DEFAULT_CONFIG.review_cost_delta,
        step=0.05,
    )
    cap = st.sidebar.slider(
        "Human Review Capacity (%)",
        min_value=0.05,
        max_value=0.50,
        value=DEFAULT_CONFIG.review_capacity,
        step=0.01,
    )
    thresh = st.sidebar.number_input(
        "Small Group Warning Threshold",
        min_value=20,
        max_value=500,
        value=DEFAULT_CONFIG.small_group_threshold,
        step=10,
    )

    return AuditLensConfig(
        target_coverage=target,
        cost_ratio_r=r,
        review_cost_delta=delta,
        review_capacity=cap,
        small_group_threshold=int(thresh),
    )


def _render_sidebar_data_loader() -> Tuple[str, Optional[pd.DataFrame]]:
    """Renders dataset selector and uploader controls."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Ingestion")
    choice = st.sidebar.selectbox(
        "Load Demo Dataset",
        [
            "None (Upload CSV)",
            "Good Model (UCI LightGBM)",
            "Flawed Model (Biased < 25)",
        ],
    )

    df_raw = None
    if choice == "Good Model (UCI LightGBM)":
        df_raw = pd.read_csv("data/demo/good_model.csv")
        st.sidebar.info("Demo: Well-calibrated model derived from UCI Credit dataset.")
    elif choice == "Flawed Model (Biased < 25)":
        df_raw = pd.read_csv("data/demo/flawed_model.csv")
        st.sidebar.warning("Demo: Flawed model with under-estimated risk for '< 25'.")
    else:
        uploaded = st.sidebar.file_uploader("Upload Credit Model CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)

    return choice, df_raw


def _configure_group_columns(
    df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, str]]:
    """Configures demographic columns and operational tags."""
    potential_groups = [
        c for c in df.columns if c not in ("score", "outcome", "timestamp")
    ]
    st.sidebar.markdown("---")
    st.sidebar.subheader("Demographic & Operational Groups")
    selected_groups = st.sidebar.multiselect(
        "Select Group Columns",
        options=potential_groups,
        default=potential_groups,
    )

    tags: Dict[str, str] = {}
    for col in selected_groups:
        tag_val = st.sidebar.selectbox(
            f"Tag '{col}'",
            ["operational", "audit-only"],
            index=0 if "limit" in col.lower() else 1,
            key=f"tag_{col}",
        )
        tags[col] = tag_val

    return selected_groups, tags


@st.cache_data
def _get_cached_split(
    df: pd.DataFrame,
    group_cols: List[str],
    tags: Dict[str, str],
    seed: int,
    cal_ratio: float,
    min_rows: int,
    small_thresh: int,
) -> Tuple[AuditDataset, DatasetSplits]:
    """Caches data validation and stratified splitting."""
    cfg = AuditLensConfig(min_rows=min_rows, small_group_threshold=small_thresh)
    dataset = validate_and_load(
        df, group_columns=group_cols, group_tags=tags, config=cfg
    )
    splits = create_stratified_split(dataset, seed=seed, cal_ratio=cal_ratio)
    return dataset, splits


@st.cache_data
def _get_cached_perf(
    cal_scores: Tuple[float, ...],
    cal_outcomes: Tuple[int, ...],
    test_scores: Tuple[float, ...],
    test_outcomes: Tuple[int, ...],
    psi_bins: int,
    cal_bins: int,
) -> ModelPerformanceMetrics:
    """Caches performance metrics evaluation."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from core.performance import (
        compute_calibration_curve,
        compute_ks_statistic,
        compute_psi,
    )

    s_test = pd.Series(test_scores)
    y_test = pd.Series(test_outcomes)
    s_cal = pd.Series(cal_scores)

    auc = float(roc_auc_score(y_test, s_test))
    ks_stat, ks_pval = compute_ks_statistic(s_test, y_test)
    brier = float(brier_score_loss(y_test, s_test))
    psi_val = compute_psi(s_cal, s_test, n_bins=psi_bins)
    prob_true, prob_pred = compute_calibration_curve(y_test, s_test, n_bins=cal_bins)

    return ModelPerformanceMetrics(
        auc=auc,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        brier_score=brier,
        psi=psi_val,
        calibration_curve_true=prob_true,
        calibration_curve_pred=prob_pred,
        n_samples=len(s_test),
    )


def _render_report_download(
    dataset: AuditDataset,
    splits: DatasetSplits,
    perf: ModelPerformanceMetrics,
    config: AuditLensConfig,
    dataset_name: str,
    cap_res: Optional[Any],
) -> None:
    """Renders the self-contained HTML report download button in sidebar."""
    all_audits = audit_all_groups(dataset, splits, config=config)
    ctx = AuditReportContext(
        dataset_name=dataset_name,
        data_provenance=(
            "Derived from UCI Credit Card dataset (demo)"
            if "Demo" in dataset_name
            else "User Uploaded CSV"
        ),
        n_rows=dataset.n_rows,
        default_rate=float(dataset.outcome.mean()),
        config=config,
        perf_metrics=perf,
        audits_by_column=all_audits,
        capacity_res=cap_res,
    )
    report_html = render_html_report(ctx)
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download Audit Report (HTML)",
        data=report_html,
        file_name="auditlens_audit_report.html",
        mime="text/html",
        help="Export executive self-contained audit report.",
    )


def main() -> None:
    """Main application execution pipeline."""
    _init_page()
    config = _render_sidebar_settings()
    demo_choice, df_raw = _render_sidebar_data_loader()

    if df_raw is None:
        st.info(
            "👈 Please upload a CSV file or select a Demo Model from the sidebar to begin."
        )
        return

    try:
        selected_groups, group_tags = _configure_group_columns(df_raw)
        dataset, splits = _get_cached_split(
            df_raw,
            selected_groups,
            group_tags,
            config.random_seed,
            config.calibration_split_ratio,
            config.min_rows,
            config.small_group_threshold,
        )

        perf_metrics = _get_cached_perf(
            tuple(splits.cal_score.tolist()),
            tuple(splits.cal_outcome.tolist()),
            tuple(splits.test_score.tolist()),
            tuple(splits.test_outcome.tolist()),
            config.psi_bins,
            config.n_calibration_bins,
        )

        cap_res = None
        try:
            cap_res = search_capacity_constrained_alpha(
                splits.cal_score,
                splits.cal_outcome,
                splits.test_score,
                splits.test_outcome,
                config=config,
            )
        except CapacityError as exc:
            logger.warning(f"Capacity search error: {exc}")

        _render_report_download(
            dataset, splits, perf_metrics, config, demo_choice, cap_res
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "1. Upload & Quality",
                "2. Performance & Stability",
                "3. Reliability Audit",
                "4. Review Capacity & Cost",
            ]
        )

        with tab1:
            render_upload_tab(dataset)

        with tab2:
            render_performance_tab(perf_metrics, splits)

        with tab3:
            if not dataset.group_columns:
                st.info(
                    "Select at least one group column in the sidebar to run the reliability audit."
                )
            else:
                audit_col = st.selectbox(
                    "Select Group Column for Reliability Audit",
                    options=dataset.group_columns,
                )
                audit_summary = audit_group_column(
                    dataset, splits, audit_col, config=config
                )
                render_audit_tab(audit_summary)

        with tab4:
            if cap_res:
                render_capacity_tab(
                    cap_res, config.review_capacity, config.target_coverage
                )
            else:
                st.error(
                    "Capacity constraints cannot be satisfied with current parameters."
                )

    except AuditLensError as exc:
        st.error(f"❌ {exc.message}")
        st.info(f"💡 Suggestion: {exc.fix_hint}")
    except Exception as exc:
        logger.exception(f"Unexpected application error: {exc}")
        st.error(f"An unexpected error occurred: {exc}")


if __name__ == "__main__":
    main()
