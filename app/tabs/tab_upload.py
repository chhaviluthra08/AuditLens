"""Tab 1: Upload, data quality summary, and schema inspection."""

from typing import List

import streamlit as st

from core.validation import AuditDataset, ValidationWarning


def _render_summary_metrics(dataset: AuditDataset) -> None:
    """Renders high-level data quality metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", f"{dataset.n_rows:,}")
    with col2:
        default_rate = float(dataset.outcome.mean())
        st.metric("Default Rate", f"{default_rate:.2%}")
    with col3:
        st.metric("Identified Groups", f"{len(dataset.group_columns)}")
    with col4:
        st.metric("Quality Warnings", f"{len(dataset.warnings)}")


def _render_warnings_expander(warnings: List[ValidationWarning]) -> None:
    """Displays data quality warnings inside an expander."""
    if not warnings:
        st.success("No data quality warnings detected. Input contract fully satisfied.")
        return

    with st.expander("Data Quality Warnings", expanded=True):
        for w in warnings:
            st.warning(f"⚠️ {w.message}")


def render_upload_tab(dataset: AuditDataset) -> None:
    """Renders data overview, summary statistics, and sample observations."""
    st.subheader("Data Quality & Summary Overview")
    _render_summary_metrics(dataset)

    _render_warnings_expander(dataset.warnings)

    st.markdown("#### Sample Dataset Preview")
    preview_cols = ["score", "outcome"] + [
        c for c in dataset.group_columns if c not in ("score", "outcome")
    ]
    if dataset.timestamp_column:
        preview_cols.append(dataset.timestamp_column)

    st.dataframe(
        dataset.data[preview_cols].head(15),
        use_container_width=True,
    )
