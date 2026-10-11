"""Tab 3: Per-group reliability audit, coverage dot plot, and side-by-side comparison."""

from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.audit import AuditReportSummary, GroupAuditMetrics


def plot_coverage_dot_plot(summary: AuditReportSummary) -> go.Figure:
    """Generates Plotly dot plot comparing Split vs Mondrian coverage with Wilson CIs."""
    fig = go.Figure()
    groups = [m.group_value for m in summary.group_metrics]
    split_covs = [m.split_coverage for m in summary.group_metrics]
    mond_covs = [m.mondrian_coverage for m in summary.group_metrics]

    # Asymmetric error bars for Wilson intervals
    err_plus = [m.split_wilson_upper - m.split_coverage for m in summary.group_metrics]
    err_minus = [m.split_coverage - m.split_wilson_lower for m in summary.group_metrics]

    colors = [
        "#d62728" if m.split_flagged else ("#ff7f0e" if m.is_unreliable else "#1f77b4")
        for m in summary.group_metrics
    ]

    fig.add_trace(
        go.Scatter(
            x=split_covs,
            y=groups,
            mode="markers",
            marker=dict(size=11, color=colors),
            error_x=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                color="rgba(100, 100, 100, 0.6)",
                thickness=2,
                width=5,
            ),
            name="Split Conformal (with 95% CI)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mond_covs,
            y=groups,
            mode="markers",
            marker=dict(size=9, symbol="diamond", color="#2ca02c"),
            name="Mondrian (Group-Aware)",
        )
    )

    fig.add_vline(
        x=summary.target_coverage,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target ({summary.target_coverage:.0%})",
        annotation_position="top left",
    )

    wilson_lowers = [m.split_wilson_lower for m in summary.group_metrics]
    min_lower = min(wilson_lowers) if wilson_lowers else 0.70
    margin = 0.05
    x_min = max(0.0, min_lower - margin)

    fig.update_layout(
        title=f"Coverage per Subgroup: '{summary.group_column}'",
        xaxis_title="Empirical Coverage",
        yaxis_title="Group Category",
        height=max(350, len(groups) * 45 + 100),
        xaxis=dict(range=[x_min, 1.0]),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


_plot_coverage_dot_plot = plot_coverage_dot_plot


def _create_metrics_table(metrics: List[GroupAuditMetrics]) -> pd.DataFrame:
    """Formats group audit metrics for tabular display."""
    rows = []
    for m in metrics:
        risk_str = (
            f"{m.split_selective_risk:.2%}"
            if m.split_selective_risk is not None
            else "N/A"
        )
        rows.append(
            {
                "Subgroup": m.group_value,
                "Sample Size (n)": f"{m.n_samples:,}",
                "Split Coverage": f"{m.split_coverage:.2%}",
                "Wilson 95% CI": f"[{m.split_wilson_lower:.1%}, {m.split_wilson_upper:.1%}]",
                "Split Deferral": f"{m.split_deferral_rate:.2%}",
                "Selective Risk": risk_str,
                "Mondrian Coverage": f"{m.mondrian_coverage:.2%}",
                "Audit Status": m.status_label,
            }
        )
    return pd.DataFrame(rows)


def render_audit_tab(summary: AuditReportSummary) -> None:
    """Renders reliability audit overview, dot plot, and detailed group table."""
    st.subheader(f"Reliability & Fairness Audit: {summary.group_column}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Coverage", f"{summary.target_coverage:.1%}")
    with col2:
        st.metric("Overall Split Coverage", f"{summary.overall_split_coverage:.2%}")
    with col3:
        flag_cnt = len(summary.flagged_groups)
        st.metric(
            "Flagged Subgroups",
            f"{flag_cnt}",
            delta="Under-covered" if flag_cnt > 0 else "All Reliable",
            delta_color="inverse" if flag_cnt > 0 else "normal",
        )

    if summary.flagged_groups:
        st.error(
            f"🚨 **Statistically clear under-coverage detected in**: {', '.join(summary.flagged_groups)}. "
            f"Wilson 95% upper bound falls strictly below the {summary.target_coverage:.0%} target."
        )

    st.plotly_chart(_plot_coverage_dot_plot(summary), use_container_width=True)

    st.markdown("#### Subgroup Comparison Table")
    df_table = _create_metrics_table(summary.group_metrics)
    st.dataframe(df_table, use_container_width=True)
