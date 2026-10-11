"""Tab 4: Review capacity constraint simulation and cost analysis."""

from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

from core.decisions import CapacitySearchResult


def _render_capacity_metrics(res: CapacitySearchResult, target_cov: float) -> None:
    """Renders capacity optimization results across metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Chosen Alpha", f"{res.optimal_alpha:.4f}")
    with col2:
        st.metric("Implied Coverage", f"{res.implied_target_coverage:.2%}")
    with col3:
        st.metric("Cal Deferral", f"{res.cal_deferral_rate:.2%}")
    with col4:
        st.metric("Test Deferral", f"{res.test_deferral_rate:.2%}")
    with col5:
        st.metric("Expected Cost (J)", f"{res.test_results.expected_cost_j:.3f}")

    if res.below_target_coverage:
        st.warning(
            f"⚠️ **Capacity Constraint Warning**: To respect human review capacity, "
            f"coverage was reduced to **{res.implied_target_coverage:.1%}** (below the "
            f"target of {target_cov:.1%}). Consider increasing reviewer capacity."
        )


def _plot_capacity_curve(
    curve_data: List[Dict[str, Any]],
    review_capacity: float,
    optimal_alpha: float,
) -> go.Figure:
    """Plots trade-off curve between coverage and calibration deferral rate."""
    coverages = [d["target_coverage"] for d in curve_data]
    def_rates = [d["cal_deferral_rate"] for d in curve_data]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=coverages,
            y=def_rates,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            name="Calibration Deferral Rate",
        )
    )
    fig.add_hline(
        y=review_capacity,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max Capacity ({review_capacity:.0%})",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Human Review Deferral Rate vs Target Coverage",
        xaxis_title="Target Coverage (1 - alpha)",
        yaxis_title="Estimated Deferral Rate",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def render_capacity_tab(
    res: CapacitySearchResult,
    review_capacity: float,
    target_coverage: float,
) -> None:
    """Renders capacity search results, warnings, and optimization curve."""
    st.subheader("Human Review Capacity & Workflow Simulation")
    _render_capacity_metrics(res, target_coverage)

    st.plotly_chart(
        _plot_capacity_curve(res.grid_curve, review_capacity, res.optimal_alpha),
        use_container_width=True,
    )

    st.markdown("#### Decision Breakdown (Test Split)")
    d_res = res.test_results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(
            f"- **Approved Cases**: {d_res.n_approved:,} ({d_res.n_approved/d_res.n_total:.1%})"
        )
    with col2:
        st.write(
            f"- **Refused Cases**: {d_res.n_refused:,} ({d_res.n_refused/d_res.n_total:.1%})"
        )
    with col3:
        st.write(
            f"- **Deferred to Humans**: {d_res.n_deferred:,} ({d_res.deferral_rate:.1%})"
        )
