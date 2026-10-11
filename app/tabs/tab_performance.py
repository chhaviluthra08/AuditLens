"""Tab 2: Model performance, calibration, and score stability."""

import plotly.graph_objects as go
import streamlit as st

from core.performance import ModelPerformanceMetrics
from core.splits import DatasetSplits


def _render_metric_cards(metrics: ModelPerformanceMetrics) -> None:
    """Renders key discrimination, calibration, and stability metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC AUC", f"{metrics.auc:.4f}")
    with col2:
        st.metric(
            "KS Statistic",
            f"{metrics.ks_statistic:.4f}",
            help=f"p-value: {metrics.ks_pvalue:.2e}",
        )
    with col3:
        st.metric("Brier Score", f"{metrics.brier_score:.4f}")
    with col4:
        st.metric(
            "Score PSI",
            f"{metrics.psi:.4f}",
            help="Calibration vs Test score distribution drift",
        )


def _plot_calibration_curve(metrics: ModelPerformanceMetrics) -> go.Figure:
    """Creates Plotly figure for empirical calibration curve vs ideal reference."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect Calibration",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.calibration_curve_pred,
            y=metrics.calibration_curve_true,
            mode="lines+markers",
            marker=dict(size=8, color="#1f77b4"),
            line=dict(color="#1f77b4", width=2),
            name="Empirical Reliability",
        )
    )
    fig.update_layout(
        title="Calibration Curve (Test Split)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Default Fraction",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def _plot_score_distributions(splits: DatasetSplits) -> go.Figure:
    """Creates Plotly figure showing score distributions split by outcome."""
    y = splits.test_outcome.to_numpy()
    scores = splits.test_score.to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores[y == 0],
            name="Repaid (0)",
            opacity=0.6,
            marker_color="#2ca02c",
            nbinsx=30,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=scores[y == 1],
            name="Defaulted (1)",
            opacity=0.6,
            marker_color="#d62728",
            nbinsx=30,
        )
    )
    fig.update_layout(
        title="Test Score Distribution by Actual Outcome",
        barmode="overlay",
        xaxis_title="Predicted Probability of Default",
        yaxis_title="Count",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def render_performance_tab(
    metrics: ModelPerformanceMetrics,
    splits: DatasetSplits,
) -> None:
    """Renders performance metric summary cards and diagnostic plots."""
    st.subheader("Model Performance & Distribution Stability")
    _render_metric_cards(metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_plot_calibration_curve(metrics), use_container_width=True)
    with col2:
        st.plotly_chart(_plot_score_distributions(splits), use_container_width=True)
