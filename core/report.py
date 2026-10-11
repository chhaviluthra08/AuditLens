"""Report export module: generates self-contained executive HTML audit reports."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from jinja2 import Environment, select_autoescape

from core.audit import AuditReportSummary
from core.config import DEFAULT_CONFIG, AuditLensConfig
from core.decisions import CapacitySearchResult
from core.performance import ModelPerformanceMetrics


@dataclass
class AuditReportContext:
    """Consolidated audit report execution context."""

    dataset_name: str
    data_provenance: str
    n_rows: int
    default_rate: float
    config: AuditLensConfig
    perf_metrics: ModelPerformanceMetrics
    audits_by_column: Dict[str, AuditReportSummary]
    capacity_res: Optional[CapacitySearchResult] = None
    findings: List[str] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def generate_findings(
    audits_by_column: Dict[str, AuditReportSummary],
    config: Optional[AuditLensConfig] = None,
) -> List[str]:
    """Auto-generates structured audit findings from per-group audit results.

    Args:
        audits_by_column: Mapping of column name to AuditReportSummary.
        config: AuditLensConfig. Defaults to DEFAULT_CONFIG.

    Returns:
        List of plain-English finding statements.
    """
    cfg = config or DEFAULT_CONFIG
    findings: List[str] = []

    for col, summary in audits_by_column.items():
        for m in summary.group_metrics:
            if m.split_flagged:
                gap = cfg.target_coverage - m.split_coverage
                findings.append(
                    f"Subgroup '{m.group_value}' (column '{col}') is statistically "
                    f"under-covered: split coverage {m.split_coverage:.1%} vs "
                    f"target {cfg.target_coverage:.1%} (coverage gap: {gap:.1%}, "
                    f"Wilson upper bound: {m.split_wilson_upper:.1%})."
                )
            elif m.is_unreliable:
                findings.append(
                    f"Subgroup '{m.group_value}' (column '{col}') contains only "
                    f"{m.n_samples} observations (< {cfg.small_group_threshold} threshold). "
                    "Marked as unreliable (small sample size)."
                )

    if not findings:
        findings.append(
            f"All audited subgroups meet target coverage ({cfg.target_coverage:.0%}) "
            "within statistical tolerance. No under-coverage flags triggered."
        )

    return findings


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AuditLens — Credit Model Audit Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background: #f7fafc; padding: 2rem; margin: 0; line-height: 1.6; }
  .container { max-width: 1000px; margin: 0 auto; background: #ffffff; padding: 2.5rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
  h1 { color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.75rem; margin-top: 0; }
  h2 { color: #2b6cb0; margin-top: 2rem; border-bottom: 1px solid #edf2f7; padding-bottom: 0.5rem; }
  h3 { color: #4a5568; margin-top: 1.5rem; }
  table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.95rem; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #e2e8f0; }
  th { background: #edf2f7; color: #4a5568; font-weight: 600; }
  .flagged-row { background: #fff5f5; color: #c53030; font-weight: 600; }
  .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }
  .card { background: #edf2f7; padding: 1rem; border-radius: 6px; }
  .card-val { font-size: 1.5rem; font-weight: bold; color: #2b6cb0; }
  .card-lbl { font-size: 0.85rem; color: #718096; }
  .alert { padding: 1rem; border-radius: 6px; margin: 1rem 0; }
  .alert-warning { background: #fffaf0; border-left: 4px solid #dd6b20; color: #7b341e; }
  .alert-info { background: #ebf8ff; border-left: 4px solid #3182ce; color: #2b6cb0; }
  ul { padding-left: 1.5rem; }
  .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; font-size: 0.85rem; color: #a0aec0; text-align: center; }
</style>
</head>
<body>
<div class="container">
  <h1>🛡️ AuditLens — Model Reliability & Fair Lending Audit Report</h1>
  <p><strong>Generated At:</strong> {{ ctx.generated_at }} | <strong>Dataset:</strong> {{ ctx.dataset_name }}</p>

  <h2>1. Model Summary & Data Provenance</h2>
  <div class="card-grid">
    <div class="card"><div class="card-val">{{ "{:,}".format(ctx.n_rows) }}</div><div class="card-lbl">Total Audit Sample</div></div>
    <div class="card"><div class="card-val">{{ "{:.2%}".format(ctx.default_rate) }}</div><div class="card-lbl">Base Default Rate</div></div>
    <div class="card"><div class="card-val">{{ "{:.0%}".format(ctx.config.target_coverage) }}</div><div class="card-lbl">Target Coverage</div></div>
    <div class="card"><div class="card-val">{{ ctx.audits_by_column|length }}</div><div class="card-lbl">Audited Group Attributes</div></div>
  </div>
  <p><strong>Provenance:</strong> {{ ctx.data_provenance }}</p>

  <h2>2. Audit Configuration & Settings</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
    <tr><td>Target Coverage</td><td>{{ "{:.1%}".format(ctx.config.target_coverage) }}</td><td>1 - alpha nominal prediction set coverage requirement</td></tr>
    <tr><td>Cost Ratio (r)</td><td>{{ ctx.config.cost_ratio_r }}</td><td>Cost ratio: approving a defaulter vs refusing a repaid borrower</td></tr>
    <tr><td>Review Cost (delta)</td><td>{{ ctx.config.review_cost_delta }}</td><td>Unit operational cost per case deferred to human team</td></tr>
    <tr><td>Review Capacity</td><td>{{ "{:.0%}".format(ctx.config.review_capacity) }}</td><td>Maximum workflow percentage handled by human review team</td></tr>
    <tr><td>Small Group Threshold</td><td>{{ ctx.config.small_group_threshold }}</td><td>Sample size below which subgroups are labelled unreliable</td></tr>
    <tr><td>Coverage Tolerance</td><td>{{ "{:.1%}".format(ctx.config.coverage_tolerance) }}</td><td>Required minimum coverage deficit for flagging</td></tr>
  </table>

  <h2>3. Model Performance & Stability</h2>
  <div class="card-grid">
    <div class="card"><div class="card-val">{{ "{:.4f}".format(ctx.perf_metrics.auc) }}</div><div class="card-lbl">ROC AUC</div></div>
    <div class="card"><div class="card-val">{{ "{:.4f}".format(ctx.perf_metrics.ks_statistic) }}</div><div class="card-lbl">KS Statistic (p={{ "{:.2e}".format(ctx.perf_metrics.ks_pvalue) }})</div></div>
    <div class="card"><div class="card-val">{{ "{:.4f}".format(ctx.perf_metrics.brier_score) }}</div><div class="card-lbl">Brier Score</div></div>
    <div class="card"><div class="card-val">{{ "{:.4f}".format(ctx.perf_metrics.psi) }}</div><div class="card-lbl">Score PSI (Cal vs Test)</div></div>
  </div>

  <h2>4. Per-Group Reliability Audit</h2>
  {% for col_name, summary in ctx.audits_by_column.items() %}
  <h3>Attribute: {{ col_name }}</h3>
  <table>
    <thead>
      <tr>
        <th>Subgroup</th>
        <th>N</th>
        <th>Split Coverage</th>
        <th>Wilson CI (Bonferroni)</th>
        <th>Split Deferral</th>
        <th>Selective Risk</th>
        <th>Mondrian Coverage</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {% for m in summary.group_metrics %}
      <tr class="{{ 'flagged-row' if m.split_flagged else '' }}">
        <td>{{ m.group_value }}</td>
        <td>{{ "{:,}".format(m.n_samples) }}</td>
        <td>{{ "{:.2%}".format(m.split_coverage) }}</td>
        <td>[{{ "{:.1%}".format(m.split_wilson_lower) }}, {{ "{:.1%}".format(m.split_wilson_upper) }}]</td>
        <td>{{ "{:.2%}".format(m.split_deferral_rate) }}</td>
        <td>{{ "{:.2%}".format(m.split_selective_risk) if m.split_selective_risk is not none else "N/A" }}</td>
        <td>{{ "{:.2%}".format(m.mondrian_coverage) }}</td>
        <td>{{ m.status_label }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endfor %}

  <h2>5. Human Review Capacity & Expected Cost</h2>
  {% if ctx.capacity_res %}
  <div class="card-grid">
    <div class="card"><div class="card-val">{{ "{:.4f}".format(ctx.capacity_res.optimal_alpha) }}</div><div class="card-lbl">Optimal Alpha</div></div>
    <div class="card"><div class="card-val">{{ "{:.2%}".format(ctx.capacity_res.implied_target_coverage) }}</div><div class="card-lbl">Implied Coverage</div></div>
    <div class="card"><div class="card-val">{{ "{:.2%}".format(ctx.capacity_res.test_deferral_rate) }}</div><div class="card-lbl">Test Deferral Rate</div></div>
    <div class="card"><div class="card-val">{{ "{:.3f}".format(ctx.capacity_res.test_results.expected_cost_j) }}</div><div class="card-lbl">Expected Cost (J)</div></div>
  </div>
  {% if ctx.capacity_res.below_target_coverage %}
  <div class="alert alert-warning">
    <strong>Capacity Constraint Notice:</strong> Human review capacity constrained coverage to {{ "{:.1%}".format(ctx.capacity_res.implied_target_coverage) }}, below the {{ "{:.0%}".format(ctx.config.target_coverage) }} target.
  </div>
  {% endif %}
  {% else %}
  <p>Capacity simulation not computed.</p>
  {% endif %}

  <h2>6. Key Findings</h2>
  <ul>
    {% for finding in ctx.findings %}
    <li>{{ finding }}</li>
    {% endfor %}
  </ul>

  <h2>7. Regulatory Topic Mapping</h2>
  <div class="alert alert-info">
    <strong>Indicative mapping only; verify against the final RBI guidance.</strong>
  </div>
  <table>
    <thead><tr><th>Regulatory Topic</th><th>AuditLens Coverage & Evidence</th></tr></thead>
    <tbody>
      <tr><td>Independent Validation</td><td>Independent 50/50 calibration and test partitioning with pure nonconformity quantile evaluation.</td></tr>
      <tr><td>Ongoing Monitoring</td><td>Population Stability Index (PSI) score drift tracking and distribution monitoring.</td></tr>
      <tr><td>Explainability</td><td>Empirical calibration curves and probability binning to explain model uncertainty.</td></tr>
      <tr><td>Bias and Fairness</td><td>Per-group split conformal vs Mondrian coverage auditing, identifying protected demographic under-coverage.</td></tr>
      <tr><td>Model Inventory</td><td>Automated audit documentation capturing dataset provenance, parameter configurations, and evaluation metrics.</td></tr>
    </tbody>
  </table>

  <h2>8. Scope & Limitations</h2>
  <ul>
    <li>The audit covers strictly the uploaded sample observations and does not guarantee future operational performance.</li>
    <li>Mondrian conformal prediction is evaluated one group column at a time, not for cross-group intersections.</li>
    <li>The flagging rule employs dynamic Bonferroni multiplicity adjustment and a minimum coverage gap threshold.</li>
    <li>Expected cost analysis assumes human reviewers resolve deferred cases correctly without residual error.</li>
    <li>Demo data is derived from the 2005 Taiwan UCI Credit Card dataset.</li>
    <li>The audit assumes exchangeability between calibration and test partitions and does not verify whether evaluation data is strictly out-of-time.</li>
  </ul>

  <div class="footer">
    AuditLens — Confidential Model Risk Governance Report &copy; {{ ctx.generated_at[:4] }}
  </div>
</div>
</body>
</html>
"""


def render_html_report(ctx: AuditReportContext) -> str:
    """Renders the HTML audit report using Jinja2 with autoescaping enabled.

    Args:
        ctx: AuditReportContext with report data.

    Returns:
        Self-contained HTML string.
    """
    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(HTML_TEMPLATE)
    findings = ctx.findings or generate_findings(ctx.audits_by_column, ctx.config)
    return template.render(ctx=ctx, findings=findings)
