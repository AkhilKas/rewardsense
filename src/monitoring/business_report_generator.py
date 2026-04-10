"""
On-Demand Business Report Generator.

Compiles latency, cost, and business usage metrics into HTML and PDF
reports.  This is an internal/business artifact kept outside the user
product UI.

Report contents (per spec):
- Total requests
- Average / p95 latency
- Explanation latency split
- Estimated LLM cost by period
- Fallback / error rates
- Top-used recommendation flows

Usage::

    from src.monitoring.business_metrics import BusinessMetricsCollector
    from src.monitoring.business_report_generator import BusinessReportGenerator

    collector = BusinessMetricsCollector()
    metrics = collector.collect(days=7)
    generator = BusinessReportGenerator()
    generator.generate_html(metrics, "reports/business_report.html")
    generator.generate_pdf(metrics, "reports/business_report.pdf")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Import from sibling module
from src.monitoring.business_metrics import PeriodMetrics

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# =====================================================================
# Chart generation (embedded base64 PNGs for HTML)
# =====================================================================


def _generate_daily_requests_chart(metrics: PeriodMetrics) -> Optional[str]:
    """Generate a daily request count bar chart as base64 PNG."""
    if not MATPLOTLIB_AVAILABLE or not metrics.daily_breakdown:
        return None

    import base64
    import io

    dates = [d.date for d in metrics.daily_breakdown]
    counts = [d.request_count for d in metrics.daily_breakdown]

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_colors = ["#3b82f6" if c > 0 else "#e5e7eb" for c in counts]
    ax.bar(
        range(len(dates)), counts, color=bar_colors, edgecolor="#1e40af", linewidth=0.5
    )
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(
        [d.split("-", 1)[-1] for d in dates], rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Requests")
    ax.set_title("Daily Request Volume")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _generate_latency_trend_chart(metrics: PeriodMetrics) -> Optional[str]:
    """Generate a daily latency trend line chart as base64 PNG."""
    if not MATPLOTLIB_AVAILABLE or not metrics.daily_breakdown:
        return None

    import base64
    import io

    dates = [d.date for d in metrics.daily_breakdown]
    avg_lat = [d.latency.mean_ms for d in metrics.daily_breakdown]
    p95_lat = [d.latency.p95_ms for d in metrics.daily_breakdown]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        range(len(dates)),
        avg_lat,
        marker="o",
        label="Avg",
        color="#3b82f6",
        linewidth=2,
    )
    ax.plot(
        range(len(dates)),
        p95_lat,
        marker="s",
        label="p95",
        color="#ef4444",
        linewidth=2,
    )
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(
        [d.split("-", 1)[-1] for d in dates], rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Daily Latency Trend")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _generate_flow_pie_chart(metrics: PeriodMetrics) -> Optional[str]:
    """Generate a recommendation flow distribution pie chart."""
    if not MATPLOTLIB_AVAILABLE or not metrics.overall_flow_counts:
        return None

    import base64
    import io

    labels = list(metrics.overall_flow_counts.keys())
    sizes = list(metrics.overall_flow_counts.values())
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=palette[: len(labels)],
        startangle=90,
    )
    ax.set_title("Recommendation Flows")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =====================================================================
# HTML report
# =====================================================================

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RewardSense Business Metrics Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         color: #1e293b; background: #f8fafc; padding: 2rem; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1.25rem;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase;
                  letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }}
  .card .detail {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; }}
  .section {{ background: #fff; border-radius: 8px; padding: 1.5rem;
              box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }}
  .section h2 {{ font-size: 1.1rem; margin-bottom: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 0.5rem; border-bottom: 2px solid #e2e8f0;
        color: #64748b; font-weight: 600; }}
  td {{ padding: 0.5rem; border-bottom: 1px solid #f1f5f9; }}
  .chart-img {{ max-width: 100%; height: auto; margin: 0.5rem 0; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; }}
  .badge-green {{ background: #dcfce7; color: #166534; }}
  .badge-red {{ background: #fef2f2; color: #991b1b; }}
  .badge-yellow {{ background: #fefce8; color: #854d0e; }}
  .footer {{ text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 2rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>RewardSense Business Metrics</h1>
  <p class="subtitle">{start_date} to {end_date} &middot; Generated {generated_at}</p>

  <div class="cards">
    <div class="card">
      <div class="label">Total Requests</div>
      <div class="value">{total_requests:,}</div>
      <div class="detail">{avg_daily_requests:.0f} avg/day</div>
    </div>
    <div class="card">
      <div class="label">Avg Latency</div>
      <div class="value">{avg_latency_ms:.1f} ms</div>
      <div class="detail">p95: {p95_latency_ms:.1f} ms</div>
    </div>
    <div class="card">
      <div class="label">Error Rate</div>
      <div class="value">{error_rate_pct:.2f}%</div>
      <div class="detail">{total_errors:,} errors</div>
    </div>
    <div class="card">
      <div class="label">LLM Calls</div>
      <div class="value">{llm_total_calls:,}</div>
      <div class="detail">{llm_success_rate_pct:.1f}% success</div>
    </div>
    <div class="card">
      <div class="label">Est. LLM Cost</div>
      <div class="value">${llm_cost_usd:.2f}</div>
      <div class="detail">~{llm_tokens:,} tokens</div>
    </div>
    <div class="card">
      <div class="label">Personalization</div>
      <div class="value">{personalization_rate_pct:.1f}%</div>
      <div class="detail">of requests personalized</div>
    </div>
  </div>

  {daily_requests_chart}

  {latency_trend_chart}

  <div class="section">
    <h2>Latency Breakdown by Stage</h2>
    <table>
      <tr><th>Stage</th><th>Avg (ms)</th></tr>
      <tr><td>Normalize</td><td>{stage_normalize:.2f}</td></tr>
      <tr><td>Deterministic Scoring</td><td>{stage_deterministic:.2f}</td></tr>
      <tr><td>Personalization</td><td>{stage_personalization:.2f}</td></tr>
      <tr><td>Ranking</td><td>{stage_rank:.2f}</td></tr>
      <tr><td>LLM Explanation</td><td>{stage_llm:.2f}</td></tr>
    </table>
  </div>

  {flow_chart}

  <div class="section">
    <h2>Recommendation Flows</h2>
    <table>
      <tr><th>Flow</th><th>Requests</th><th>Share</th></tr>
      {flow_rows}
    </table>
  </div>

  <div class="section">
    <h2>Top Recommended Cards</h2>
    <table>
      <tr><th>Card</th><th>Times Recommended</th></tr>
      {card_rows}
    </table>
  </div>

  <div class="section">
    <h2>LLM Telemetry</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Total LLM Calls</td><td>{llm_total_calls:,}</td></tr>
      <tr><td>Successes</td><td>{llm_successes:,}</td></tr>
      <tr><td>Fallbacks</td><td>{llm_fallbacks:,}</td></tr>
      <tr><td>Success Rate</td><td>{llm_success_rate_pct:.1f}%</td></tr>
      <tr><td>Avg Explanation Latency</td><td>{llm_avg_expl_ms:.1f} ms</td></tr>
      <tr><td>Est. Tokens</td><td>{llm_tokens:,}</td></tr>
      <tr><td>Est. Cost</td><td>${llm_cost_usd:.2f}</td></tr>
      <tr><td>Models Used</td><td>{llm_models}</td></tr>
      <tr><td>Prompt Versions</td><td>{llm_prompt_versions}</td></tr>
    </table>
  </div>

  <div class="section">
    <h2>Daily Breakdown</h2>
    <table>
      <tr><th>Date</th><th>Requests</th><th>Errors</th><th>Avg Latency</th><th>p95 Latency</th><th>LLM Calls</th></tr>
      {daily_rows}
    </table>
  </div>

  <p class="footer">RewardSense Business Metrics Report &middot; Internal Use Only</p>
</div>
</body>
</html>
"""


class BusinessReportGenerator:
    """Generate HTML and PDF business metric reports."""

    def _build_template_vars(self, metrics: PeriodMetrics) -> Dict[str, Any]:
        """Build the template variable dict from PeriodMetrics."""
        num_days = max(len(metrics.daily_breakdown), 1)
        total = metrics.total_requests

        # Flow rows
        flow_rows = ""
        for flow, count in sorted(
            metrics.overall_flow_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            share = (count / total * 100) if total > 0 else 0
            flow_rows += f"      <tr><td>{flow}</td><td>{count:,}</td><td>{share:.1f}%</td></tr>\n"
        if not flow_rows:
            flow_rows = "      <tr><td colspan='3'>No data</td></tr>\n"

        # Card rows
        sorted_cards = sorted(
            metrics.overall_top_cards.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        card_rows = ""
        for card, count in sorted_cards:
            card_rows += f"      <tr><td>{card}</td><td>{count:,}</td></tr>\n"
        if not card_rows:
            card_rows = "      <tr><td colspan='2'>No data</td></tr>\n"

        # Daily rows
        daily_rows = ""
        for day in metrics.daily_breakdown:
            daily_rows += (
                f"      <tr>"
                f"<td>{day.date}</td>"
                f"<td>{day.request_count:,}</td>"
                f"<td>{day.error_count}</td>"
                f"<td>{day.latency.mean_ms:.1f} ms</td>"
                f"<td>{day.latency.p95_ms:.1f} ms</td>"
                f"<td>{day.llm.total_calls}</td>"
                f"</tr>\n"
            )

        # Charts
        requests_chart = _generate_daily_requests_chart(metrics)
        latency_chart = _generate_latency_trend_chart(metrics)
        flow_chart = _generate_flow_pie_chart(metrics)

        def _chart_html(b64: Optional[str], alt: str) -> str:
            if b64:
                return (
                    f'<div class="section">'
                    f'<img class="chart-img" src="data:image/png;base64,{b64}" alt="{alt}">'
                    f"</div>"
                )
            return ""

        llm = metrics.overall_llm
        stage = metrics.overall_stage_latency

        return {
            "start_date": metrics.start_date,
            "end_date": metrics.end_date,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "total_requests": total,
            "avg_daily_requests": total / num_days,
            "avg_latency_ms": metrics.overall_latency.mean_ms,
            "p95_latency_ms": metrics.overall_latency.p95_ms,
            "error_rate_pct": metrics.overall_error_rate * 100,
            "total_errors": metrics.total_errors,
            "llm_total_calls": llm.total_calls,
            "llm_successes": llm.total_successes,
            "llm_fallbacks": llm.total_fallbacks,
            "llm_success_rate_pct": llm.success_rate * 100,
            "llm_avg_expl_ms": llm.avg_explanation_latency_ms,
            "llm_tokens": llm.estimated_tokens,
            "llm_cost_usd": llm.estimated_cost_usd,
            "llm_models": ", ".join(llm.model_names_used) or "N/A",
            "llm_prompt_versions": ", ".join(llm.prompt_versions_used) or "N/A",
            "personalization_rate_pct": metrics.overall_personalization_rate * 100,
            "stage_normalize": (
                stage.normalize_ms if hasattr(stage, "normalize_ms") else 0.0
            ),
            "stage_deterministic": (
                stage.deterministic_ms if hasattr(stage, "deterministic_ms") else 0.0
            ),
            "stage_personalization": (
                stage.personalization_ms
                if hasattr(stage, "personalization_ms")
                else 0.0
            ),
            "stage_rank": stage.rank_ms if hasattr(stage, "rank_ms") else 0.0,
            "stage_llm": (
                stage.llm_explanation_ms
                if hasattr(stage, "llm_explanation_ms")
                else 0.0
            ),
            "daily_requests_chart": _chart_html(requests_chart, "Daily Requests"),
            "latency_trend_chart": _chart_html(latency_chart, "Latency Trend"),
            "flow_chart": _chart_html(flow_chart, "Flow Distribution"),
            "flow_rows": flow_rows,
            "card_rows": card_rows,
            "daily_rows": daily_rows,
        }

    # -----------------------------------------------------------------
    # HTML
    # -----------------------------------------------------------------

    def generate_html(
        self,
        metrics: PeriodMetrics,
        output_path: str,
    ) -> Path:
        """Generate an HTML business report.

        Parameters
        ----------
        metrics : PeriodMetrics
            Aggregated metrics from ``BusinessMetricsCollector``.
        output_path : str
            File path for the output HTML file.

        Returns
        -------
        Path
            The path to the generated report.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        template_vars = self._build_template_vars(metrics)
        html = _HTML_TEMPLATE.format(**template_vars)
        path.write_text(html, encoding="utf-8")

        logger.info("HTML business report written to %s", path)
        return path

    # -----------------------------------------------------------------
    # PDF
    # -----------------------------------------------------------------

    def generate_pdf(
        self,
        metrics: PeriodMetrics,
        output_path: str,
    ) -> Optional[Path]:
        """Generate a PDF business report using ReportLab.

        Returns None if ReportLab is not installed.
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning(
                "reportlab not installed, skipping PDF generation. "
                "Install with: pip install reportlab"
            )
            return None

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=18,
            spaceAfter=6,
        )
        story.append(Paragraph("RewardSense Business Metrics Report", title_style))
        story.append(
            Paragraph(
                f"{metrics.start_date} to {metrics.end_date}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 18))

        # Summary table
        num_days = max(len(metrics.daily_breakdown), 1)
        llm = metrics.overall_llm
        summary_data = [
            ["Metric", "Value"],
            ["Total Requests", f"{metrics.total_requests:,}"],
            ["Avg Requests/Day", f"{metrics.total_requests / num_days:.0f}"],
            ["Avg Latency", f"{metrics.overall_latency.mean_ms:.1f} ms"],
            ["p95 Latency", f"{metrics.overall_latency.p95_ms:.1f} ms"],
            ["Error Rate", f"{metrics.overall_error_rate * 100:.2f}%"],
            ["LLM Calls", f"{llm.total_calls:,}"],
            ["LLM Success Rate", f"{llm.success_rate * 100:.1f}%"],
            ["Est. LLM Cost", f"${llm.estimated_cost_usd:.2f}"],
            [
                "Personalization Rate",
                f"{metrics.overall_personalization_rate * 100:.1f}%",
            ],
        ]
        summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8fafc")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(Paragraph("Overview", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(summary_table)
        story.append(Spacer(1, 18))

        # Stage latency table
        stage = metrics.overall_stage_latency
        stage_data = [
            ["Stage", "Avg (ms)"],
            ["Normalize", f"{stage.normalize_ms:.2f}"],
            ["Deterministic Scoring", f"{stage.deterministic_ms:.2f}"],
            ["Personalization", f"{stage.personalization_ms:.2f}"],
            ["Ranking", f"{stage.rank_ms:.2f}"],
            ["LLM Explanation", f"{stage.llm_explanation_ms:.2f}"],
        ]
        stage_table = Table(stage_data, colWidths=[3 * inch, 3 * inch])
        stage_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(Paragraph("Latency Breakdown by Stage", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(stage_table)
        story.append(Spacer(1, 18))

        # Flow distribution table
        if metrics.overall_flow_counts:
            flow_data = [["Flow", "Requests", "Share"]]
            for flow, count in sorted(
                metrics.overall_flow_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                share = (
                    (count / metrics.total_requests * 100)
                    if metrics.total_requests > 0
                    else 0
                )
                flow_data.append([flow, f"{count:,}", f"{share:.1f}%"])
            flow_table = Table(flow_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
            flow_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(Paragraph("Recommendation Flows", styles["Heading2"]))
            story.append(Spacer(1, 6))
            story.append(flow_table)
            story.append(Spacer(1, 18))

        # Top cards table
        sorted_cards = sorted(
            metrics.overall_top_cards.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        if sorted_cards:
            card_data = [["Card", "Times Recommended"]]
            for card, count in sorted_cards:
                card_data.append([card, f"{count:,}"])
            card_table = Table(card_data, colWidths=[4 * inch, 2 * inch])
            card_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(Paragraph("Top Recommended Cards", styles["Heading2"]))
            story.append(Spacer(1, 6))
            story.append(card_table)
            story.append(Spacer(1, 18))

        # Daily breakdown table
        if metrics.daily_breakdown:
            daily_data = [["Date", "Requests", "Errors", "Avg Lat.", "p95 Lat.", "LLM"]]
            for day in metrics.daily_breakdown:
                daily_data.append(
                    [
                        day.date,
                        f"{day.request_count:,}",
                        str(day.error_count),
                        f"{day.latency.mean_ms:.1f}",
                        f"{day.latency.p95_ms:.1f}",
                        str(day.llm.total_calls),
                    ]
                )
            daily_table = Table(daily_data, colWidths=[1.2 * inch] * 6)
            daily_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f8fafc")],
                        ),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(Paragraph("Daily Breakdown", styles["Heading2"]))
            story.append(Spacer(1, 6))
            story.append(daily_table)

        # Footer
        story.append(Spacer(1, 24))
        footer_style = ParagraphStyle(
            "Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey
        )
        story.append(
            Paragraph(
                f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
                f"| RewardSense Business Metrics | Internal Use Only",
                footer_style,
            )
        )

        doc.build(story)
        logger.info("PDF business report written to %s", path)
        return path

    # -----------------------------------------------------------------
    # JSON (for programmatic consumption)
    # -----------------------------------------------------------------

    def generate_json(
        self,
        metrics: PeriodMetrics,
        output_path: str,
    ) -> Path:
        """Export raw metrics as JSON for programmatic consumption."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(metrics.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("JSON business report written to %s", path)
        return path
