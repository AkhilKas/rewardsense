#!/usr/bin/env python3
"""
Generate Business Metrics Report.

On-demand script that reads inference logs, aggregates business
metrics, and produces HTML and/or PDF reports.

Usage::

    # 7-day HTML report (default)
    PYTHONPATH=. python scripts/generate_business_report.py

    # 30-day HTML + PDF report
    PYTHONPATH=. python scripts/generate_business_report.py \\
        --days 30 --format html pdf

    # Custom date range and output directory
    PYTHONPATH=. python scripts/generate_business_report.py \\
        --start-date 2026-03-01 --end-date 2026-03-31 \\
        --output-dir reports/march

    # JSON export only (for programmatic use)
    PYTHONPATH=. python scripts/generate_business_report.py --format json

    # Use a specific local log directory
    PYTHONPATH=. python scripts/generate_business_report.py \\
        --log-dir /tmp/rewardsense-inference-logs
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_business_report")


def _parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RewardSense business metrics reports.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to include (default: 7). Ignored if --start-date is set.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides --days.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["html", "pdf", "json"],
        default=["html"],
        help="Output formats (default: html). Can specify multiple.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output files (default: reports/).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Local inference log directory (overrides env).",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help="GCS bucket for inference logs (overrides env).",
    )
    parser.add_argument(
        "--cost-per-1k-tokens",
        type=float,
        default=None,
        help="LLM cost per 1K tokens for estimates (default: $0.005).",
    )
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = _parse_args(argv)

    from src.monitoring.business_metrics import BusinessMetricsCollector
    from src.monitoring.business_report_generator import BusinessReportGenerator

    # Resolve date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    else:
        end_date = datetime.now(timezone.utc)

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        days = (end_date - start_date).days + 1
    else:
        days = args.days

    logger.info(
        "Collecting business metrics for %d days ending %s",
        days,
        end_date.strftime("%Y-%m-%d"),
    )

    # Collect metrics
    collector = BusinessMetricsCollector(
        local_log_dir=args.log_dir,
        gcs_bucket=args.gcs_bucket,
        cost_per_1k_tokens=args.cost_per_1k_tokens,
    )
    metrics = collector.collect(days=days, end_date=end_date)

    if metrics.total_requests == 0:
        logger.warning(
            "No inference logs found for the period %s to %s. "
            "The report will be generated with empty data.",
            metrics.start_date,
            metrics.end_date,
        )

    # Generate reports
    generator = BusinessReportGenerator()
    output_dir = Path(args.output_dir)
    date_suffix = f"{metrics.start_date}_to_{metrics.end_date}"
    generated: list[str] = []

    if "html" in args.format:
        html_path = output_dir / f"business_report_{date_suffix}.html"
        generator.generate_html(metrics, str(html_path))
        generated.append(str(html_path))
        logger.info("HTML report: %s", html_path)

    if "pdf" in args.format:
        pdf_path = output_dir / f"business_report_{date_suffix}.pdf"
        result = generator.generate_pdf(metrics, str(pdf_path))
        if result:
            generated.append(str(result))
            logger.info("PDF report: %s", result)
        else:
            logger.warning("PDF generation skipped (reportlab not installed).")

    if "json" in args.format:
        json_path = output_dir / f"business_report_{date_suffix}.json"
        generator.generate_json(metrics, str(json_path))
        generated.append(str(json_path))
        logger.info("JSON report: %s", json_path)

    # Summary
    logger.info("--- Report Summary ---")
    logger.info("Period: %s to %s", metrics.start_date, metrics.end_date)
    logger.info("Total requests: %d", metrics.total_requests)
    logger.info("Error rate: %.2f%%", metrics.overall_error_rate * 100)
    logger.info(
        "Avg latency: %.1f ms (p95: %.1f ms)",
        metrics.overall_latency.mean_ms,
        metrics.overall_latency.p95_ms,
    )
    logger.info(
        "LLM calls: %d (%.1f%% success, est. $%.2f)",
        metrics.overall_llm.total_calls,
        metrics.overall_llm.success_rate * 100,
        metrics.overall_llm.estimated_cost_usd,
    )
    logger.info("Generated %d report(s): %s", len(generated), ", ".join(generated))

    return 0


if __name__ == "__main__":
    sys.exit(main())
