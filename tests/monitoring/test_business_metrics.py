"""
Tests for Business Monitoring & Reporting.

Coverage:
- BusinessMetricsCollector: record loading, daily aggregation, period
  aggregation, LLM telemetry, edge cases
- BusinessReportGenerator: HTML output, PDF output (if reportlab
  available), JSON output, empty data handling
- CLI script: argument parsing, end-to-end with synthetic logs

Run::

    PYTHONPATH=. pytest tests/monitoring/test_business_metrics.py -v -o "addopts="
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.monitoring.business_metrics import (
    BusinessMetricsCollector,
    LatencyStats,
    PeriodMetrics,
)
from src.monitoring.business_report_generator import (
    BusinessReportGenerator,
    REPORTLAB_AVAILABLE,
)


# =====================================================================
# Fixtures: synthetic inference log records
# =====================================================================


def _make_record(
    *,
    request_id: str = "req-001",
    timestamp: str | None = None,
    top_card: str = "chase_sapphire_preferred",
    total_latency: float = 150.0,
    normalize_latency: float = 5.0,
    deterministic_latency: float = 80.0,
    personalization_latency: float = 40.0,
    rank_latency: float = 10.0,
    llm_latency: float = 0.0,
    is_personalized: bool = True,
    recommendation_flow: str = "predict",
    request_status: str = "success",
    explanation_latency_ms: float | None = None,
    llm_telemetry: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a synthetic inference log record."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    record: Dict[str, Any] = {
        "timestamp": timestamp,
        "request_id": request_id,
        "user_hash": "abc123",
        "input_features": {
            "spending_categories": {"dining": 500, "groceries": 300},
            "monthly_spend": 2000,
            "preferred_rewards": ["travel"],
            "transaction_history_count": 5,
        },
        "predicted_scores": [
            {
                "card_name": top_card,
                "rank": 1,
                "deterministic_score": 25.0,
                "personalization_score": 30.0,
                "blended_score": 27.0,
            }
        ],
        "top_card": top_card,
        "model_version": "1.0.0",
        "latency_breakdown_ms": {
            "normalize": normalize_latency,
            "deterministic": deterministic_latency,
            "personalization": personalization_latency,
            "rank": rank_latency,
            "total": total_latency,
        },
        "is_personalized": is_personalized,
        "recommendation_flow": recommendation_flow,
        "request_status": request_status,
    }

    if llm_latency > 0:
        record["latency_breakdown_ms"]["llm_explanation"] = llm_latency

    if explanation_latency_ms is not None:
        record["explanation_latency_ms"] = explanation_latency_ms

    if llm_telemetry is not None:
        record["llm_telemetry"] = llm_telemetry

    return record


def _make_records_for_day(
    date: datetime,
    count: int = 10,
    error_count: int = 0,
    with_llm: bool = False,
) -> List[Dict[str, Any]]:
    """Generate *count* records for a specific date."""
    records = []
    for i in range(count):
        ts = date.replace(hour=10, minute=i % 60).isoformat()
        status = "error" if i < error_count else "success"

        llm_telem = None
        expl_lat = None
        llm_lat = 0.0
        if with_llm and status == "success":
            successes = 2 if i % 3 != 0 else 1
            fallbacks = 3 - successes
            llm_telem = {
                "llm_calls": 3,
                "llm_successes": successes,
                "llm_fallbacks": fallbacks,
                "llm_model_name": "gemini-1.5-flash",
                "llm_token_estimate": 450,
                "llm_cost_estimate_usd": 0.00225,
                "llm_prompt_version": "v2.1-abc123",
            }
            expl_lat = 120.0 + i * 5
            llm_lat = expl_lat

        flow = ["predict", "portfolio", "transaction"][i % 3]
        cards = [
            "chase_sapphire_preferred",
            "amex_gold",
            "citi_double_cash",
            "capital_one_venture_x",
        ]

        records.append(
            _make_record(
                request_id=f"req-{date.strftime('%Y%m%d')}-{i:03d}",
                timestamp=ts,
                top_card=cards[i % len(cards)],
                total_latency=100.0 + i * 20,
                is_personalized=i % 4 != 0,
                recommendation_flow=flow,
                request_status=status,
                explanation_latency_ms=expl_lat,
                llm_latency=llm_lat,
                llm_telemetry=llm_telem,
            )
        )
    return records


def _write_records_to_dir(base_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Write records to a date-partitioned local directory."""
    for record in records:
        ts = record["timestamp"]
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        day_dir = base_dir / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        filepath = day_dir / f"{record['request_id']}.json"
        filepath.write_text(json.dumps(record, default=str), encoding="utf-8")


# =====================================================================
# Tests: LatencyStats
# =====================================================================


class TestLatencyStats:
    def test_from_empty(self):
        stats = LatencyStats.from_values([])
        assert stats.mean_ms == 0.0
        assert stats.p95_ms == 0.0

    def test_from_single_value(self):
        stats = LatencyStats.from_values([100.0])
        assert stats.mean_ms == 100.0
        assert stats.p50_ms == 100.0
        assert stats.p95_ms == 100.0

    def test_from_multiple_values(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats = LatencyStats.from_values(values)
        assert stats.mean_ms == 55.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 100.0
        assert stats.p50_ms == 60.0
        assert stats.p95_ms == 100.0

    def test_to_dict(self):
        stats = LatencyStats.from_values([100.0, 200.0])
        d = stats.to_dict()
        assert "mean_ms" in d
        assert "p95_ms" in d
        assert isinstance(d["mean_ms"], float)


# =====================================================================
# Tests: BusinessMetricsCollector - aggregate from records
# =====================================================================


class TestBusinessMetricsCollectorFromRecords:
    def setup_method(self):
        self.collector = BusinessMetricsCollector()

    def test_empty_records(self):
        metrics = self.collector.collect_from_records([])
        assert metrics.total_requests == 0
        assert metrics.overall_latency.mean_ms == 0.0

    def test_single_record(self):
        records = [_make_record(total_latency=200.0)]
        metrics = self.collector.collect_from_records(records)
        assert metrics.total_requests == 1
        assert metrics.total_successes == 1
        assert metrics.total_errors == 0
        assert metrics.overall_latency.mean_ms == 200.0

    def test_error_rate(self):
        records = [
            _make_record(request_id="ok-1", request_status="success"),
            _make_record(request_id="ok-2", request_status="success"),
            _make_record(request_id="err-1", request_status="error"),
        ]
        metrics = self.collector.collect_from_records(records)
        assert metrics.total_requests == 3
        assert metrics.total_errors == 1
        assert abs(metrics.overall_error_rate - 1 / 3) < 0.01

    def test_flow_counts(self):
        records = [
            _make_record(request_id="r1", recommendation_flow="predict"),
            _make_record(request_id="r2", recommendation_flow="portfolio"),
            _make_record(request_id="r3", recommendation_flow="predict"),
            _make_record(request_id="r4", recommendation_flow="transaction"),
        ]
        metrics = self.collector.collect_from_records(records)
        assert metrics.overall_flow_counts["predict"] == 2
        assert metrics.overall_flow_counts["portfolio"] == 1
        assert metrics.overall_flow_counts["transaction"] == 1

    def test_top_cards(self):
        records = [
            _make_record(request_id="r1", top_card="amex_gold"),
            _make_record(request_id="r2", top_card="amex_gold"),
            _make_record(request_id="r3", top_card="citi_double_cash"),
        ]
        metrics = self.collector.collect_from_records(records)
        assert metrics.overall_top_cards["amex_gold"] == 2
        assert metrics.overall_top_cards["citi_double_cash"] == 1

    def test_personalization_rate(self):
        records = [
            _make_record(request_id="r1", is_personalized=True),
            _make_record(request_id="r2", is_personalized=True),
            _make_record(request_id="r3", is_personalized=False),
            _make_record(request_id="r4", is_personalized=False),
        ]
        metrics = self.collector.collect_from_records(records)
        assert abs(metrics.overall_personalization_rate - 0.5) < 0.01

    def test_llm_telemetry_aggregation(self):
        llm = {
            "llm_calls": 3,
            "llm_successes": 2,
            "llm_fallbacks": 1,
            "llm_model_name": "gemini-1.5-flash",
            "llm_token_estimate": 500,
            "llm_prompt_version": "v2.1-abc",
        }
        records = [
            _make_record(request_id="r1", llm_telemetry=llm),
            _make_record(request_id="r2", llm_telemetry=llm),
        ]
        metrics = self.collector.collect_from_records(records)
        assert metrics.overall_llm.total_calls == 6
        assert metrics.overall_llm.total_successes == 4
        assert metrics.overall_llm.total_fallbacks == 2
        assert metrics.overall_llm.estimated_tokens == 1000
        assert "gemini-1.5-flash" in metrics.overall_llm.model_names_used

    def test_daily_breakdown(self):
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        records = [
            _make_record(
                request_id="r1",
                timestamp=now.isoformat(),
            ),
            _make_record(
                request_id="r2",
                timestamp=yesterday.isoformat(),
            ),
            _make_record(
                request_id="r3",
                timestamp=now.isoformat(),
            ),
        ]
        metrics = self.collector.collect_from_records(records)
        assert len(metrics.daily_breakdown) >= 2
        day_counts = {d.date: d.request_count for d in metrics.daily_breakdown}
        assert day_counts.get(now.strftime("%Y-%m-%d")) == 2
        assert day_counts.get(yesterday.strftime("%Y-%m-%d")) == 1

    def test_legacy_records_without_story51_fields(self):
        """Records without Story 5.1 fields should still aggregate."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": "legacy-001",
            "user_hash": "abc",
            "input_features": {},
            "predicted_scores": [],
            "top_card": "amex_gold",
            "model_version": "0.9.0",
            "latency_breakdown_ms": {"total": 100.0},
            "is_personalized": True,
        }
        metrics = self.collector.collect_from_records([record])
        assert metrics.total_requests == 1
        # Missing fields should default gracefully
        assert metrics.overall_flow_counts.get("predict", 0) == 1
        assert metrics.overall_error_rate == 0.0

    def test_to_dict_serializable(self):
        records = [_make_record()]
        metrics = self.collector.collect_from_records(records)
        d = metrics.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0


# =====================================================================
# Tests: BusinessMetricsCollector - load from local filesystem
# =====================================================================


class TestBusinessMetricsCollectorLocal:
    def test_collect_from_local_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            now = datetime.now(timezone.utc)
            records = _make_records_for_day(now, count=5, with_llm=True)
            _write_records_to_dir(Path(tmpdir), records)

            collector = BusinessMetricsCollector(local_log_dir=tmpdir)
            metrics = collector.collect(days=1, end_date=now)

            assert metrics.total_requests == 5
            assert len(metrics.daily_breakdown) == 1
            assert metrics.daily_breakdown[0].request_count == 5

    def test_collect_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BusinessMetricsCollector(local_log_dir=tmpdir)
            metrics = collector.collect(days=3)
            assert metrics.total_requests == 0
            assert len(metrics.daily_breakdown) == 3

    def test_collect_multi_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            now = datetime.now(timezone.utc)
            yesterday = now - timedelta(days=1)

            records_today = _make_records_for_day(now, count=8)
            records_yesterday = _make_records_for_day(yesterday, count=4, error_count=1)

            _write_records_to_dir(Path(tmpdir), records_today + records_yesterday)

            collector = BusinessMetricsCollector(local_log_dir=tmpdir)
            metrics = collector.collect(days=2, end_date=now)

            assert metrics.total_requests == 12
            assert metrics.total_errors == 1
            assert len(metrics.daily_breakdown) == 2

    def test_collect_with_llm_telemetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            now = datetime.now(timezone.utc)
            records = _make_records_for_day(now, count=6, with_llm=True)
            _write_records_to_dir(Path(tmpdir), records)

            collector = BusinessMetricsCollector(local_log_dir=tmpdir)
            metrics = collector.collect(days=1, end_date=now)

            assert metrics.overall_llm.total_calls > 0
            assert metrics.overall_llm.estimated_tokens > 0
            assert metrics.overall_llm.estimated_cost_usd > 0


# =====================================================================
# Tests: BusinessReportGenerator - HTML
# =====================================================================


class TestBusinessReportHTML:
    def _make_test_metrics(self) -> PeriodMetrics:
        now = datetime.now(timezone.utc)
        records = _make_records_for_day(now, count=20, error_count=2, with_llm=True)
        collector = BusinessMetricsCollector()
        return collector.collect_from_records(records)

    def test_generate_html_creates_file(self):
        metrics = self._make_test_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            result = BusinessReportGenerator().generate_html(metrics, str(path))
            assert result.exists()
            content = path.read_text(encoding="utf-8")
            assert "RewardSense Business Metrics" in content
            assert "Total Requests" in content

    def test_html_contains_key_metrics(self):
        metrics = self._make_test_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            BusinessReportGenerator().generate_html(metrics, str(path))
            content = path.read_text(encoding="utf-8")
            # Check key sections are present
            assert "Latency Breakdown" in content
            assert "Recommendation Flows" in content
            assert "Top Recommended Cards" in content
            assert "LLM Telemetry" in content
            assert "Daily Breakdown" in content

    def test_html_with_empty_metrics(self):
        metrics = PeriodMetrics(start_date="2026-04-01", end_date="2026-04-07")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty_report.html"
            result = BusinessReportGenerator().generate_html(metrics, str(path))
            assert result.exists()
            content = path.read_text(encoding="utf-8")
            assert "No data" in content

    def test_html_creates_parent_dirs(self):
        metrics = self._make_test_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "report.html"
            result = BusinessReportGenerator().generate_html(metrics, str(path))
            assert result.exists()


# =====================================================================
# Tests: BusinessReportGenerator - PDF
# =====================================================================


class TestBusinessReportPDF:
    def _make_test_metrics(self) -> PeriodMetrics:
        now = datetime.now(timezone.utc)
        records = _make_records_for_day(now, count=15, error_count=1, with_llm=True)
        collector = BusinessMetricsCollector()
        return collector.collect_from_records(records)

    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    def test_generate_pdf_creates_file(self):
        metrics = self._make_test_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.pdf"
            result = BusinessReportGenerator().generate_pdf(metrics, str(path))
            assert result is not None
            assert result.exists()
            assert result.stat().st_size > 0

    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    def test_pdf_with_empty_metrics(self):
        metrics = PeriodMetrics(start_date="2026-04-01", end_date="2026-04-07")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.pdf"
            result = BusinessReportGenerator().generate_pdf(metrics, str(path))
            assert result is not None
            assert result.exists()

    def test_pdf_returns_none_without_reportlab(self):
        metrics = self._make_test_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.pdf"
            with patch(
                "src.monitoring.business_report_generator.REPORTLAB_AVAILABLE",
                False,
            ):
                result = BusinessReportGenerator().generate_pdf(metrics, str(path))
                assert result is None


# =====================================================================
# Tests: BusinessReportGenerator - JSON
# =====================================================================


class TestBusinessReportJSON:
    def test_generate_json(self):
        now = datetime.now(timezone.utc)
        records = _make_records_for_day(now, count=5)
        metrics = BusinessMetricsCollector().collect_from_records(records)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            result = BusinessReportGenerator().generate_json(metrics, str(path))
            assert result.exists()

            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["total_requests"] == 5
            assert "daily_breakdown" in data
            assert "overall_latency" in data


# =====================================================================
# Tests: CLI script
# =====================================================================


class TestCLI:
    def test_parse_defaults(self):
        from scripts.generate_business_report import _parse_args

        args = _parse_args([])
        assert args.days == 7
        assert args.format == ["html"]
        assert args.output_dir == "reports"

    def test_parse_custom_args(self):
        from scripts.generate_business_report import _parse_args

        args = _parse_args(
            [
                "--days",
                "30",
                "--format",
                "html",
                "pdf",
                "json",
                "--output-dir",
                "/tmp/custom",
                "--log-dir",
                "/tmp/logs",
            ]
        )
        assert args.days == 30
        assert args.format == ["html", "pdf", "json"]
        assert args.output_dir == "/tmp/custom"
        assert args.log_dir == "/tmp/logs"

    def test_parse_date_range(self):
        from scripts.generate_business_report import _parse_args

        args = _parse_args(
            [
                "--start-date",
                "2026-03-01",
                "--end-date",
                "2026-03-31",
            ]
        )
        assert args.start_date == "2026-03-01"
        assert args.end_date == "2026-03-31"

    def test_main_with_synthetic_data(self):
        from scripts.generate_business_report import main

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            output_dir = Path(tmpdir) / "reports"

            now = datetime.now(timezone.utc)
            records = _make_records_for_day(now, count=10, with_llm=True)
            _write_records_to_dir(log_dir, records)

            exit_code = main(
                [
                    "--days",
                    "1",
                    "--format",
                    "html",
                    "json",
                    "--output-dir",
                    str(output_dir),
                    "--log-dir",
                    str(log_dir),
                ]
            )
            assert exit_code == 0

            html_files = list(output_dir.glob("*.html"))
            json_files = list(output_dir.glob("*.json"))
            assert len(html_files) == 1
            assert len(json_files) == 1

    def test_main_with_empty_logs(self):
        from scripts.generate_business_report import main

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main(
                [
                    "--days",
                    "1",
                    "--format",
                    "html",
                    "--output-dir",
                    str(Path(tmpdir) / "reports"),
                    "--log-dir",
                    str(Path(tmpdir) / "empty_logs"),
                ]
            )
            assert exit_code == 0


# =====================================================================
# Tests: Extended inference_logger build_log_record
# =====================================================================


class TestExtendedLogRecord:
    def test_backward_compatible(self):
        """Old callers (without Story 5.1 fields) should still work."""
        from src.serving.inference_logger import build_log_record

        record = build_log_record(
            request_id="test-001",
            user_hash="abc",
            input_features={"test": True},
            scores=[],
            top_card="amex_gold",
            model_version="1.0.0",
            latency_breakdown={"total": 100.0},
            is_personalized=True,
        )
        assert record["recommendation_flow"] == "predict"
        assert record["request_status"] == "success"
        assert "llm_telemetry" not in record

    def test_with_story51_fields(self):
        from src.serving.inference_logger import build_log_record

        llm_telem = {
            "llm_calls": 3,
            "llm_successes": 2,
            "llm_fallbacks": 1,
            "llm_model_name": "gemini-1.5-flash",
            "llm_token_estimate": 500,
        }
        record = build_log_record(
            request_id="test-002",
            user_hash="abc",
            input_features={},
            scores=[],
            top_card="chase_sapphire_preferred",
            model_version="1.0.0",
            latency_breakdown={"total": 200.0},
            is_personalized=True,
            recommendation_flow="portfolio",
            request_status="success",
            llm_telemetry=llm_telem,
        )
        assert record["recommendation_flow"] == "portfolio"
        assert record["llm_telemetry"]["llm_calls"] == 3
        assert record["llm_telemetry"]["llm_model_name"] == "gemini-1.5-flash"

    def test_explanation_latency_included(self):
        from src.serving.inference_logger import build_log_record

        record = build_log_record(
            request_id="test-003",
            user_hash="abc",
            input_features={},
            scores=[],
            top_card="amex_gold",
            model_version="1.0.0",
            latency_breakdown={"total": 300.0},
            is_personalized=False,
            explanation_latency_ms=150.5,
        )
        assert record["explanation_latency_ms"] == 150.5

    def test_explanation_latency_excluded_when_none(self):
        from src.serving.inference_logger import build_log_record

        record = build_log_record(
            request_id="test-004",
            user_hash="abc",
            input_features={},
            scores=[],
            top_card="amex_gold",
            model_version="1.0.0",
            latency_breakdown={"total": 100.0},
            is_personalized=True,
        )
        assert "explanation_latency_ms" not in record
