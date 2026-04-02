"""
Tests for Monitoring Pipeline

Covers:
  - InferenceDataCollector
  - DriftDetector
  - PerformanceTracker
  - SlackNotifier
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_collector import (
    InferenceDataCollector,
    InferenceDataSummary,
)
from src.monitoring.drift_detector import DriftDetector, DriftResult
from src.monitoring.performance_tracker import (
    PerformanceSnapshot,
    PerformanceTracker,
)
from src.monitoring.notifier import SlackNotifier

from src.monitoring.drift_detector import EVIDENTLY_AVAILABLE

# =====================================================================
# Fixtures: sample inference log records
# =====================================================================


def _make_log_record(
    request_id: str = "req-001",
    monthly_spend: float = 3000.0,
    top_card: str = "Chase Sapphire Preferred",
    total_ms: float = 150.0,
    model_version: str = "3",
    is_personalized: bool = True,
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "user_hash": "abc123def456",
        "input_features": {
            "spending_categories": {"dining": 500, "travel": 800, "groceries": 400},
            "monthly_spend": monthly_spend,
            "preferred_rewards": ["travel_points"],
            "transaction_history_count": 25,
        },
        "predicted_scores": [
            {
                "card_name": "Chase Sapphire Preferred",
                "rank": 1,
                "deterministic_score": 45.0,
                "personalization_score": 52.0,
                "blended_score": 48.2,
            },
            {
                "card_name": "Amex Gold Card",
                "rank": 2,
                "deterministic_score": 40.0,
                "personalization_score": 38.0,
                "blended_score": 39.2,
            },
        ],
        "top_card": top_card,
        "model_version": model_version,
        "latency_breakdown_ms": {
            "normalize": 2.0,
            "deterministic": 15.0,
            "personalization": 80.0,
            "rank": 3.0,
            "total": total_ms,
        },
        "is_personalized": is_personalized,
    }


@pytest.fixture
def sample_records():
    """Generate 20 sample inference log records."""
    rng = np.random.default_rng(42)
    records = []
    cards = ["Chase Sapphire Preferred", "Amex Gold Card", "Citi Double Cash"]
    for i in range(20):
        records.append(
            _make_log_record(
                request_id=f"req-{i:03d}",
                monthly_spend=max(500.0, float(rng.normal(3000, 1000))),
                top_card=rng.choice(cards),
                total_ms=max(50.0, float(rng.normal(200, 50))),
                model_version="3",
                is_personalized=bool(rng.random() > 0.2),
            )
        )
    return records


@pytest.fixture
def local_logs_dir(tmp_path, sample_records):
    """Write sample records to local filesystem in date-partitioned dirs."""
    now = datetime.now(timezone.utc)
    prefix = f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
    log_dir = tmp_path / "logs" / prefix
    log_dir.mkdir(parents=True)

    for record in sample_records:
        filepath = log_dir / f"{record['request_id']}.json"
        filepath.write_text(json.dumps(record), encoding="utf-8")

    return tmp_path / "logs"


@pytest.fixture
def sample_summary(sample_records):
    """Build an InferenceDataSummary from sample records."""
    summary = InferenceDataSummary(
        start_date=datetime.now(timezone.utc) - timedelta(days=1),
        end_date=datetime.now(timezone.utc),
        records=sample_records,
    )
    summary.to_dataframes()
    return summary


# =====================================================================
# InferenceDataCollector
# =====================================================================


class TestInferenceDataCollector:
    def test_collect_from_local(self, local_logs_dir):
        collector = InferenceDataCollector(
            bucket="unused", local_dir=str(local_logs_dir)
        )
        collector._gcs_client = None  # force local mode
        summary = collector.collect(days=1)
        assert summary.total_records == 20
        assert summary.input_features_df is not None
        assert len(summary.input_features_df) == 20

    def test_dataframes_schema(self, local_logs_dir):
        collector = InferenceDataCollector(
            bucket="unused", local_dir=str(local_logs_dir)
        )
        collector._gcs_client = None
        summary = collector.collect(days=1)

        assert "monthly_spend" in summary.input_features_df.columns
        assert "spend_dining" in summary.input_features_df.columns
        assert "blended_score" in summary.predictions_df.columns
        assert "total_ms" in summary.latency_df.columns
        assert "model_version" in summary.metadata_df.columns

    def test_empty_directory(self, tmp_path):
        collector = InferenceDataCollector(
            bucket="unused", local_dir=str(tmp_path / "empty")
        )
        collector._gcs_client = None
        summary = collector.collect(days=1)
        assert summary.total_records == 0
        assert summary.input_features_df is not None
        assert summary.input_features_df.empty

    def test_summary_stats(self, local_logs_dir):
        collector = InferenceDataCollector(
            bucket="unused", local_dir=str(local_logs_dir)
        )
        collector._gcs_client = None
        summary = collector.collect(days=1)
        stats = summary.summary_stats
        assert stats["total_records"] == 20
        assert "latency" in stats
        assert "p50_ms" in stats["latency"]
        assert "top_card_frequency" in stats

    def test_custom_date_range(self, local_logs_dir):
        collector = InferenceDataCollector(
            bucket="unused", local_dir=str(local_logs_dir)
        )
        collector._gcs_client = None
        now = datetime.now(timezone.utc)
        summary = collector.collect(
            start_date=now - timedelta(hours=12),
            end_date=now,
        )
        assert summary.total_records == 20


# =====================================================================
# DriftDetector
# =====================================================================


class TestDriftDetector:
    @pytest.fixture
    def reference_df(self):
        rng = np.random.default_rng(42)
        n = 100
        return pd.DataFrame(
            {
                "monthly_spend": rng.normal(3000, 1000, n),
                "spend_dining": rng.normal(500, 200, n),
                "spend_travel": rng.normal(800, 300, n),
                "spend_groceries": rng.normal(400, 150, n),
            }
        )

    def test_no_drift_similar_distributions(self, tmp_path, reference_df):
        """Same distribution should show no drift."""
        ref_path = tmp_path / "reference.csv"
        reference_df.to_csv(ref_path, index=False)

        # Current data from same distribution
        rng = np.random.default_rng(99)
        current_df = pd.DataFrame(
            {
                "monthly_spend": rng.normal(3000, 1000, 100),
                "spend_dining": rng.normal(500, 200, 100),
                "spend_travel": rng.normal(800, 300, 100),
                "spend_groceries": rng.normal(400, 150, 100),
            }
        )

        detector = DriftDetector(
            reference_path=str(ref_path),
            output_dir=str(tmp_path / "reports"),
        )
        result = detector.detect(current_df)
        assert isinstance(result, DriftResult)
        # Similar distributions should have low drift share
        assert result.dataset_drift_share < 0.5

    @pytest.mark.skipif(not EVIDENTLY_AVAILABLE, reason="evidently not installed")
    def test_drift_detected_shifted_distribution(self, tmp_path, reference_df):
        """Heavily shifted distribution should detect drift."""
        ref_path = tmp_path / "reference.csv"
        reference_df.to_csv(ref_path, index=False)

        rng = np.random.default_rng(99)
        current_df = pd.DataFrame(
            {
                "monthly_spend": rng.normal(8000, 500, 100),  # big shift
                "spend_dining": rng.normal(2000, 100, 100),  # big shift
                "spend_travel": rng.normal(5000, 200, 100),  # big shift
                "spend_groceries": rng.normal(3000, 100, 100),  # big shift
            }
        )

        detector = DriftDetector(
            reference_path=str(ref_path),
            feature_drift_threshold=0.3,
            output_dir=str(tmp_path / "reports"),
        )
        result = detector.detect(current_df)
        assert result.drift_detected is True
        assert len(result.drifted_features) > 0

    def test_saves_html_report(self, tmp_path, reference_df):
        ref_path = tmp_path / "reference.csv"
        reference_df.to_csv(ref_path, index=False)

        detector = DriftDetector(
            reference_path=str(ref_path),
            output_dir=str(tmp_path / "reports"),
        )
        result = detector.detect(reference_df.copy())
        if result.html_report_path:
            assert Path(result.html_report_path).exists()

    def test_saves_json_report(self, tmp_path, reference_df):
        ref_path = tmp_path / "reference.csv"
        reference_df.to_csv(ref_path, index=False)

        detector = DriftDetector(
            reference_path=str(ref_path),
            output_dir=str(tmp_path / "reports"),
        )
        result = detector.detect(reference_df.copy())
        if result.json_report_path:
            data = json.loads(Path(result.json_report_path).read_text())
            assert "summary" in data

    @pytest.mark.skipif(not EVIDENTLY_AVAILABLE, reason="evidently not installed")
    def test_missing_reference_raises(self, tmp_path):
        detector = DriftDetector(
            reference_path=str(tmp_path / "nonexistent.csv"),
            output_dir=str(tmp_path / "reports"),
        )
        detector._reference_df = None
        with pytest.raises(FileNotFoundError):
            detector.load_reference()

    def test_no_common_columns(self, tmp_path):
        ref_path = tmp_path / "reference.csv"
        pd.DataFrame({"col_a": [1, 2, 3]}).to_csv(ref_path, index=False)

        detector = DriftDetector(
            reference_path=str(ref_path),
            output_dir=str(tmp_path / "reports"),
        )
        result = detector.detect(pd.DataFrame({"col_b": [4, 5, 6]}))
        assert result.drift_detected is False

    def test_result_summary(self, tmp_path, reference_df):
        ref_path = tmp_path / "reference.csv"
        reference_df.to_csv(ref_path, index=False)

        detector = DriftDetector(
            reference_path=str(ref_path),
            output_dir=str(tmp_path / "reports"),
        )
        result = detector.detect(reference_df.copy())
        s = result.summary
        assert "drift_detected" in s
        assert "n_reference" in s
        assert "n_current" in s


# =====================================================================
# PerformanceTracker
# =====================================================================


class TestPerformanceTracker:
    def test_compute_snapshot(self, sample_summary):
        tracker = PerformanceTracker()
        snapshot = tracker.compute(sample_summary)
        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.total_requests == 20
        assert snapshot.latency_p50_ms > 0
        assert snapshot.latency_p95_ms >= snapshot.latency_p50_ms

    def test_score_metrics(self, sample_summary):
        tracker = PerformanceTracker()
        snapshot = tracker.compute(sample_summary)
        assert snapshot.score_mean > 0
        assert snapshot.score_std >= 0

    def test_personalization_rate(self, sample_summary):
        tracker = PerformanceTracker()
        snapshot = tracker.compute(sample_summary)
        assert 0 <= snapshot.personalization_rate <= 1

    def test_top_card_distribution(self, sample_summary):
        tracker = PerformanceTracker()
        snapshot = tracker.compute(sample_summary)
        assert len(snapshot.top_card_distribution) > 0

    def test_top_card_entropy(self, sample_summary):
        tracker = PerformanceTracker()
        snapshot = tracker.compute(sample_summary)
        assert snapshot.top_card_entropy > 0  # multiple cards

    def test_latency_alert(self, sample_summary):
        tracker = PerformanceTracker(latency_threshold_ms=50)  # very low
        snapshot = tracker.compute(sample_summary)
        assert snapshot.has_alerts
        assert any("LATENCY" in a for a in snapshot.alerts)

    def test_no_alert_normal_latency(self, sample_summary):
        tracker = PerformanceTracker(latency_threshold_ms=50000)
        snapshot = tracker.compute(sample_summary)
        assert not snapshot.has_alerts

    def test_save_snapshot(self, tmp_path, sample_summary):
        tracker = PerformanceTracker(output_dir=str(tmp_path / "perf"))
        snapshot = tracker.compute(sample_summary)
        path = tracker.save_snapshot(snapshot)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_requests"] == 20

    def test_load_history(self, tmp_path, sample_summary):
        tracker = PerformanceTracker(output_dir=str(tmp_path / "perf"))
        snapshot = tracker.compute(sample_summary)
        tracker.save_snapshot(snapshot)
        history = tracker.load_history()
        assert len(history) == 1

    def test_empty_summary(self):
        summary = InferenceDataSummary(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
        )
        summary.to_dataframes()
        tracker = PerformanceTracker()
        snapshot = tracker.compute(summary)
        assert snapshot.total_requests == 0

    def test_snapshot_serializable(self, sample_summary):
        tracker = PerformanceTracker()
        snapshot = tracker.compute(sample_summary)
        d = snapshot.to_dict()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100


# =====================================================================
# SlackNotifier
# =====================================================================


class TestSlackNotifier:
    def test_dry_run_doesnt_send(self):
        notifier = SlackNotifier(dry_run=True)
        drift = MagicMock()
        drift.summary = {"drift_detected": True, "drifted_features": ["col_a"]}
        perf = MagicMock()
        perf.to_dict.return_value = {
            "total_requests": 100,
            "latency": {"p95_ms": 200},
            "personalization_rate": 0.8,
            "alerts": [],
        }
        result = notifier.send_monitoring_summary(drift, perf)
        assert result is True

    def test_unconfigured_skips(self):
        notifier = SlackNotifier(webhook_url="")
        result = notifier.send_retrain_trigger("test_reason")
        assert result is False

    def test_is_configured(self):
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier.is_configured is True

    def test_not_configured_without_url(self):
        notifier = SlackNotifier(webhook_url="")
        assert notifier.is_configured is False

    def test_send_retrain_trigger_dry_run(self):
        notifier = SlackNotifier(dry_run=True)
        result = notifier.send_retrain_trigger(
            reason="data_drift",
            drift_report_path="/path/to/report.html",
            threshold_values={"drift_share": 0.45},
        )
        assert result is True

    def test_send_redeployment_dry_run(self):
        notifier = SlackNotifier(dry_run=True)
        result = notifier.send_redeployment(
            new_version="4",
            old_version="3",
            performance_comparison={"accuracy_delta": "+0.02"},
        )
        assert result is True

    @patch("src.monitoring.notifier.http_requests")
    def test_send_success(self, mock_requests):
        mock_requests.post.return_value = MagicMock(status_code=200)
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        result = notifier.send_retrain_trigger("drift")
        assert result is True
        mock_requests.post.assert_called_once()

    @patch("src.monitoring.notifier.http_requests")
    def test_send_failure_doesnt_crash(self, mock_requests):
        mock_requests.post.side_effect = ConnectionError("timeout")
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        result = notifier.send_retrain_trigger("drift")
        assert result is False
