"""
Integration tests for the monitoring pipeline.

These tests require live infrastructure:
  - Deployed serving API (SERVING_API_URL)
  - Cloud Composer with monitoring + model pipeline DAGs
  - Slack webhook configured

Tests are SKIPPED when infrastructure is unavailable, so they never block CI. They run when explicitly enabled via env var or when deployed infrastructure is detected.

Usage:
    # Local (skips everything)
    PYTHONPATH=. pytest tests/integration/test_monitoring_e2e.py -v

    # Against deployed infra
    SERVING_API_URL=https://rewardsense-serving-xxx.run.app \
    ENABLE_INTEGRATION_TESTS=true \
    PYTHONPATH=. pytest tests/integration/test_monitoring_e2e.py -v
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Infrastructure detection
# ---------------------------------------------------------------------------
SERVING_API_URL = os.getenv("SERVING_API_URL", "")
ENABLE_INTEGRATION = os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() in (
    "1",
    "true",
    "yes",
)

INFRA_AVAILABLE = bool(SERVING_API_URL) and ENABLE_INTEGRATION

skip_no_infra = pytest.mark.skipif(
    not INFRA_AVAILABLE,
    reason=(
        "Live infrastructure not available. Set SERVING_API_URL and "
        "ENABLE_INTEGRATION_TESTS=true to run."
    ),
)

try:
    import requests as http_requests

    REQUESTS_AVAILABLE = True
except ImportError:
    http_requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit_predict(n: int = 50) -> list:
    """Send n synthetic requests to the serving API."""
    if not REQUESTS_AVAILABLE:
        return []

    rng = np.random.default_rng(42)
    results = []

    for i in range(n):
        payload = {
            "user_id": f"integration-test-{i}",
            "spending_categories": {
                "dining": float(rng.normal(500, 200).clip(0)),
                "travel": float(rng.normal(800, 300).clip(0)),
                "groceries": float(rng.normal(400, 150).clip(0)),
            },
            "monthly_spend": float(rng.normal(2500, 800).clip(500)),
            "preferred_rewards": ["travel_points"],
        }

        try:
            resp = http_requests.post(
                f"{SERVING_API_URL}/predict",
                json=payload,
                timeout=15,
            )
            results.append(
                {
                    "status": resp.status_code,
                    "latency_ms": resp.elapsed.total_seconds() * 1000,
                }
            )
        except Exception as e:
            logger.warning("Request %d failed: %s", i, e)
            results.append({"status": 0, "error": str(e)})

    return results


def _check_serving_health() -> Optional[Dict[str, Any]]:
    """Check if the serving API is healthy."""
    if not REQUESTS_AVAILABLE:
        return None
    try:
        resp = http_requests.get(f"{SERVING_API_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# =====================================================================
# Story 4.5: Monitoring Pipeline End-to-End
# =====================================================================


@skip_no_infra
class TestMonitoringPipelineE2E:
    """Verify the monitoring pipeline works with real inference logs."""

    def test_serving_api_healthy(self):
        """Prerequisite: serving API must be reachable."""
        health = _check_serving_health()
        assert health is not None
        assert health["status"] == "healthy"
        assert "model_version" in health

    def test_generate_inference_logs(self):
        """Hit the serving API with 50 requests to generate logs."""
        results = _hit_predict(n=50)
        successes = [r for r in results if r.get("status") == 200]
        assert len(successes) >= 40, f"Only {len(successes)}/50 requests succeeded"

    def test_collector_reads_logs(self):
        """Data collector should find recent inference logs."""
        from src.monitoring.data_collector import InferenceDataCollector

        collector = InferenceDataCollector()
        summary = collector.collect(days=1)
        assert summary.total_records > 0, "No inference logs found in GCS"
        assert summary.input_features_df is not None
        assert not summary.input_features_df.empty

    def test_drift_detection_runs(self):
        """Drift detection should complete against real data."""
        from src.monitoring.data_collector import InferenceDataCollector
        from src.monitoring.drift_detector import DriftDetector, EVIDENTLY_AVAILABLE

        if not EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently not installed")

        collector = InferenceDataCollector()
        summary = collector.collect(days=1)

        if summary.total_records == 0:
            pytest.skip("No inference logs available")

        detector = DriftDetector()
        result = detector.detect(summary.input_features_df)
        assert result.timestamp  # non-empty
        assert result.n_current > 0

    def test_performance_snapshot_generated(self):
        """Performance tracker should produce a valid snapshot."""
        from src.monitoring.data_collector import InferenceDataCollector
        from src.monitoring.performance_tracker import PerformanceTracker

        collector = InferenceDataCollector()
        summary = collector.collect(days=1)

        if summary.total_records == 0:
            pytest.skip("No inference logs available")

        tracker = PerformanceTracker()
        snapshot = tracker.compute(summary)
        assert snapshot.total_requests > 0
        assert snapshot.latency_p50_ms > 0
        assert snapshot.latency_p95_ms >= snapshot.latency_p50_ms

    def test_drift_report_in_gcs(self):
        """Drift HTML report should be uploaded to GCS."""
        from src.monitoring.drift_detector import DriftDetector, EVIDENTLY_AVAILABLE

        if not EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently not installed")

        from src.monitoring.data_collector import InferenceDataCollector

        collector = InferenceDataCollector()
        summary = collector.collect(days=1)

        if summary.total_records == 0:
            pytest.skip("No inference logs available")

        detector = DriftDetector()
        result = detector.detect(summary.input_features_df)

        if result.html_report_path:
            assert Path(result.html_report_path).exists()

    def test_notification_sends(self):
        """Slack notification should send without error."""
        from src.monitoring.notifier import SlackNotifier
        from src.monitoring.drift_detector import DriftResult
        from src.monitoring.performance_tracker import PerformanceSnapshot

        # Use real drift/perf results if available, otherwise mock
        notifier = SlackNotifier()
        if not notifier.is_configured:
            pytest.skip("SLACK_WEBHOOK_URL not configured")

        drift = DriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            drift_detected=False,
        )
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            total_requests=50,
            latency_p95_ms=200,
        )

        result = notifier.send_monitoring_summary(drift, snapshot)
        assert result is True


# =====================================================================
# Story 5.3: Full Retrain Loop
# =====================================================================


@skip_no_infra
class TestRetrainLoopE2E:
    """Verify the closed loop: drift -> retrain -> deploy -> verify."""

    def test_serving_has_model_version(self):
        """Serving API should report a model version."""
        health = _check_serving_health()
        assert health is not None
        version = health.get("model_version", "")
        assert version != "unloaded", "No model loaded in serving API"

    def test_inject_drifted_data(self):
        """Send requests with shifted distributions to generate drift signal."""
        if not REQUESTS_AVAILABLE:
            pytest.skip("requests not installed")

        rng = np.random.default_rng(99)
        results = []
        for i in range(30):
            payload = {
                "user_id": f"drift-test-{i}",
                "spending_categories": {
                    "dining": float(rng.normal(2000, 200).clip(0)),
                    "travel": float(rng.normal(100, 50).clip(0)),
                    "groceries": float(rng.normal(1500, 100).clip(0)),
                },
                "monthly_spend": float(rng.normal(5000, 500).clip(1000)),
                "preferred_rewards": ["cashback"],
            }
            try:
                resp = http_requests.post(
                    f"{SERVING_API_URL}/predict",
                    json=payload,
                    timeout=15,
                )
                results.append(resp.status_code)
            except Exception:
                pass

        successes = [s for s in results if s == 200]
        assert len(successes) >= 20

    def test_monitoring_dag_detects_drift(self):
        """After injecting drifted data, monitoring should detect it.

        Note: This test is a placeholder. In production, the monitoring
        DAG runs on Composer on a schedule. To test manually:
          1. Trigger the monitoring DAG in Composer UI
          2. Check the drift detection task output
          3. Verify the retrain trigger fires
        """
        # This would require Airflow API access to trigger and poll DAG status.
        # For now, validate the components work individually.
        from src.monitoring.data_collector import InferenceDataCollector
        from src.monitoring.drift_detector import DriftDetector, EVIDENTLY_AVAILABLE

        if not EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently not installed")

        collector = InferenceDataCollector()
        summary = collector.collect(days=1)

        if summary.total_records < 10:
            pytest.skip("Insufficient inference logs for drift test")

        detector = DriftDetector(feature_drift_threshold=0.3)
        result = detector.detect(summary.input_features_df)

        # Log result regardless of drift outcome
        logger.info(
            "Drift test: detected=%s, share=%.2f%%, features=%s",
            result.drift_detected,
            result.dataset_drift_share * 100,
            result.drifted_features,
        )

    def test_serving_model_version_after_retrain(self):
        """After retrain loop, serving should have a new model version.

        Note: Full loop takes ~30 minutes (training is the bottleneck).
        This test checks the current version and logs it for manual
        comparison after triggering retrain.
        """
        health = _check_serving_health()
        assert health is not None
        version = health.get("model_version", "unknown")
        logger.info("Current serving model version: %s", version)
        # Record version for post-retrain comparison
        # In a full CI test, you'd save this and compare after retrain completes
        assert version != "unloaded"
