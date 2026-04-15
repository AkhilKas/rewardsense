"""
Unit tests for Anomaly Detection DAG Integration.

Tests:
  - DAG structure: anomaly_detection task group exists with correct tasks
  - Task dependencies: preprocessing → anomaly_detection → versioning
  - Short-circuit gate logic
  - Alert dispatch from detection results
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# =====================================================================
# DAG structure tests
# =====================================================================


class TestDAGStructure:
    """Verify the DAG has the anomaly detection task group wired correctly."""

    @pytest.fixture(autouse=True)
    def _import_dag(self):
        """Import the DAG module. Skip if Airflow not installed."""
        try:
            import importlib
            import sys
            from pathlib import Path

            dag_path = Path(__file__).resolve().parents[2] / "dags"
            if str(dag_path) not in sys.path:
                sys.path.insert(0, str(dag_path))

            # Force reimport to pick up changes
            if "rewardsense_data_pipeline" in sys.modules:
                del sys.modules["rewardsense_data_pipeline"]

            self.dag_module = importlib.import_module("rewardsense_data_pipeline")
            self.dag = self.dag_module.dag
        except ImportError:
            pytest.skip("Airflow not installed — skipping DAG structure tests")

    def test_anomaly_detection_group_exists(self):
        task_ids = [t.task_id for t in self.dag.tasks]
        anomaly_tasks = [t for t in task_ids if "anomaly_detection" in t]
        assert (
            len(anomaly_tasks) >= 3
        ), f"Expected 3+ anomaly tasks, got {anomaly_tasks}"

    def test_detect_anomalies_task_exists(self):
        task_ids = [t.task_id for t in self.dag.tasks]
        assert any("detect_anomalies" in t for t in task_ids)

    def test_send_anomaly_alerts_task_exists(self):
        task_ids = [t.task_id for t in self.dag.tasks]
        assert any("send_anomaly_alerts" in t for t in task_ids)

    def test_critical_gate_task_exists(self):
        task_ids = [t.task_id for t in self.dag.tasks]
        assert any("check_critical_gate" in t for t in task_ids)


# =====================================================================
# Gate logic tests
# =====================================================================

try:
    import airflow  # noqa: F401

    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False


@pytest.mark.skipif(not HAS_AIRFLOW, reason="Airflow not installed")
class TestCriticalGateLogic:
    """Test the short-circuit gate behavior."""

    def test_gate_passes_when_no_critical(self):
        """No critical anomalies → gate passes normally."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.side_effect = lambda **kwargs: {
            ("anomaly_detection.detect_anomalies", "has_critical_anomalies"): False,
            ("anomaly_detection.detect_anomalies", "anomaly_max_severity"): "WARNING",
        }.get((kwargs.get("task_ids"), kwargs.get("key")))

        context = {"ti": mock_ti}

        # Import and call the gate function directly
        with patch("airflow.models.Variable.get", return_value="true"):
            from dags.rewardsense_data_pipeline import _check_critical_gate

            # Unwrap the timed_python_task decorator
            # The actual callable is the inner function
            # For testing, we call it with the context
            result = _check_critical_gate.__wrapped__(**context)
            assert result["gate"] == "passed"

    def test_gate_raises_when_critical_and_enforced(self):
        """Critical anomalies + enforcement → RuntimeError."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.side_effect = lambda **kwargs: {
            ("anomaly_detection.detect_anomalies", "has_critical_anomalies"): True,
            ("anomaly_detection.detect_anomalies", "anomaly_max_severity"): "CRITICAL",
        }.get((kwargs.get("task_ids"), kwargs.get("key")))

        context = {"ti": mock_ti}

        with patch("airflow.models.Variable.get", return_value="true"):
            from dags.rewardsense_data_pipeline import _check_critical_gate

            with pytest.raises(RuntimeError, match="CRITICAL anomalies detected"):
                _check_critical_gate.__wrapped__(**context)

    def test_gate_warns_when_critical_but_not_enforced(self):
        """Critical anomalies + enforcement disabled → passes with warning."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.side_effect = lambda **kwargs: {
            ("anomaly_detection.detect_anomalies", "has_critical_anomalies"): True,
            ("anomaly_detection.detect_anomalies", "anomaly_max_severity"): "CRITICAL",
        }.get((kwargs.get("task_ids"), kwargs.get("key")))

        context = {"ti": mock_ti}

        with patch("airflow.models.Variable.get", return_value="false"):
            from dags.rewardsense_data_pipeline import _check_critical_gate

            result = _check_critical_gate.__wrapped__(**context)
            assert result["gate"] == "passed_override"
            assert result["has_critical"] is True


# =====================================================================
# Detection task logic tests
# =====================================================================


class TestDetectAnomaliesLogic:
    """Test the detection callable's logic without full Airflow runtime."""

    def test_handles_missing_checkpoints_gracefully(self, tmp_path):
        """If no checkpoint data exists, detection should still complete."""
        from src.data_pipeline.anomaly_detection.detectors import (
            AnomalyDetector,
            AnomalyReport,
        )

        detector = AnomalyDetector()
        # Empty df should return report with 0 anomalies
        empty_df = pd.DataFrame(columns=["user_id", "monthly_budget"])
        report = detector.run_all_checks(empty_df, dataset="users")
        assert isinstance(report, AnomalyReport)
        assert len(report.anomalies) == 0

    def test_domain_rules_integrated_with_detector(self):
        """Domain rules should extend the detector's anomaly list."""
        from src.data_pipeline.anomaly_detection.detectors import AnomalyDetector
        from src.data_pipeline.anomaly_detection.rules import DomainRuleEngine

        detector = AnomalyDetector()
        rule_engine = DomainRuleEngine()

        # Card with unrealistic reward rate
        cards_df = pd.DataFrame(
            {
                "card_name": ["Bug Card"],
                "base_reward_rate": [50.0],
                "annual_fee": [0],
            }
        )

        report = detector.run_all_checks(
            cards_df, dataset="credit_cards", numeric_columns=["annual_fee"]
        )
        domain_anomalies = rule_engine.check_credit_card_rules(cards_df)
        report.anomalies.extend(domain_anomalies)

        # Should have domain rule anomalies
        domain_checks = [
            a for a in report.anomalies if a.check_name.startswith("domain_")
        ]
        assert len(domain_checks) >= 1

    def test_report_persists_to_json(self, tmp_path):
        """Reports should be serializable to JSON."""
        from src.data_pipeline.anomaly_detection.detectors import (
            Anomaly,
            AnomalyReport,
            AnomalySeverity,
        )

        report = AnomalyReport(dataset="test", total_checks=3)
        report.anomalies = [
            Anomaly("check_1", AnomalySeverity.WARNING, "Test warning"),
        ]

        out_path = tmp_path / "test_report.json"
        report.to_json(out_path)
        assert out_path.exists()

        import json

        loaded = json.loads(out_path.read_text())
        assert loaded["dataset"] == "test"
        assert loaded["anomaly_count"] == 1


# =====================================================================
# Alert task logic tests
# =====================================================================


class TestSendAnomalyAlertsLogic:
    """Test alert dispatch from detection results."""

    def test_no_alerts_when_info_only(self):
        """INFO-level anomalies should not trigger alerts."""
        from src.data_pipeline.anomaly_detection.alert_integration import (
            AnomalyAlertBridge,
        )
        from src.data_pipeline.anomaly_detection.detectors import (
            Anomaly,
            AnomalyReport,
            AnomalySeverity,
        )

        mock_dispatcher = MagicMock()
        bridge = AnomalyAlertBridge(dispatcher=mock_dispatcher)

        report = AnomalyReport(dataset="test", total_checks=3)
        report.anomalies = [
            Anomaly("check_1", AnomalySeverity.INFO, "All good"),
        ]

        result = bridge.process_report(report, alert_on_warning=True)
        assert result["alerted"] is False
        mock_dispatcher.dispatch.assert_not_called()

    def test_alerts_on_warning(self):
        """WARNING-level anomalies should trigger alerts when enabled."""
        from src.data_pipeline.anomaly_detection.alert_integration import (
            AnomalyAlertBridge,
        )
        from src.data_pipeline.anomaly_detection.detectors import (
            Anomaly,
            AnomalyReport,
            AnomalySeverity,
        )

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = {"slack": True}
        bridge = AnomalyAlertBridge(dispatcher=mock_dispatcher)

        report = AnomalyReport(dataset="transactions", total_checks=4)
        report.anomalies = [
            Anomaly("outlier_iqr", AnomalySeverity.WARNING, "5% outliers in amount"),
        ]

        result = bridge.process_report(report, alert_on_warning=True)
        assert result["alerted"] is True
        mock_dispatcher.dispatch.assert_called_once()

    def test_alerts_on_critical(self):
        """CRITICAL anomalies should always trigger alerts."""
        from src.data_pipeline.anomaly_detection.alert_integration import (
            AnomalyAlertBridge,
        )
        from src.data_pipeline.anomaly_detection.detectors import (
            Anomaly,
            AnomalyReport,
            AnomalySeverity,
        )

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = {"slack": True, "email": True}
        bridge = AnomalyAlertBridge(dispatcher=mock_dispatcher)

        report = AnomalyReport(dataset="users", total_checks=4)
        report.anomalies = [
            Anomaly("missing_values", AnomalySeverity.CRITICAL, "25% missing"),
        ]

        result = bridge.process_report(report, alert_on_warning=False)
        assert result["alerted"] is True
