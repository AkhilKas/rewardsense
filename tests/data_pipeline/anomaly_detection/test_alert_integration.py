"""
Unit tests for anomaly → alert bridge.

Verifies anomaly reports are correctly translated to alert dispatches using the existing alerting system.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data_pipeline.anomaly_detection.alert_integration import (
    AnomalyAlertBridge,
    _format_anomaly_message,
)
from src.data_pipeline.anomaly_detection.detectors import (
    Anomaly,
    AnomalyReport,
    AnomalySeverity,
)
from src.data_pipeline.monitoring.alerting import Severity

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def mock_dispatcher():
    d = MagicMock()
    d.dispatch.return_value = {"slack": True}
    d.enabled = True
    return d


@pytest.fixture
def bridge(mock_dispatcher):
    return AnomalyAlertBridge(dispatcher=mock_dispatcher)


@pytest.fixture
def critical_report():
    report = AnomalyReport(dataset="transactions", total_checks=4)
    report.anomalies = [
        Anomaly("missing_values", AnomalySeverity.CRITICAL, "25% missing in amount"),
        Anomaly("outlier_iqr", AnomalySeverity.WARNING, "5% outliers in amount"),
    ]
    return report


@pytest.fixture
def warning_report():
    report = AnomalyReport(dataset="users", total_checks=3)
    report.anomalies = [
        Anomaly("outlier_zscore", AnomalySeverity.WARNING, "Outliers detected"),
    ]
    return report


@pytest.fixture
def clean_report():
    report = AnomalyReport(dataset="credit_cards", total_checks=4)
    report.anomalies = [
        Anomaly("schema_check", AnomalySeverity.INFO, "All columns present"),
    ]
    return report


# =====================================================================
# Message formatting
# =====================================================================


class TestMessageFormatting:
    def test_format_includes_dataset(self, critical_report):
        msg = _format_anomaly_message(critical_report)
        assert "transactions" in msg

    def test_format_includes_severity_counts(self, critical_report):
        msg = _format_anomaly_message(critical_report)
        assert "CRITICAL: 1" in msg
        assert "WARNING:  1" in msg

    def test_format_includes_anomaly_messages(self, critical_report):
        msg = _format_anomaly_message(critical_report)
        assert "25% missing in amount" in msg

    def test_format_limits_to_5_issues(self):
        report = AnomalyReport(dataset="test", total_checks=1)
        report.anomalies = [
            Anomaly(f"check_{i}", AnomalySeverity.WARNING, f"Issue {i}")
            for i in range(10)
        ]
        msg = _format_anomaly_message(report)
        assert "... and 5 more" in msg


# =====================================================================
# Alert dispatching
# =====================================================================


class TestAlertDispatching:
    def test_critical_report_triggers_alert(
        self, bridge, critical_report, mock_dispatcher
    ):
        result = bridge.process_report(critical_report)
        assert result["alerted"] is True
        mock_dispatcher.dispatch.assert_called_once()

        # Verify severity mapping
        call_kwargs = mock_dispatcher.dispatch.call_args
        assert call_kwargs.kwargs["severity"] == Severity.CRITICAL

    def test_warning_report_triggers_when_enabled(
        self, bridge, warning_report, mock_dispatcher
    ):
        result = bridge.process_report(warning_report, alert_on_warning=True)
        assert result["alerted"] is True
        mock_dispatcher.dispatch.assert_called_once()

    def test_warning_report_skipped_when_disabled(
        self, bridge, warning_report, mock_dispatcher
    ):
        result = bridge.process_report(warning_report, alert_on_warning=False)
        assert result["alerted"] is False
        mock_dispatcher.dispatch.assert_not_called()

    def test_clean_report_no_alert(self, bridge, clean_report, mock_dispatcher):
        result = bridge.process_report(clean_report, alert_on_warning=True)
        assert result["alerted"] is False
        mock_dispatcher.dispatch.assert_not_called()

    def test_result_contains_expected_keys(self, bridge, critical_report):
        result = bridge.process_report(critical_report)
        assert "alerted" in result
        assert "severity" in result
        assert "anomaly_count" in result
        assert "channels" in result
        assert "message" in result

    def test_result_severity_matches_report(self, bridge, critical_report):
        result = bridge.process_report(critical_report)
        assert result["severity"] == "CRITICAL"

    def test_result_anomaly_count(self, bridge, critical_report):
        result = bridge.process_report(critical_report)
        assert result["anomaly_count"] == 2


# =====================================================================
# Multiple reports
# =====================================================================


class TestMultipleReports:
    def test_process_multiple_reports(
        self, bridge, critical_report, warning_report, clean_report
    ):
        results = bridge.process_multiple_reports(
            [critical_report, warning_report, clean_report]
        )
        assert "transactions" in results
        assert "users" in results
        assert "credit_cards" in results
        assert results["transactions"]["alerted"] is True
        assert results["users"]["alerted"] is True
        assert results["credit_cards"]["alerted"] is False

    def test_empty_reports_list(self, bridge):
        results = bridge.process_multiple_reports([])
        assert results == {}


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_report_with_no_anomalies(self, bridge, mock_dispatcher):
        report = AnomalyReport(dataset="test", total_checks=3)
        result = bridge.process_report(report)
        assert result["alerted"] is False
        mock_dispatcher.dispatch.assert_not_called()

    def test_dispatcher_returns_empty_channels(self, bridge, critical_report):
        bridge.dispatcher.dispatch.return_value = {}
        result = bridge.process_report(critical_report)
        assert result["alerted"] is True
        assert result["channels"] == {}

    def test_bridge_without_dispatcher_uses_default(self):
        """Bridge with no args should create a default AlertDispatcher."""
        # This will try to load default config; may log a warning
        bridge = AnomalyAlertBridge()
        assert bridge.dispatcher is not None
