"""
RewardSense - Monitoring Module Unit Tests

Tests for:
    - PipelineReportGenerator
    - SlackAlerter
    - EmailAlerter
    - AlertDispatcher (severity routing, throttling)
    - PipelineMetricsLogger
    - Airflow callbacks

Run with:
    pytest tests/data_pipeline/monitoring/test_monitoring.py -v
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_pipeline.monitoring.alerting import (
    AlertDispatcher,
    EmailAlerter,
    Severity,
    SlackAlerter,
)
from data_pipeline.monitoring.callbacks import (
    on_dag_success_callback,
    on_task_failure_callback,
    on_task_success_callback,
)
from data_pipeline.monitoring.metrics import PipelineMetricsLogger
from data_pipeline.monitoring.pipeline_report import PipelineReportGenerator

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_reports_dir(tmp_path):
    """Provides a temporary directory for pipeline reports."""
    return tmp_path / "reports"


@pytest.fixture
def tmp_metrics_dir(tmp_path):
    """Provides a temporary directory for pipeline metrics."""
    return tmp_path / "metrics"


@pytest.fixture
def mock_dag_run():
    """Provides a mock Airflow DagRun with two task instances."""
    dag_run = MagicMock()
    dag_run.dag_id = "rewardsense_data_pipeline"
    dag_run.run_id = "manual__2025-06-01T00:00:00"
    dag_run.execution_date = datetime(2025, 6, 1, tzinfo=timezone.utc)
    dag_run.state = "success"
    dag_run.start_date = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)
    dag_run.end_date = datetime(2025, 6, 1, 6, 30, 0, tzinfo=timezone.utc)

    dag = MagicMock()
    dag.task_ids = [
        "ingestion.scrape_nerdwallet",
        "preprocessing.clean_data",
        "reporting.generate_pipeline_report",
    ]
    dag_run.dag = dag

    ti1 = MagicMock()
    ti1.task_id = "ingestion.scrape_nerdwallet"
    ti1.state = "success"
    ti1.start_date = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)
    ti1.end_date = datetime(2025, 6, 1, 6, 5, 0, tzinfo=timezone.utc)

    ti2 = MagicMock()
    ti2.task_id = "preprocessing.clean_data"
    ti2.state = "success"
    ti2.start_date = datetime(2025, 6, 1, 6, 5, 0, tzinfo=timezone.utc)
    ti2.end_date = datetime(2025, 6, 1, 6, 15, 0, tzinfo=timezone.utc)

    dag_run.get_task_instances.return_value = [ti1, ti2]
    return dag_run


@pytest.fixture
def mock_task_instance():
    """Provides a mock Airflow TaskInstance with xcom_pull."""
    ti = MagicMock()
    ti.task_id = "reporting.generate_pipeline_report"
    ti.dag_id = "rewardsense_data_pipeline"

    def _xcom_pull(task_ids=None, dag_id=None):
        data = {
            "ingestion.scrape_nerdwallet": {"source": "nerdwallet", "cards_found": 45},
            "preprocessing.clean_data": {"datasets_cleaned": 3},
        }
        return data.get(task_ids)

    ti.xcom_pull = _xcom_pull
    return ti


@pytest.fixture
def alerting_config_path(tmp_path):
    """Provides a temporary alerting config YAML file."""
    cfg = {
        "alerting": {
            "enabled": True,
            "channels": {
                "slack": {
                    "enabled": True,
                    "webhook_url_env": "TEST_SLACK_WEBHOOK",
                    "channel_env": "TEST_SLACK_CHANNEL",
                    "severity_min": "WARNING",
                },
                "email": {
                    "enabled": True,
                    "api_key_env": "TEST_SENDGRID_KEY",
                    "recipients_env": "TEST_ALERT_EMAIL",
                    "severity_min": "CRITICAL",
                },
            },
            "throttle": {
                "cooldown_seconds": 1,
                "dedup_window_seconds": 2,
            },
        }
    }
    config_file = tmp_path / "alerting_config.yaml"
    import yaml

    config_file.write_text(yaml.dump(cfg))
    return config_file


# =============================================================================
# PipelineReportGenerator Tests
# =============================================================================


class TestPipelineReportGenerator:
    """Tests for the pipeline report generator."""

    def test_generate_creates_report_file(
        self, tmp_reports_dir, mock_dag_run, mock_task_instance
    ):
        """
        Given: A PipelineReportGenerator and a valid Airflow context
        When: generate() is called
        Then: A JSON report file should be created on disk
        """
        # Given
        gen = PipelineReportGenerator(reports_dir=tmp_reports_dir)
        context = {"dag_run": mock_dag_run, "ti": mock_task_instance}

        # When
        result = gen.generate(context)

        # Then
        assert result["status"] == "completed"
        assert Path(result["report_path"]).exists()

    def test_report_contains_timing_stats(
        self, tmp_reports_dir, mock_dag_run, mock_task_instance
    ):
        """
        Given: A DAG run lasting 30 minutes
        When: A report is generated
        Then: Timing data should be present with correct total duration
        """
        # Given
        gen = PipelineReportGenerator(reports_dir=tmp_reports_dir)
        context = {"dag_run": mock_dag_run, "ti": mock_task_instance}

        # When
        result = gen.generate(context)
        report = json.loads(Path(result["report_path"]).read_text())

        # Then
        assert "timing" in report
        assert report["timing"]["total_duration_sec"] == 1800.0

    def test_report_collects_upstream_xcoms(
        self, tmp_reports_dir, mock_dag_run, mock_task_instance
    ):
        """
        Given: Upstream tasks that pushed XCom values
        When: A report is generated
        Then: Those XCom values should appear in the report's task_results
        """
        # Given
        gen = PipelineReportGenerator(reports_dir=tmp_reports_dir)
        context = {"dag_run": mock_dag_run, "ti": mock_task_instance}

        # When
        result = gen.generate(context)
        report = json.loads(Path(result["report_path"]).read_text())

        # Then
        assert "ingestion.scrape_nerdwallet" in report["task_results"]
        assert (
            report["task_results"]["ingestion.scrape_nerdwallet"]["cards_found"] == 45
        )

    def test_generate_handles_none_dag_run(self, tmp_reports_dir):
        """
        Given: A None dag_run (e.g. dry-run context)
        When: generate() is called
        Then: A report should still be created with 'unknown' identifiers
        """
        # Given
        gen = PipelineReportGenerator(reports_dir=tmp_reports_dir)
        context = {"dag_run": None, "ti": None}

        # When
        result = gen.generate(context)

        # Then
        assert result["status"] == "completed"
        assert result["dag_run_id"] == "unknown"


# =============================================================================
# SlackAlerter Tests
# =============================================================================


class TestSlackAlerter:
    """Tests for the Slack alerter."""

    @patch("data_pipeline.monitoring.alerting.requests.post")
    def test_send_returns_true_on_success(self, mock_post):
        """
        Given: A valid Slack webhook URL
        When: send() is called and Slack responds with 200
        Then: The method should return True
        """
        # Given
        mock_post.return_value = MagicMock(status_code=200)
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")

        # When
        result = alerter.send("Test alert", Severity.WARNING)

        # Then
        assert result is True
        mock_post.assert_called_once()

    @patch("data_pipeline.monitoring.alerting.requests.post")
    def test_send_includes_severity_in_payload(self, mock_post):
        """
        Given: A WARNING severity alert
        When: send() is called
        Then: The payload text should contain 'RewardSense WARNING'
        """
        # Given
        mock_post.return_value = MagicMock(status_code=200)
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")

        # When
        alerter.send("Test alert", Severity.WARNING)

        # Then
        payload = mock_post.call_args.kwargs["json"]
        assert "RewardSense WARNING" in payload["text"]

    @patch("data_pipeline.monitoring.alerting.requests.post")
    def test_send_returns_false_on_error_status(self, mock_post):
        """
        Given: A Slack webhook that responds with 500
        When: send() is called
        Then: The method should return False
        """
        # Given
        mock_post.return_value = MagicMock(status_code=500, text="error")
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")

        # When
        result = alerter.send("Test alert")

        # Then
        assert result is False

    @patch("data_pipeline.monitoring.alerting.requests.post")
    def test_send_returns_false_on_network_error(self, mock_post):
        """
        Given: A network exception during the HTTP call
        When: send() is called
        Then: The method should return False gracefully
        """
        # Given
        import requests

        mock_post.side_effect = requests.RequestException("timeout")
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")

        # When
        result = alerter.send("Test alert")

        # Then
        assert result is False

    @patch("data_pipeline.monitoring.alerting.requests.post")
    def test_channel_override_in_payload(self, mock_post):
        """
        Given: A SlackAlerter with a channel override
        When: send() is called
        Then: The payload should include the custom channel
        """
        # Given
        mock_post.return_value = MagicMock(status_code=200)
        alerter = SlackAlerter(
            webhook_url="https://hooks.slack.com/test",
            channel="#custom-channel",
        )

        # When
        alerter.send("msg")

        # Then
        payload = mock_post.call_args.kwargs["json"]
        assert payload["channel"] == "#custom-channel"


# =============================================================================
# EmailAlerter Tests
# =============================================================================


class TestEmailAlerter:
    """Tests for the email alerter."""

    @patch("data_pipeline.monitoring.alerting.requests.post")
    def test_sendgrid_returns_true_on_success(self, mock_post):
        """
        Given: A valid SendGrid API key and recipient list
        When: send() is called and SendGrid responds with 202
        Then: The method should return True
        """
        # Given
        mock_post.return_value = MagicMock(status_code=202)
        alerter = EmailAlerter(
            recipients=["test@example.com"],
            sendgrid_api_key="SG.test_key",
        )

        # When
        result = alerter.send("Test Subject", "Test body")

        # Then
        assert result is True
        mock_post.assert_called_once()

    @patch("data_pipeline.monitoring.alerting.requests.post")
    def test_sendgrid_uses_bearer_auth(self, mock_post):
        """
        Given: An EmailAlerter configured with SendGrid
        When: send() is called
        Then: The Authorization header should be 'Bearer <key>'
        """
        # Given
        mock_post.return_value = MagicMock(status_code=202)
        alerter = EmailAlerter(
            recipients=["test@example.com"],
            sendgrid_api_key="SG.test_key",
        )

        # When
        alerter.send("Subject", "Body")

        # Then
        auth_header = mock_post.call_args.kwargs["headers"]["Authorization"]
        assert "Bearer SG.test_key" in auth_header

    @patch("data_pipeline.monitoring.alerting.requests.post")
    def test_sendgrid_returns_false_on_failure(self, mock_post):
        """
        Given: A SendGrid API that responds with 400
        When: send() is called
        Then: The method should return False
        """
        # Given
        mock_post.return_value = MagicMock(status_code=400, text="bad request")
        alerter = EmailAlerter(
            recipients=["test@example.com"],
            sendgrid_api_key="SG.test_key",
        )

        # When
        result = alerter.send("Subject", "Body")

        # Then
        assert result is False

    @patch("data_pipeline.monitoring.alerting.smtplib.SMTP")
    def test_smtp_fallback_when_no_sendgrid_key(self, mock_smtp):
        """
        Given: No SendGrid API key configured
        When: send() is called
        Then: It should fall back to SMTP and use starttls + login
        """
        # Given
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        alerter = EmailAlerter(
            recipients=["test@example.com"],
            smtp_user="user",
            smtp_password="pass",
        )

        # When
        result = alerter.send("Subject", "Body")

        # Then
        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")


# =============================================================================
# AlertDispatcher Tests
# =============================================================================


class TestAlertDispatcher:
    """Tests for alert severity routing, throttling, and deduplication."""

    def test_warning_reaches_slack_but_not_email(self, alerting_config_path):
        """
        Given: Slack min severity = WARNING, Email min severity = CRITICAL
        When: A WARNING alert is dispatched
        Then: Only Slack should receive the alert
        """
        # Given
        with patch.dict(
            os.environ,
            {
                "TEST_SLACK_WEBHOOK": "https://hooks.slack.com/test",
                "TEST_SLACK_CHANNEL": "#test",
                "TEST_SENDGRID_KEY": "SG.key",
                "TEST_ALERT_EMAIL": "test@example.com",
            },
        ):
            dispatcher = AlertDispatcher(config_path=alerting_config_path)

            # When
            with patch.object(
                dispatcher._slack, "send", return_value=True
            ) as slack_send:
                results = dispatcher.dispatch("warn msg", Severity.WARNING)

            # Then
            assert results.get("slack") is True
            assert "email" not in results
            slack_send.assert_called_once()

    def test_critical_reaches_both_channels(self, alerting_config_path):
        """
        Given: Both Slack and Email channels are enabled
        When: A CRITICAL alert is dispatched
        Then: Both channels should receive the alert
        """
        # Given
        with patch.dict(
            os.environ,
            {
                "TEST_SLACK_WEBHOOK": "https://hooks.slack.com/test",
                "TEST_SLACK_CHANNEL": "#test",
                "TEST_SENDGRID_KEY": "SG.key",
                "TEST_ALERT_EMAIL": "test@example.com",
            },
        ):
            dispatcher = AlertDispatcher(config_path=alerting_config_path)

            # When
            with patch.object(
                dispatcher._slack, "send", return_value=True
            ), patch.object(dispatcher._email, "send", return_value=True):
                results = dispatcher.dispatch("crit msg", Severity.CRITICAL)

            # Then
            assert results.get("slack") is True
            assert results.get("email") is True

    def test_no_alerts_when_disabled(self, alerting_config_path):
        """
        Given: Alerting is globally disabled
        When: A CRITICAL alert is dispatched
        Then: No alerts should be sent and result should be empty
        """
        # Given
        with patch.dict(
            os.environ,
            {
                "TEST_SLACK_WEBHOOK": "https://hooks.slack.com/test",
                "TEST_SLACK_CHANNEL": "#test",
                "TEST_SENDGRID_KEY": "",
                "TEST_ALERT_EMAIL": "",
            },
        ):
            dispatcher = AlertDispatcher(config_path=alerting_config_path)
            dispatcher.enabled = False

            # When
            results = dispatcher.dispatch("msg", Severity.CRITICAL)

            # Then
            assert results == {}

    def test_duplicate_alerts_are_suppressed(self, alerting_config_path):
        """
        Given: A dispatcher that already sent an alert with the same message
        When: The same alert is dispatched again within the dedup window
        Then: The second dispatch should return empty (throttled)
        """
        # Given
        with patch.dict(
            os.environ,
            {
                "TEST_SLACK_WEBHOOK": "https://hooks.slack.com/test",
                "TEST_SLACK_CHANNEL": "#test",
                "TEST_SENDGRID_KEY": "",
                "TEST_ALERT_EMAIL": "",
            },
        ):
            dispatcher = AlertDispatcher(config_path=alerting_config_path)

            with patch.object(dispatcher._slack, "send", return_value=True):
                # When
                r1 = dispatcher.dispatch("same msg", Severity.WARNING)
                r2 = dispatcher.dispatch("same msg", Severity.WARNING)

            # Then
            assert r1.get("slack") is True
            assert r2 == {}

    def test_missing_config_disables_alerting(self, tmp_path):
        """
        Given: A config path pointing to a non-existent file
        When: AlertDispatcher is initialized
        Then: Alerting should be disabled gracefully
        """
        # Given
        bad_path = tmp_path / "nonexistent.yaml"

        # When
        dispatcher = AlertDispatcher(config_path=bad_path)

        # Then
        assert dispatcher.enabled is False


# =============================================================================
# Severity Tests
# =============================================================================


class TestSeverity:
    """Tests for the Severity enum ordering and parsing."""

    def test_severity_ordering(self):
        """
        Given: The three severity levels
        When: Compared with less-than operators
        Then: INFO < WARNING < CRITICAL
        """
        assert Severity.INFO < Severity.WARNING < Severity.CRITICAL

    def test_from_str_parses_lowercase(self):
        """
        Given: A lowercase severity string like 'warning'
        When: from_str() is called
        Then: The correct Severity enum member should be returned
        """
        # When / Then
        assert Severity.from_str("warning") == Severity.WARNING
        assert Severity.from_str("CRITICAL") == Severity.CRITICAL

    def test_from_str_raises_on_invalid(self):
        """
        Given: An invalid severity string
        When: from_str() is called
        Then: A KeyError should be raised
        """
        with pytest.raises(KeyError):
            Severity.from_str("nonexistent")


# =============================================================================
# PipelineMetricsLogger Tests
# =============================================================================


class TestPipelineMetricsLogger:
    """Tests for pipeline metrics collection and persistence."""

    def test_log_metrics_creates_file(self, tmp_metrics_dir, mock_dag_run):
        """
        Given: A valid DAG run with task instances
        When: log_metrics() is called
        Then: A metrics JSON file should be created on disk
        """
        # Given
        logger = PipelineMetricsLogger(metrics_dir=tmp_metrics_dir)
        context = {"dag_run": mock_dag_run}

        # When
        result = logger.log_metrics(context)

        # Then
        assert result["status"] == "completed"
        assert Path(result["metrics_path"]).exists()

    def test_metrics_contain_per_task_info(self, tmp_metrics_dir, mock_dag_run):
        """
        Given: Two successful task instances in the DAG run
        When: log_metrics() is called
        Then: Metrics should report 2 total tasks, 0 failed
        """
        # Given
        logger = PipelineMetricsLogger(metrics_dir=tmp_metrics_dir)
        context = {"dag_run": mock_dag_run}

        # When
        result = logger.log_metrics(context)
        metrics = json.loads(Path(result["metrics_path"]).read_text())

        # Then
        assert metrics["total_tasks"] == 2
        assert metrics["failed_tasks"] == 0
        assert len(metrics["tasks"]) == 2

    def test_log_metrics_handles_none_dag_run(self, tmp_metrics_dir):
        """
        Given: A None dag_run
        When: log_metrics() is called
        Then: Should still succeed with zero counts
        """
        # Given
        logger = PipelineMetricsLogger(metrics_dir=tmp_metrics_dir)
        context = {"dag_run": None}

        # When
        result = logger.log_metrics(context)

        # Then
        assert result["status"] == "completed"
        assert result["total_tasks"] == 0


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for Airflow on_failure / on_success / on_dag_success callbacks."""

    @patch("data_pipeline.monitoring.callbacks._get_dispatcher")
    def test_on_task_failure_dispatches_critical(self, mock_get_disp):
        """
        Given: A task failure context with exception details
        When: on_task_failure_callback() is invoked
        Then: A CRITICAL alert mentioning the failed task should be dispatched
        """
        # Given
        mock_dispatcher = MagicMock()
        mock_get_disp.return_value = mock_dispatcher
        ti = MagicMock()
        ti.task_id = "ingestion.scrape_nerdwallet"
        ti.dag_id = "rewardsense_data_pipeline"
        context = {
            "ti": ti,
            "execution_date": "2025-06-01",
            "exception": ValueError("test error"),
        }

        # When
        on_task_failure_callback(context)

        # Then
        mock_dispatcher.dispatch.assert_called_once()
        call_kwargs = mock_dispatcher.dispatch.call_args.kwargs
        assert call_kwargs["severity"] == Severity.CRITICAL
        assert "scrape_nerdwallet" in call_kwargs["message"]

    def test_on_task_success_does_not_raise(self):
        """
        Given: A successful task with start and end dates
        When: on_task_success_callback() is invoked
        Then: It should log without raising any exceptions
        """
        # Given
        ti = MagicMock()
        ti.task_id = "preprocessing.clean_data"
        ti.start_date = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)
        ti.end_date = datetime(2025, 6, 1, 6, 5, 0, tzinfo=timezone.utc)
        context = {"ti": ti}

        # When / Then — should not raise
        on_task_success_callback(context)

    @patch("data_pipeline.monitoring.callbacks._get_dispatcher")
    def test_on_dag_success_dispatches_info(self, mock_get_disp):
        """
        Given: A completed DAG run
        When: on_dag_success_callback() is invoked
        Then: An INFO summary alert should be dispatched
        """
        # Given
        mock_dispatcher = MagicMock()
        mock_get_disp.return_value = mock_dispatcher
        dag_run = MagicMock()
        dag_run.dag_id = "rewardsense_data_pipeline"
        dag_run.run_id = "manual__2025-06-01"
        dag_run.start_date = datetime(2025, 6, 1, 6, 0, 0, tzinfo=timezone.utc)
        dag_run.end_date = datetime(2025, 6, 1, 6, 30, 0, tzinfo=timezone.utc)
        context = {"dag_run": dag_run}

        # When
        on_dag_success_callback(context)

        # Then
        mock_dispatcher.dispatch.assert_called_once()
        call_kwargs = mock_dispatcher.dispatch.call_args.kwargs
        assert call_kwargs["severity"] == Severity.INFO
