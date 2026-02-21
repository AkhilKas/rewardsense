"""
Anomaly Alert Bridge

Connects anomaly detection results to the existing AlertDispatcher .

Translates AnomalySeverity → alerting.Severity, aggregates anomalies
into alert messages, and dispatches via configured channels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_pipeline.anomaly_detection.detectors import (
    AnomalyReport,
    AnomalySeverity,
)
from src.data_pipeline.monitoring.alerting import AlertDispatcher, Severity

logger = logging.getLogger(__name__)


# Mapping from anomaly severity to alerting severity
_SEVERITY_MAP = {
    AnomalySeverity.INFO: Severity.INFO,
    AnomalySeverity.WARNING: Severity.WARNING,
    AnomalySeverity.CRITICAL: Severity.CRITICAL,
}


def _format_anomaly_message(report: AnomalyReport) -> str:
    """Format an AnomalyReport into a human-readable alert message."""
    summary = report.summary
    lines = [
        f"Dataset: {report.dataset}",
        f"Checks run: {report.total_checks}",
        f"Anomalies found: {len(report.anomalies)}",
        f"  CRITICAL: {summary.get('CRITICAL', 0)}",
        f"  WARNING:  {summary.get('WARNING', 0)}",
        f"  INFO:     {summary.get('INFO', 0)}",
    ]

    # Include top anomaly details (limit to 5 to avoid message bloat)
    critical_and_warning = [
        a for a in report.anomalies if a.severity >= AnomalySeverity.WARNING
    ]
    if critical_and_warning:
        lines.append("")
        lines.append("Top issues:")
        for a in critical_and_warning[:5]:
            lines.append(f"  [{a.severity.name}] {a.message}")

    if len(critical_and_warning) > 5:
        lines.append(f"  ... and {len(critical_and_warning) - 5} more")

    return "\n".join(lines)


class AnomalyAlertBridge:
    """Bridges anomaly detection reports to the AlertDispatcher.

    Parameters
    ----------
    dispatcher : AlertDispatcher, optional
        Pre-configured dispatcher. If None, creates one from default config.
    config_path : Path or str, optional
        Path to alerting config YAML. Used if dispatcher is None.
    """

    def __init__(
        self,
        dispatcher: Optional[AlertDispatcher] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        if dispatcher is not None:
            self.dispatcher = dispatcher
        elif config_path is not None:
            self.dispatcher = AlertDispatcher(config_path=Path(config_path))
        else:
            self.dispatcher = AlertDispatcher()

    def process_report(
        self,
        report: AnomalyReport,
        alert_on_warning: bool = True,
    ) -> Dict[str, Any]:
        """Process an anomaly report and dispatch alerts if needed.

        Parameters
        ----------
        report : AnomalyReport
            The detection results.
        alert_on_warning : bool
            If True, dispatch alerts for WARNING-level anomalies too.
            If False, only dispatch for CRITICAL.

        Returns
        -------
        dict with keys: alerted (bool), severity, channels, message
        """
        result: Dict[str, Any] = {
            "alerted": False,
            "severity": report.max_severity.name,
            "anomaly_count": len(report.anomalies),
            "channels": {},
            "message": "",
        }

        # Determine if we should alert
        should_alert = False
        if report.has_critical:
            should_alert = True
        elif alert_on_warning and report.has_warnings:
            should_alert = True

        if not should_alert:
            logger.info(
                "[%s] No alerts needed (%d anomalies, max severity: %s)",
                report.dataset,
                len(report.anomalies),
                report.max_severity.name,
            )
            return result

        # Build message and dispatch
        message = _format_anomaly_message(report)
        alert_severity = _SEVERITY_MAP.get(report.max_severity, Severity.WARNING)
        subject = f"Anomaly Alert: {report.dataset} ({report.max_severity.name})"

        channels = self.dispatcher.dispatch(
            message=message,
            severity=alert_severity,
            subject=subject,
        )

        result["alerted"] = True
        result["channels"] = channels
        result["message"] = message

        logger.info(
            "[%s] Alert dispatched (severity=%s, channels=%s)",
            report.dataset,
            alert_severity.name,
            channels,
        )
        return result

    def process_multiple_reports(
        self,
        reports: List[AnomalyReport],
        alert_on_warning: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Process multiple reports (one per dataset).

        Returns a dict keyed by dataset name.
        """
        results = {}
        for report in reports:
            results[report.dataset] = self.process_report(report, alert_on_warning)
        return results
