"""
Monitoring Notification System

Sends structured Slack webhook notifications (and optional email) for monitoring events, retraining triggers, and redeployments.

Webhook URL is read from env var SLACK_WEBHOOK_URL (never in code).

Usage:
    notifier = SlackNotifier()
    notifier.send_monitoring_summary(drift_result, perf_snapshot)
    notifier.send_retrain_trigger(reason, drift_report_path)
    notifier.send_redeployment(new_version, old_version, metrics)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import requests as http_requests

    REQUESTS_AVAILABLE = True
except ImportError:
    http_requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False

SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
NOTIFICATION_CHANNEL: str = os.getenv("NOTIFICATION_CHANNEL", "#rewardsense-monitoring")


class SlackNotifier:
    """Send structured Slack notifications for monitoring events.

    Parameters
    ----------
    webhook_url : str, optional
        Slack incoming webhook URL. Defaults to SLACK_WEBHOOK_URL env var.
    channel : str
        Slack channel override.
    dry_run : bool
        If True, log messages instead of sending (for testing).
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        channel: str = NOTIFICATION_CHANNEL,
        dry_run: bool = False,
    ) -> None:
        self.webhook_url = webhook_url or SLACK_WEBHOOK_URL
        self.channel = channel
        self.dry_run = dry_run

    @property
    def is_configured(self) -> bool:
        return bool(self.webhook_url) and REQUESTS_AVAILABLE

    def _send(self, payload: Dict[str, Any]) -> bool:
        """Send a payload to the Slack webhook. Fire-and-forget with retry."""
        if self.dry_run:
            logger.info("[DRY RUN] Slack payload: %s", json.dumps(payload, indent=2))
            return True

        if not self.is_configured:
            logger.warning("Slack webhook not configured, skipping notification")
            return False

        try:
            resp = http_requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Slack notification sent successfully")
                return True
            logger.warning("Slack webhook returned %d: %s", resp.status_code, resp.text)
            return False
        except Exception as e:
            logger.warning("Slack notification failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Monitoring Summary (daily)
    # ------------------------------------------------------------------

    def send_monitoring_summary(
        self,
        drift_result: Any,
        perf_snapshot: Any,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send daily monitoring summary to Slack."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        drift_summary = drift_result.summary if hasattr(drift_result, "summary") else {}
        perf_dict = perf_snapshot.to_dict() if hasattr(perf_snapshot, "to_dict") else {}
        latency = perf_dict.get("latency", {})
        alerts = perf_dict.get("alerts", [])

        drift_emoji = (
            ":red_circle:"
            if drift_summary.get("drift_detected")
            else ":large_green_circle:"
        )
        alert_emoji = ":warning:" if alerts else ":white_check_mark:"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"RewardSense Monitoring Summary - {ts}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"{drift_emoji} *Drift Detection*\n"
                            f"Detected: {drift_summary.get('drift_detected', 'N/A')}\n"
                            f"Drifted features: {drift_summary.get('n_drifted_features', 0)}\n"
                            f"Drift share: {drift_summary.get('dataset_drift_share', 0):.1%}"
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"{alert_emoji} *Serving Health*\n"
                            f"Requests: {perf_dict.get('total_requests', 0)}\n"
                            f"p95 latency: {latency.get('p95_ms', 0):.0f}ms\n"
                            f"Personalization: {perf_dict.get('personalization_rate', 0):.1%}"
                        ),
                    },
                ],
            },
        ]

        if alerts:
            alert_text = "\n".join(f":warning: {a}" for a in alerts)
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Alerts:*\n{alert_text}"},
                }
            )

        if drift_summary.get("drifted_features"):
            features = ", ".join(drift_summary["drifted_features"][:5])
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Top drifted features: {features}"},
                    ],
                }
            )

        payload = {"channel": self.channel, "blocks": blocks}
        return self._send(payload)

    # ------------------------------------------------------------------
    # Retrain Trigger
    # ------------------------------------------------------------------

    def send_retrain_trigger(
        self,
        reason: str,
        drift_report_path: Optional[str] = None,
        threshold_values: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Notify that a retrain has been triggered."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":arrows_counterclockwise: RewardSense Retrain Triggered",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Reason:* {reason}"},
                    {"type": "mrkdwn", "text": f"*Timestamp:* {ts}"},
                ],
            },
        ]

        if drift_report_path:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Drift report: `{drift_report_path}`",
                        },
                    ],
                }
            )

        if threshold_values:
            threshold_text = "\n".join(
                f"  {k}: {v}" for k, v in threshold_values.items()
            )
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Threshold values:*\n```{threshold_text}```",
                    },
                }
            )

        payload = {"channel": self.channel, "blocks": blocks}
        return self._send(payload)

    # ------------------------------------------------------------------
    # Redeployment
    # ------------------------------------------------------------------

    def send_redeployment(
        self,
        new_version: str,
        old_version: str,
        performance_comparison: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Notify that a new model has been deployed."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":rocket: RewardSense Model Redeployed",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*New version:* {new_version}"},
                    {"type": "mrkdwn", "text": f"*Previous version:* {old_version}"},
                    {"type": "mrkdwn", "text": f"*Timestamp:* {ts}"},
                ],
            },
        ]

        if performance_comparison:
            comp_text = "\n".join(
                f"  {k}: {v}" for k, v in performance_comparison.items()
            )
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Performance comparison:*\n```{comp_text}```",
                    },
                }
            )

        payload = {"channel": self.channel, "blocks": blocks}
        return self._send(payload)
