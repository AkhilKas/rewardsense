"""
RewardSense - Alerting Module

Provides Slack and Email alerters plus a unified ``AlertDispatcher``
that reads configuration, determines severity, and dispatches
alerts to the appropriate channels.
"""

from __future__ import annotations

import hashlib
import logging
import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import IntEnum
from pathlib import Path
from typing import Any

import requests
import yaml

logger = logging.getLogger("airflow.task")

DEFAULT_CONFIG_PATH = Path("config/alerting_config.yaml")


def _resolve_alerting_config_path(path: Path) -> Path:
    """Resolve alerting config across local and Composer layouts."""
    candidates = [
        path,
        Path.cwd() / path,
        Path(__file__).resolve().parents[3] / path,
        Path("/home/airflow/gcs/dags") / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _get_config_secret(name: str, default: str = "") -> str:
    """Read secret/config value from env first, then Airflow Variable fallback."""
    value = os.getenv(name)
    if value:
        return value
    try:
        from airflow.models import Variable

        return Variable.get(name, default_var=default) or default
    except Exception:  # noqa: BLE001
        return default


# =========================================================================
# Severity Enum
# =========================================================================


class Severity(IntEnum):
    """Alert severity levels."""

    INFO = 0
    WARNING = 1
    CRITICAL = 2

    @classmethod
    def from_str(cls, value: str) -> "Severity":
        return cls[value.upper()]


# =========================================================================
# Slack Alerter
# =========================================================================


class SlackAlerter:
    """Send alert messages to a Slack channel via Incoming Webhook.

    Parameters
    ----------
    webhook_url : str
        Slack Incoming Webhook URL.
    channel : str | None
        Override channel (optional, normally set in webhook config).
    timeout : int
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        timeout: int = 10,
    ) -> None:
        self.webhook_url = webhook_url
        self.channel = channel
        self.timeout = timeout

    def send(self, message: str, severity: Severity = Severity.WARNING) -> bool:
        """Post *message* to Slack.  Returns ``True`` on success."""
        emoji = {
            Severity.INFO: "ℹ️",
            Severity.WARNING: "⚠️",
            Severity.CRITICAL: "🚨",
        }.get(severity, "📢")

        payload: dict[str, Any] = {
            "text": f"{emoji} *[RewardSense {severity.name}]* {message}",
        }
        if self.channel:
            payload["channel"] = self.channel

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                logger.info("Slack alert sent successfully.")
                return True
            logger.warning("Slack alert failed: %s %s", resp.status_code, resp.text)
        except requests.RequestException as exc:
            logger.error("Slack alert error: %s", exc)
        return False


# =========================================================================
# Email Alerter
# =========================================================================


class EmailAlerter:
    """Send alert emails via SendGrid HTTP API or SMTP.

    If *sendgrid_api_key* is provided, the SendGrid v3 API is used.
    Otherwise, falls back to SMTP (host/port/user/password).
    """

    SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"

    def __init__(
        self,
        recipients: list[str],
        *,
        sendgrid_api_key: str | None = None,
        from_email: str = "avadhani.a@northeastern.edu",
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        timeout: int = 10,
    ) -> None:
        self.recipients = recipients
        self.sendgrid_api_key = sendgrid_api_key
        self.from_email = from_email
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.timeout = timeout

    def send(
        self, subject: str, body: str, severity: Severity = Severity.CRITICAL
    ) -> bool:
        """Send an email alert.  Returns ``True`` on success."""
        tag = f"[RewardSense {severity.name}]"
        full_subject = f"{tag} {subject}"
        if self.sendgrid_api_key:
            return self._send_sendgrid(full_subject, body)
        return self._send_smtp(full_subject, body)

    # ── SendGrid ─────────────────────────────────────────────────────────

    def _send_sendgrid(self, subject: str, body: str) -> bool:
        payload = {
            "personalizations": [{"to": [{"email": r} for r in self.recipients]}],
            "from": {"email": self.from_email},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        }
        headers = {
            "Authorization": f"Bearer {self.sendgrid_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self.SENDGRID_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code in (200, 202):
                logger.info("Email alert sent via SendGrid.")
                return True
            logger.warning("SendGrid email failed: %s %s", resp.status_code, resp.text)
        except requests.RequestException as exc:
            logger.error("SendGrid email error: %s", exc)
        return False

    # ── SMTP fallback ────────────────────────────────────────────────────

    def _send_smtp(self, subject: str, body: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(
                self.smtp_host, self.smtp_port, timeout=self.timeout
            ) as server:
                server.ehlo()
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, self.recipients, msg.as_string())
            logger.info("Email alert sent via SMTP.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("SMTP email error: %s", exc)
        return False


# =========================================================================
# Alert Dispatcher
# =========================================================================


class AlertDispatcher:
    """Route alerts to the appropriate channels based on config.

    Reads ``config/alerting_config.yaml`` (or the path you specify)
    to determine which channels are enabled and their minimum severity.
    Supports throttling to avoid alert storms.

    Parameters
    ----------
    config_path : Path | str
        Path to the alerting YAML config.
    """

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        self.config = self._load_config(Path(config_path))
        self._recent_hashes: dict[str, float] = {}  # hash -> unix ts

        alerting_cfg = self.config.get("alerting", {})
        self.enabled = alerting_cfg.get("enabled", True)
        self.throttle_cooldown = alerting_cfg.get("throttle", {}).get(
            "cooldown_seconds", 300
        )
        self.dedup_window = alerting_cfg.get("throttle", {}).get(
            "dedup_window_seconds", 3600
        )

        # Initialise channel alerters lazily
        self._slack: SlackAlerter | None = None
        self._email: EmailAlerter | None = None
        self._init_channels(alerting_cfg.get("channels", {}))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(
        self,
        message: str,
        severity: Severity = Severity.WARNING,
        subject: str = "Pipeline Alert",
    ) -> dict[str, bool]:
        """Dispatch an alert to all enabled channels that meet severity threshold.

        Returns a dict of ``{channel: success_bool}``.
        """
        if not self.enabled:
            logger.info("Alerting is disabled — skipping dispatch.")
            return {}

        if self._is_duplicate(message, severity):
            logger.info("Alert throttled (duplicate within dedup window).")
            return {}

        results: dict[str, bool] = {}
        channels = self.config.get("alerting", {}).get("channels", {})

        # Slack
        if self._slack and channels.get("slack", {}).get("enabled", False):
            min_sev = Severity.from_str(
                channels["slack"].get("severity_min", "WARNING")
            )
            if severity >= min_sev:
                results["slack"] = self._slack.send(message, severity)

        # Email
        if self._email and channels.get("email", {}).get("enabled", False):
            min_sev = Severity.from_str(
                channels["email"].get("severity_min", "CRITICAL")
            )
            if severity >= min_sev:
                results["email"] = self._email.send(subject, message, severity)

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        path = _resolve_alerting_config_path(path)
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        logger.warning("Alerting config not found at %s — using defaults.", path)
        return {"alerting": {"enabled": False}}

    def _init_channels(self, channels: dict[str, Any]) -> None:
        # Slack
        slack_cfg = channels.get("slack", {})
        if slack_cfg.get("enabled", False):
            webhook = _get_config_secret(
                slack_cfg.get("webhook_url_env", "SLACK_WEBHOOK_URL"), ""
            )
            channel = _get_config_secret(
                slack_cfg.get("channel_env", "SLACK_CHANNEL"), ""
            )
            if webhook:
                self._slack = SlackAlerter(webhook_url=webhook, channel=channel)
            else:
                logger.warning("Slack enabled but SLACK_WEBHOOK_URL not set.")

        # Email
        email_cfg = channels.get("email", {})
        if email_cfg.get("enabled", False):
            api_key = _get_config_secret(
                email_cfg.get("api_key_env", "SENDGRID_API_KEY"), ""
            )
            recipients_raw = _get_config_secret(
                email_cfg.get("recipients_env", "ALERT_EMAIL"), ""
            )
            recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
            if recipients:
                self._email = EmailAlerter(
                    recipients=recipients,
                    sendgrid_api_key=api_key or None,
                )
            else:
                logger.warning("Email enabled but ALERT_EMAIL not set.")

    def _is_duplicate(self, message: str, severity: Severity) -> bool:
        """Check if this alert was already sent within the dedup window."""
        h = hashlib.md5(f"{severity.name}:{message}".encode()).hexdigest()  # noqa: S324
        now = time.time()

        # Purge stale hashes
        self._recent_hashes = {
            k: v for k, v in self._recent_hashes.items() if now - v < self.dedup_window
        }

        if h in self._recent_hashes:
            return True
        self._recent_hashes[h] = now
        return False
