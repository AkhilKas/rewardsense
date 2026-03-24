"""
Notification Dispatcher for Model Pipeline alerts.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

logger = logging.getLogger(__name__)


class NotificationDispatcher:
    def __init__(self, config_path: str = "config/alerts.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Alerts config not found at {path}, using defaults.")
            return {"slack": {"enabled": False}, "email": {"enabled": False}}
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load alerts config: {e}")
            return {"slack": {"enabled": False}, "email": {"enabled": False}}

    def notify(self, message: str, level: str = "INFO") -> None:
        """
        Send a notification to configured channels.
        Supported levels: INFO, WARNING, CRITICAL.
        """
        logger.info(f"[{level}] Notification Dispatch: {message}")

        if self.config.get("slack", {}).get("enabled", False):
            self._send_slack(message, level)

        if self.config.get("email", {}).get("enabled", False):
            self._send_email(message, level)

    def _send_slack(self, message: str, level: str) -> None:
        webhook_var = self.config.get("slack", {}).get(
            "webhook_env_var", "SLACK_WEBHOOK_URL"
        )
        webhook_url = os.getenv(webhook_var)
        if not webhook_url:
            logger.warning(
                "Slack webhook URL not found in environment variable '%s', skipping.",
                webhook_var,
            )
            return

        payload = {"text": f"[RewardSense {level}] {message}"}
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Slack notification sent successfully.")
            else:
                logger.warning(
                    "Slack notification failed: %s %s",
                    response.status_code,
                    response.text,
                )
        except requests.RequestException as e:
            logger.warning("Slack notification request failed: %s", e)

    def _send_email(self, message: str, level: str) -> None:
        # In a real environment, this would use smtplib
        email_config = self.config.get("email", {})
        logger.info(
            f"==> Mocked Email sent from {email_config.get('from_address')}: [{level}] {message}"
        )
