#!/usr/bin/env python3
"""
Quick smoke test — sends a test alert to Slack and Email
to verify credentials are configured correctly.

Usage:
    python scripts/test_alerts.py           # test both channels
    python scripts/test_alerts.py slack     # test Slack only
    python scripts/test_alerts.py email     # test Email only
"""

import sys
from pathlib import Path

# Load .env before anything else
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from data_pipeline.monitoring.alerting import (  # noqa: E402
    EmailAlerter,
    Severity,
    SlackAlerter,
)
import os  # noqa: E402


def test_slack():
    """Send a test message to Slack."""
    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    channel = os.getenv("SLACK_CHANNEL", "")

    if not webhook or "YOUR" in webhook:
        print("❌ SLACK_WEBHOOK_URL not configured in .env")
        return False

    print(f"📤 Sending test alert to Slack (channel: {channel})...")
    alerter = SlackAlerter(webhook_url=webhook, channel=channel)
    ok = alerter.send(
        "🧪 This is a test alert from RewardSense. If you see this, Slack alerting is working!",
        severity=Severity.INFO,
    )
    print(f"   {'✅ Slack alert sent!' if ok else '❌ Slack alert failed.'}")
    return ok


def test_email():
    """Send a test email via SendGrid."""
    api_key = os.getenv("SENDGRID_API_KEY", "")
    recipients_raw = os.getenv("ALERT_EMAIL", "")
    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    if not api_key or api_key == "your-sendgrid-api-key":
        print("❌ SENDGRID_API_KEY not configured in .env")
        return False
    if not recipients:
        print("❌ ALERT_EMAIL not configured in .env")
        return False

    print(f"📤 Sending test email to {recipients}...")
    alerter = EmailAlerter(recipients=recipients, sendgrid_api_key=api_key)
    ok = alerter.send(
        subject="RewardSense Test Alert",
        body="🧪 This is a test alert from RewardSense.\n\nIf you see this, email alerting is working!",
        severity=Severity.INFO,
    )
    print(f"   {'✅ Email sent!' if ok else '❌ Email failed.'}")
    return ok


if __name__ == "__main__":
    channel = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    print("=" * 50)
    print("  RewardSense Alert Test")
    print("=" * 50)

    results = {}
    if channel in ("both", "slack"):
        results["slack"] = test_slack()
    if channel in ("both", "email"):
        results["email"] = test_email()

    print()
    all_ok = all(results.values())
    if all_ok:
        print("✅ All alerts sent successfully!")
    else:
        print("⚠️  Some alerts failed — check the output above.")
