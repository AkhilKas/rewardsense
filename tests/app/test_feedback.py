"""Tests for Story 4.1: Feedback capture."""

from __future__ import annotations

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("DB_PATH", ":memory:")

import pytest  # noqa: E402

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("passlib")
pytest.importorskip("jose")


def _signup_and_login(client) -> dict:
    """Create a user and return auth headers."""
    resp = client.post(
        "/auth/signup",
        json={
            "email": "fb@test.com",
            "password": "testpass123",
            "display_name": "FB Tester",
        },
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_submit_feedback_like(test_client):
    headers = _signup_and_login(test_client)
    resp = test_client.post(
        "/feedback",
        json={
            "card_id": "amex_gold",
            "reaction": "like",
            "target": "card",
        },
        headers=headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["ok"] is True
    assert body["feedback_id"] > 0


def test_submit_feedback_dislike_with_reason(test_client):
    headers = _signup_and_login(test_client)
    resp = test_client.post(
        "/feedback",
        json={
            "card_id": "chase_sapphire",
            "reaction": "dislike",
            "reason_tag": "too_expensive",
            "target": "explanation",
        },
        headers=headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["ok"] is True


def test_submit_feedback_invalid_reason_tag(test_client):
    headers = _signup_and_login(test_client)
    resp = test_client.post(
        "/feedback",
        json={
            "card_id": "amex_gold",
            "reaction": "dislike",
            "reason_tag": "invalid_tag",
            "target": "card",
        },
        headers=headers,
    )
    assert resp.status_code == 422


def test_submit_feedback_requires_auth(test_client):
    resp = test_client.post(
        "/feedback",
        json={
            "card_id": "amex_gold",
            "reaction": "like",
            "target": "card",
        },
    )
    assert resp.status_code == 401


def test_submit_feedback_missing_card_id(test_client):
    headers = _signup_and_login(test_client)
    resp = test_client.post(
        "/feedback",
        json={
            "reaction": "like",
            "target": "card",
        },
        headers=headers,
    )
    assert resp.status_code == 422
