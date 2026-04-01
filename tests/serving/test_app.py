"""Unit tests for FastAPI scaffold in src/serving/app.py (Story 2.1)."""

# ruff: noqa: E402

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.serving.app import app

pytest.importorskip("fastapi")


def _valid_payload() -> dict:
    return {
        "user_id": "user-123",
        "spending_categories": {
            "groceries": 300.0,
            "dining": 200.0,
            "travel": 150.0,
        },
        "monthly_spend": 1200.0,
        "preferred_rewards": ["cashback", "travel"],
        "transaction_history": [
            {
                "merchant": "Whole Foods",
                "category": "groceries",
                "amount": 45.5,
                "date": "2026-03-20",
            }
        ],
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_valid_json(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert isinstance(body["model_version"], str)
    assert isinstance(body["uptime_seconds"], float)
    assert body["uptime_seconds"] >= 0.0


def test_predict_accepts_valid_payload(client: TestClient) -> None:
    response = client.post("/predict", json=_valid_payload())
    assert response.status_code == 200

    body = response.json()
    assert "recommended_cards" in body
    assert "model_version" in body
    assert "inference_latency_ms" in body
    assert isinstance(body["recommended_cards"], list)
    assert body["recommended_cards"][0]["card_name"]
    assert "score" in body["recommended_cards"][0]
    assert "rank" in body["recommended_cards"][0]
    assert "explanation" in body["recommended_cards"][0]


def test_predict_invalid_payload_returns_422_with_details(client: TestClient) -> None:
    response = client.post("/predict", json={"monthly_spend": 1000})
    assert response.status_code == 422

    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)
    assert any(err["loc"][-1] == "user_id" for err in body["detail"])


def test_predict_rejects_unknown_top_level_field(client: TestClient) -> None:
    payload = _valid_payload()
    payload["unknown_field"] = "not-allowed"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "extra_forbidden" for err in detail)


def test_predict_rejects_invalid_nested_transaction_amount(client: TestClient) -> None:
    payload = _valid_payload()
    payload["transaction_history"][0]["amount"] = -1

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    detail = response.json()["detail"]
    assert any(
        err["loc"][-2:] == ["transaction_history", 0]
        or "transaction_history" in err["loc"]
        for err in detail
    )


def test_cors_headers_present_for_allowed_origin(client: TestClient) -> None:
    origin = "http://localhost:5173"
    response = client.options(
        "/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
    assert response.headers.get("access-control-allow-credentials") == "true"
