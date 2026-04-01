"""Story 2.2 tests for deterministic scoring integration in /predict."""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import logging
import pytest
from fastapi.testclient import TestClient
import src.serving.app as serving_app

pytest.importorskip("fastapi")


TEST_CARD_CATALOG = [
    {
        "card_id": "amex_gold",
        "card_name": "Amex Gold Card",
        "annual_fee": 250.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 4.0, "groceries": 4.0},
        },
    },
    {
        "card_id": "capital_one_venture_x",
        "card_name": "Capital One Venture X",
        "annual_fee": 395.0,
        "reward_rates": {
            "universal_base_rate": 2.0,
            "category_bonuses": {"travel": 5.0},
        },
    },
    {
        "card_id": "citi_double_cash",
        "card_name": "Citi Double Cash",
        "annual_fee": 0.0,
        "reward_rates": {"universal_base_rate": 2.0},
    },
    {
        "card_id": "blue_cash_preferred",
        "card_name": "Blue Cash Preferred",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"groceries": 6.0, "gas": 3.0},
        },
    },
    {
        "card_id": "discover_it_cash_back",
        "card_name": "Discover it Cash Back",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"gas": 5.0, "online_shopping": 5.0},
        },
    },
]


@pytest.fixture(autouse=True)
def fixed_scoring_catalog(monkeypatch):
    monkeypatch.setattr(serving_app, "CARD_CATALOG", TEST_CARD_CATALOG)
    monkeypatch.setattr(serving_app, "MAX_RECOMMENDATIONS", 10)


@pytest.fixture
def client() -> TestClient:
    return TestClient(serving_app.app)


def _base_payload() -> dict:
    return {
        "user_id": "story22-user",
        "spending_categories": {"dining": 300.0, "groceries": 200.0},
        "monthly_spend": 800.0,
        "preferred_rewards": ["cashback"],
        "transaction_history": [],
    }


@pytest.mark.parametrize(
    "persona_name,spending_categories,expected_top",
    [
        (
            "travel_focused",
            {"travel": 3000.0, "dining": 150.0},
            "Capital One Venture X",
        ),
        ("grocery_focused", {"groceries": 2500.0, "gas": 100.0}, "Blue Cash Preferred"),
        ("dining_focused", {"dining": 2800.0}, "Amex Gold Card"),
        ("cashback_focused", {"other": 3200.0}, "Citi Double Cash"),
        ("gas_commuter", {"gas": 2200.0}, "Discover it Cash Back"),
    ],
)
def test_predict_ranks_expected_top_card_for_personas(
    client: TestClient,
    persona_name: str,
    spending_categories: dict,
    expected_top: str,
) -> None:
    payload = _base_payload()
    payload["user_id"] = f"{persona_name}-user"
    payload["spending_categories"] = spending_categories
    payload["monthly_spend"] = sum(spending_categories.values())

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["recommended_cards"], "No recommendations returned"
    assert body["recommended_cards"][0]["card_name"] == expected_top
    assert body["recommended_cards"][0]["rank"] == 1


def test_predict_scores_are_ranked_descending(client: TestClient) -> None:
    response = client.post("/predict", json=_base_payload())
    assert response.status_code == 200

    scores = [card["score"] for card in response.json()["recommended_cards"]]
    assert scores == sorted(scores, reverse=True)


def test_predict_handles_unknown_spending_categories_with_defaults(
    client: TestClient,
) -> None:
    payload = _base_payload()
    payload["spending_categories"] = {"crypto": 750.0, "dining": 120.0}
    payload["monthly_spend"] = 900.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert len(response.json()["recommended_cards"]) >= 1


def test_predict_handles_missing_optional_fields_with_defaults(
    client: TestClient,
) -> None:
    payload = {"user_id": "missing-fields-user"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_cards"]
    assert body["model_version"] == "unloaded"


def test_predict_scoring_latency_under_100ms(client: TestClient) -> None:
    payload = _base_payload()
    payload["spending_categories"] = {
        "dining": 1000.0,
        "groceries": 900.0,
        "travel": 800.0,
        "gas": 700.0,
        "online_shopping": 600.0,
    }
    payload["monthly_spend"] = 4000.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["inference_latency_ms"] < 100.0


def test_predict_logs_anonymized_input_and_scores(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    user_id = "sensitive-user-123"
    payload = _base_payload()
    payload["user_id"] = user_id

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
    scoring_logs = [
        record.getMessage()
        for record in caplog.records
        if "predict_scoring" in record.getMessage()
    ]

    assert scoring_logs
    assert any(user_hash in message for message in scoring_logs)
    assert all(user_id not in message for message in scoring_logs)
    assert any("stage_latency_ms=" in message for message in scoring_logs)
