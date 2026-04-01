"""Story 2.3 tests for personalization model blending in /predict."""

# ruff: noqa: E402

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

import src.serving.app as serving_app

pytest.importorskip("fastapi")


TEST_CATALOG = [
    {
        "card_id": "flat_two",
        "card_name": "Flat Two Percent",
        "annual_fee": 0.0,
        "reward_rates": {"universal_base_rate": 2.0},
    },
    {
        "card_id": "travel_plus",
        "card_name": "Travel Plus",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"travel": 2.5},
        },
    },
]


class _FakePersonalizationScorer:
    def __init__(self, point_value: float = 0.05):
        self.default_point_value = 0.01
        self._point_value = point_value

    def _get_point_value(self, _):
        return self._point_value, True


@pytest.fixture(autouse=True)
def fixed_catalog(monkeypatch):
    monkeypatch.setattr(serving_app, "CARD_CATALOG", TEST_CATALOG)
    monkeypatch.setattr(serving_app, "MAX_RECOMMENDATIONS", 10)


@pytest.fixture
def client() -> TestClient:
    return TestClient(serving_app.app)


def _payload() -> dict:
    return {
        "user_id": "persona-user",
        "spending_categories": {"travel": 1200.0, "other": 1800.0},
        "monthly_spend": 3000.0,
        "preferred_rewards": [],
        "transaction_history": [],
    }


def test_personalization_changes_blended_scores_and_ranking(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONALIZATION_DETERMINISTIC_WEIGHT", "0.2")

    # Baseline: model unavailable -> deterministic only.
    monkeypatch.setattr(
        serving_app,
        "get_model",
        lambda: (_ for _ in ()).throw(RuntimeError("unloaded")),
    )
    baseline = client.post("/predict", json=_payload()).json()
    baseline_scores = {
        c["card_name"]: c["score"] for c in baseline["recommended_cards"]
    }
    baseline_top = baseline["recommended_cards"][0]["card_name"]

    # Personalized: model available with strong point value uplift.
    monkeypatch.setattr(serving_app, "get_model", lambda: _FakePersonalizationScorer())
    personalized = client.post("/predict", json=_payload()).json()
    personalized_scores = {
        c["card_name"]: c["score"] for c in personalized["recommended_cards"]
    }
    personalized_top = personalized["recommended_cards"][0]["card_name"]

    assert any(
        personalized_scores[name] != baseline_scores[name] for name in baseline_scores
    )
    assert baseline_top != personalized_top


def test_blend_weight_is_configurable_via_env(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serving_app, "get_model", lambda: _FakePersonalizationScorer())

    monkeypatch.setenv("PERSONALIZATION_DETERMINISTIC_WEIGHT", "1.0")
    det_weight_body = client.post("/predict", json=_payload()).json()
    first_det = det_weight_body["recommended_cards"][0]
    assert first_det["score"] == first_det["deterministic_score"]

    monkeypatch.setenv("PERSONALIZATION_DETERMINISTIC_WEIGHT", "0.0")
    ml_weight_body = client.post("/predict", json=_payload()).json()
    first_ml = ml_weight_body["recommended_cards"][0]
    assert first_ml["score"] == first_ml["personalization_score"]


def test_personalization_latency_under_500ms(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serving_app, "get_model", lambda: _FakePersonalizationScorer())
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    assert response.json()["inference_latency_ms"] < 500.0


def test_predict_logs_score_components_for_monitoring(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(serving_app, "get_model", lambda: _FakePersonalizationScorer())
    caplog.set_level(logging.INFO)

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    scoring_logs = [
        record.getMessage()
        for record in caplog.records
        if "predict_scoring" in record.getMessage()
    ]
    assert scoring_logs
    assert any("score_components=" in message for message in scoring_logs)
    assert any("deterministic_score" in message for message in scoring_logs)
    assert any("personalization_score" in message for message in scoring_logs)
