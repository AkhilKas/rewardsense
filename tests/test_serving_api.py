"""
Comprehensive tests for the RewardSense Serving API.

Covers /health, /predict, /monitoring endpoints and CORS configuration.
"""

from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)

VALID_PREDICT_PAYLOAD = {
    "user_id": "test-user-001",
    "spending_categories": {"dining": 500, "travel": 300, "groceries": 400},
    "monthly_spend": 1200,
    "preferred_rewards": ["travel"],
}

CARD_REQUIRED_FIELDS = {
    "card_name",
    "issuer",
    "score",
    "rank",
    "explanation",
    "deterministic_score",
    "personalization_score",
    "annual_fee",
    "reward_rate",
    "key_benefits",
}


# ── /health ──────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_schema(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_version" in data
        assert "uptime_seconds" in data

    def test_uptime_is_positive(self):
        data = client.get("/health").json()
        assert data["uptime_seconds"] > 0


# ── /predict ─────────────────────────────────────────────────────────


class TestPredictEndpoint:
    def test_valid_request_returns_200(self):
        resp = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "recommended_cards" in data
        assert "model_version" in data
        assert "inference_latency_ms" in data

    def test_card_has_all_required_fields(self):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        for card in data["recommended_cards"]:
            missing = CARD_REQUIRED_FIELDS - set(card.keys())
            assert (
                not missing
            ), f"Card '{card.get('card_name')}' missing fields: {missing}"

    def test_cards_ranked_in_order(self):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        ranks = [c["rank"] for c in data["recommended_cards"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_card_benefits_nonempty(self):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        for card in data["recommended_cards"]:
            assert isinstance(card["key_benefits"], list)
            assert (
                len(card["key_benefits"]) > 0
            ), f"Card '{card['card_name']}' has empty key_benefits"

    def test_card_issuer_nonempty(self):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        for card in data["recommended_cards"]:
            assert isinstance(card["issuer"], str)
            assert (
                len(card["issuer"]) > 0
            ), f"Card '{card['card_name']}' has empty issuer"

    def test_missing_user_id_returns_422(self):
        payload = {
            "spending_categories": {"dining": 500},
            "monthly_spend": 500,
            "preferred_rewards": ["cashback"],
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_spending_categories(self):
        payload = {
            "user_id": "edge-case-empty-spend",
            "spending_categories": {},
            "monthly_spend": 0,
            "preferred_rewards": ["cashback"],
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "recommended_cards" in data

    def test_invalid_field_type_returns_422(self):
        payload = {
            "user_id": "type-mismatch",
            "spending_categories": {"dining": 500},
            "monthly_spend": "not_a_number",
            "preferred_rewards": ["travel"],
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ── /monitoring ──────────────────────────────────────────────────────


class TestMonitoringEndpoint:
    def test_returns_200(self):
        resp = client.get("/monitoring")
        assert resp.status_code == 200

    def test_response_schema(self):
        data = client.get("/monitoring").json()
        assert "model_version" in data
        assert "last_deployment_time" in data
        assert "drift_check" in data
        assert "serving_metrics" in data
        assert "retrain_history" in data

    def test_drift_check_structure(self):
        data = client.get("/monitoring").json()
        drift = data["drift_check"]
        assert isinstance(drift["feature_drift"], dict)
        assert "detected" in drift

    def test_serving_metrics_fields(self):
        data = client.get("/monitoring").json()
        m = data["serving_metrics"]
        for field in (
            "request_count",
            "avg_latency_ms",
            "error_rate",
            "p95_latency_ms",
        ):
            assert field in m, f"serving_metrics missing '{field}'"

    def test_retrain_history_is_list(self):
        data = client.get("/monitoring").json()
        assert isinstance(data["retrain_history"], list)

    def test_model_version_matches_health(self):
        health_version = client.get("/health").json()["model_version"]
        monitoring_version = client.get("/monitoring").json()["model_version"]
        assert monitoring_version == health_version


# ── CORS ─────────────────────────────────────────────────────────────


class TestCORS:
    def test_preflight_predict(self):
        resp = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_cors_headers_on_response(self):
        resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" in resp.headers
        assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
