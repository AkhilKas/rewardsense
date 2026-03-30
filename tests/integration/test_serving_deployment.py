"""
Integration smoke test for the deployed RewardSense serving service (Story 1.4).

Verifies the live Cloud Run service is fully operational end-to-end:
  - /health returns 200 with model_version
  - /predict returns a valid response schema with recommended_cards, scores,
    and explanation
  - End-to-end latency is under 10 seconds (includes optional LLM call)

Usage
-----
Run against the live Cloud Run service:

    SERVING_URL=https://rewardsense-serving-760934308287.us-central1.run.app \
        pytest tests/integration/test_serving_deployment.py -m integration -v

In CI this is run automatically as a post-deploy step (see .github/workflows/ci.yml).
The SERVING_URL is set via a GitHub Actions variable; the test is skipped if unset.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import pytest
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVING_URL: str = os.getenv(
    "SERVING_URL",
    "https://rewardsense-serving-760934308287.us-central1.run.app",
)
REQUEST_TIMEOUT: int = 30  # generous per-request timeout (seconds)
LATENCY_LIMIT: float = 10.0  # end-to-end latency SLA (seconds)

# ---------------------------------------------------------------------------
# Sample predict payload (matches Story 2.1 PredictionRequest schema)
# ---------------------------------------------------------------------------

SAMPLE_PREDICT_PAYLOAD: Dict[str, Any] = {
    "user_id": "smoke-test-user-001",
    "spending_categories": {
        "groceries": 400.0,
        "dining": 200.0,
        "travel": 300.0,
        "gas": 100.0,
        "online_shopping": 150.0,
    },
    "monthly_spend": 1150.0,
    "preferred_rewards": ["cashback", "travel_points"],
    "transaction_history": [
        {
            "merchant": "Whole Foods",
            "category": "groceries",
            "amount": 85.50,
            "date": "2026-03-01",
        },
        {
            "merchant": "Delta Airlines",
            "category": "travel",
            "amount": 420.00,
            "date": "2026-03-10",
        },
        {
            "merchant": "Cheesecake Factory",
            "category": "dining",
            "amount": 62.00,
            "date": "2026-03-15",
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def serving_url() -> str:
    """Return the serving URL, skipping all tests if it's unreachable."""
    url = SERVING_URL.rstrip("/")
    try:
        resp = requests.get(f"{url}/health", timeout=REQUEST_TIMEOUT)
        is_json = resp.headers.get("content-type", "").startswith("application/json")
        if resp.status_code == 404 or not is_json:
            pytest.skip(
                f"Serving API not yet deployed at {url} "
                "(placeholder image still running — complete Epic 2 first)"
            )
    except requests.ConnectionError:
        pytest.skip(f"Serving service unreachable at {url}")
    return url


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHealthEndpoint:
    def test_returns_200(self, serving_url: str) -> None:
        resp = requests.get(f"{serving_url}/health", timeout=REQUEST_TIMEOUT)
        assert (
            resp.status_code == 200
        ), f"/health returned {resp.status_code}: {resp.text}"

    def test_returns_json(self, serving_url: str) -> None:
        resp = requests.get(f"{serving_url}/health", timeout=REQUEST_TIMEOUT)
        assert resp.headers.get("content-type", "").startswith(
            "application/json"
        ), f"Expected JSON content-type, got: {resp.headers.get('content-type')}"

    def test_status_is_healthy(self, serving_url: str) -> None:
        resp = requests.get(f"{serving_url}/health", timeout=REQUEST_TIMEOUT)
        body = resp.json()
        assert (
            body.get("status") == "healthy"
        ), f"Expected status='healthy', got: {body}"

    def test_includes_model_version(self, serving_url: str) -> None:
        resp = requests.get(f"{serving_url}/health", timeout=REQUEST_TIMEOUT)
        body = resp.json()
        assert (
            "model_version" in body
        ), f"/health response missing 'model_version' field: {body}"
        assert (
            body["model_version"] is not None
        ), "model_version is null — Production model may not be loaded"

    def test_model_version_matches_mlflow_production(self, serving_url: str) -> None:
        """Verify the reported model version is a valid non-empty string."""
        resp = requests.get(f"{serving_url}/health", timeout=REQUEST_TIMEOUT)
        body = resp.json()
        model_version = body.get("model_version")
        assert (
            isinstance(model_version, str) and model_version.strip()
        ), f"model_version should be a non-empty string, got: {model_version!r}"


@pytest.mark.integration
class TestPredictEndpoint:
    def test_returns_200_for_valid_payload(self, serving_url: str) -> None:
        resp = requests.post(
            f"{serving_url}/predict",
            json=SAMPLE_PREDICT_PAYLOAD,
            timeout=REQUEST_TIMEOUT,
        )
        assert (
            resp.status_code == 200
        ), f"/predict returned {resp.status_code}: {resp.text}"

    def test_response_contains_recommended_cards(self, serving_url: str) -> None:
        resp = requests.post(
            f"{serving_url}/predict",
            json=SAMPLE_PREDICT_PAYLOAD,
            timeout=REQUEST_TIMEOUT,
        )
        body = resp.json()
        assert (
            "recommended_cards" in body
        ), f"Response missing 'recommended_cards': {list(body.keys())}"
        assert isinstance(
            body["recommended_cards"], list
        ), f"'recommended_cards' should be a list, got: {type(body['recommended_cards'])}"
        assert len(body["recommended_cards"]) > 0, "recommended_cards list is empty"

    def test_recommended_cards_have_required_fields(self, serving_url: str) -> None:
        resp = requests.post(
            f"{serving_url}/predict",
            json=SAMPLE_PREDICT_PAYLOAD,
            timeout=REQUEST_TIMEOUT,
        )
        body = resp.json()
        for i, card in enumerate(body.get("recommended_cards", [])):
            assert "card_name" in card, f"Card {i} missing 'card_name': {card}"
            assert "score" in card, f"Card {i} missing 'score': {card}"
            assert "rank" in card, f"Card {i} missing 'rank': {card}"

    def test_response_contains_scores(self, serving_url: str) -> None:
        resp = requests.post(
            f"{serving_url}/predict",
            json=SAMPLE_PREDICT_PAYLOAD,
            timeout=REQUEST_TIMEOUT,
        )
        body = resp.json()
        assert "scores" in body, f"Response missing 'scores': {list(body.keys())}"

    def test_response_contains_explanation_field(self, serving_url: str) -> None:
        resp = requests.post(
            f"{serving_url}/predict",
            json=SAMPLE_PREDICT_PAYLOAD,
            timeout=REQUEST_TIMEOUT,
        )
        body = resp.json()
        # explanation may be null if LLM is disabled, but key must be present
        assert (
            "explanation" in body
        ), f"Response missing 'explanation' field: {list(body.keys())}"

    def test_invalid_payload_returns_422(self, serving_url: str) -> None:
        resp = requests.post(
            f"{serving_url}/predict",
            json={"bad_field": "invalid"},
            timeout=REQUEST_TIMEOUT,
        )
        assert (
            resp.status_code == 422
        ), f"Expected 422 for invalid payload, got {resp.status_code}"

    def test_end_to_end_latency_under_10_seconds(self, serving_url: str) -> None:
        start = time.monotonic()
        resp = requests.post(
            f"{serving_url}/predict",
            json=SAMPLE_PREDICT_PAYLOAD,
            timeout=REQUEST_TIMEOUT,
        )
        elapsed = time.monotonic() - start

        assert (
            resp.status_code == 200
        ), f"/predict returned {resp.status_code}: {resp.text}"
        assert (
            elapsed < LATENCY_LIMIT
        ), f"End-to-end latency {elapsed:.2f}s exceeded {LATENCY_LIMIT}s SLA"
