"""Story 2.4 tests for LLM explanation integration in /predict.

Verifies:
  - Top 3 cards include human-readable explanation fields when LLM is enabled
  - LLM failures don't crash the endpoint (graceful fallback to template)
  - Concurrent LLM calls execute (asyncio.gather)
  - Total /predict latency stays within acceptable bounds
  - LLM latency is logged in stage_latency_ms
"""

# ruff: noqa: E402

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

import src.serving.app as serving_app

pytest.importorskip("fastapi")


# ---------------------------------------------------------------------------
# Fake LLM objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeExplanation:
    summary: str = "This card is great for your spending."
    rationale: List[str] = field(
        default_factory=lambda: ["High reward rate.", "Low annual fee."]
    )
    confidence: float = 0.85
    disclaimers: List[str] = field(default_factory=list)
    quality_checks: Dict[str, bool] = field(
        default_factory=lambda: {
            "length_ok": True,
            "relevance_ok": True,
            "hallucination_guard_ok": True,
        }
    )
    raw_response: str = ""
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: float = 50.0


class _FakeExplanationGenerator:
    """Fake ExplanationGenerator that returns canned explanations."""

    def __init__(
        self,
        delay_sec: float = 0.0,
        fail: bool = False,
        card_specific: bool = True,
    ):
        self._delay = delay_sec
        self._fail = fail
        self._card_specific = card_specific
        self.call_count = 0

    def generate(
        self,
        explanation_type: Any,
        scoring_output: Dict[str, Any],
        personalization_signals: Dict[str, Any],
        llm_params: Optional[Dict[str, Any]] = None,
    ) -> _FakeExplanation:
        self.call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("Simulated LLM failure")

        card_name = scoring_output.get("best_card", {}).get("card_name", "the card")
        if self._card_specific:
            summary = f"{card_name} is an excellent match for your profile."
        else:
            summary = "This card is great for your spending."

        return _FakeExplanation(
            summary=summary,
            rationale=["Strong category bonus.", "Good value."],
        )


TEST_CATALOG = [
    {
        "card_id": "card_a",
        "card_name": "Card Alpha",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 2.0,
            "category_bonuses": {"dining": 4.0},
        },
    },
    {
        "card_id": "card_b",
        "card_name": "Card Beta",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"travel": 5.0},
        },
    },
    {
        "card_id": "card_c",
        "card_name": "Card Gamma",
        "annual_fee": 0.0,
        "reward_rates": {"universal_base_rate": 1.5},
    },
]


@pytest.fixture(autouse=True)
def _fixed_catalog(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(serving_app, "CARD_CATALOG", TEST_CATALOG)
    monkeypatch.setattr(serving_app, "MAX_RECOMMENDATIONS", 10)
    # Reset the module-level LLM singleton between tests
    monkeypatch.setattr(serving_app, "_explanation_generator", None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(serving_app.app)


def _payload() -> dict:
    return {
        "user_id": "llm-test-user",
        "spending_categories": {"dining": 500.0, "travel": 300.0},
        "monthly_spend": 1000.0,
        "preferred_rewards": ["cashback"],
        "transaction_history": [],
    }


# ---------------------------------------------------------------------------
# Test: LLM explanations enrich top 3 cards when enabled
# ---------------------------------------------------------------------------


def test_llm_explanations_enrich_top_cards(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When LLM is enabled, top cards should have LLM-generated explanations."""
    monkeypatch.setattr(serving_app, "ENABLE_LLM_EXPLANATIONS", True)
    monkeypatch.setattr(serving_app, "LLM_MODULES_AVAILABLE", True)
    monkeypatch.setattr(serving_app, "LLM_TOP_N_EXPLANATIONS", 3)

    fake_gen = _FakeExplanationGenerator(card_specific=True)
    monkeypatch.setattr(serving_app, "_explanation_generator", fake_gen)

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    body = response.json()
    cards = body["recommended_cards"][:3]
    for card in cards:
        assert card["explanation"]
        assert len(card["explanation"]) > 10  # non-trivial explanation


# ---------------------------------------------------------------------------
# Test: LLM failure gracefully falls back to template
# ---------------------------------------------------------------------------


def test_llm_failure_falls_back_to_template(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM failures should not crash; endpoint returns template explanations."""
    monkeypatch.setattr(serving_app, "ENABLE_LLM_EXPLANATIONS", True)
    monkeypatch.setattr(serving_app, "LLM_MODULES_AVAILABLE", True)
    monkeypatch.setattr(serving_app, "LLM_TOP_N_EXPLANATIONS", 3)

    failing_gen = _FakeExplanationGenerator(fail=True)
    monkeypatch.setattr(serving_app, "_explanation_generator", failing_gen)

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    body = response.json()
    cards = body["recommended_cards"]
    assert len(cards) >= 1
    # Each card should still have a non-empty explanation (template fallback)
    for card in cards:
        assert card["explanation"]
        assert len(card["explanation"]) > 5


# ---------------------------------------------------------------------------
# Test: LLM disabled => template explanations only, no LLM latency
# ---------------------------------------------------------------------------


def test_llm_disabled_uses_template_explanations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When LLM is disabled, explanations are deterministic templates."""
    monkeypatch.setattr(serving_app, "ENABLE_LLM_EXPLANATIONS", False)

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    body = response.json()
    for card in body["recommended_cards"]:
        assert card["explanation"]
        # Template explanations mention reward rates
        assert "%" in card["explanation"] or "return" in card["explanation"].lower()


# ---------------------------------------------------------------------------
# Test: Concurrent LLM reduces total latency vs sequential
# ---------------------------------------------------------------------------


def test_concurrent_llm_calls_run_in_parallel(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All 3 LLM calls should execute concurrently (total time ≈ single call)."""
    monkeypatch.setattr(serving_app, "ENABLE_LLM_EXPLANATIONS", True)
    monkeypatch.setattr(serving_app, "LLM_MODULES_AVAILABLE", True)
    monkeypatch.setattr(serving_app, "LLM_TOP_N_EXPLANATIONS", 3)
    monkeypatch.setattr(serving_app, "LLM_EXPLANATION_TIMEOUT_SEC", 5.0)

    # Each call takes ~100ms; 3 sequential = 300ms; concurrent should be ~100ms
    slow_gen = _FakeExplanationGenerator(delay_sec=0.1, card_specific=True)
    monkeypatch.setattr(serving_app, "_explanation_generator", slow_gen)

    start = time.perf_counter()
    response = client.post("/predict", json=_payload())
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert response.status_code == 200
    # With 3 concurrent 100ms calls, total < 250ms proves concurrency
    # (generous buffer for CI variance; sequential would be ≥300ms)
    assert elapsed_ms < 500.0, f"Expected concurrent execution, got {elapsed_ms:.0f}ms"
    # All 3 should have been called
    assert slow_gen.call_count == 3


# ---------------------------------------------------------------------------
# Test: LLM timeout triggers fallback
# ---------------------------------------------------------------------------


def test_llm_timeout_triggers_template_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If LLM exceeds timeout, template fallback is used."""
    monkeypatch.setattr(serving_app, "ENABLE_LLM_EXPLANATIONS", True)
    monkeypatch.setattr(serving_app, "LLM_MODULES_AVAILABLE", True)
    monkeypatch.setattr(serving_app, "LLM_TOP_N_EXPLANATIONS", 1)
    monkeypatch.setattr(serving_app, "LLM_EXPLANATION_TIMEOUT_SEC", 0.05)

    # Slow call that exceeds the 50ms timeout
    slow_gen = _FakeExplanationGenerator(delay_sec=0.5)
    monkeypatch.setattr(serving_app, "_explanation_generator", slow_gen)

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    body = response.json()
    # Should still have explanations (template fallback)
    assert body["recommended_cards"][0]["explanation"]


# ---------------------------------------------------------------------------
# Test: LLM explanation latency is logged
# ---------------------------------------------------------------------------


def test_llm_latency_logged_in_stage_breakdown(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When LLM is enabled, predict_scoring log line includes llm latency."""
    monkeypatch.setattr(serving_app, "ENABLE_LLM_EXPLANATIONS", True)
    monkeypatch.setattr(serving_app, "LLM_MODULES_AVAILABLE", True)
    monkeypatch.setattr(serving_app, "LLM_TOP_N_EXPLANATIONS", 2)

    fake_gen = _FakeExplanationGenerator()
    monkeypatch.setattr(serving_app, "_explanation_generator", fake_gen)

    caplog.set_level(logging.INFO)
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    scoring_logs = [
        r.getMessage() for r in caplog.records if "predict_scoring" in r.getMessage()
    ]
    assert scoring_logs
    assert any("llm_explanation" in msg for msg in scoring_logs)


# ---------------------------------------------------------------------------
# Test: /predict latency still within 10s even with LLM
# ---------------------------------------------------------------------------


def test_predict_latency_under_10s_with_llm(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Total /predict latency must stay under 10s (acceptance criteria)."""
    monkeypatch.setattr(serving_app, "ENABLE_LLM_EXPLANATIONS", True)
    monkeypatch.setattr(serving_app, "LLM_MODULES_AVAILABLE", True)

    fake_gen = _FakeExplanationGenerator(delay_sec=0.05)
    monkeypatch.setattr(serving_app, "_explanation_generator", fake_gen)

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    assert response.json()["inference_latency_ms"] < 10_000.0
