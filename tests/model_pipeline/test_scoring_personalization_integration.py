"""
Integration tests — Scoring Engine + Personalization Model (Story 3.5).

Validates that the deterministic scoring engine (Epic 2) and the ML
personalization model (Epic 3) interoperate correctly.

Acceptance criteria tested:
1. Integration tests pass for scoring + personalization pipeline
2. Cold-start fallback produces valid (non-personalized) recommendations
3. Combined latency under 200ms for a single recommendation
4. Serialized model produces identical results to in-memory model
"""

import time

import joblib
import numpy as np
import pandas as pd
import pytest

from src.model_pipeline.personalization.personalized_scorer import (
    DEFAULT_POINT_VALUE,
    PersonalizedScorer,
)

# ── Shared fixtures ───────────────────────────────────────────────────

SAMPLE_PORTFOLIO = [
    {
        "card_id": "chase_sapphire",
        "card_name": "Chase Sapphire Preferred",
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 3.0, "travel": 2.0},
        },
        "annual_fee": 95,
    },
    {
        "card_id": "amex_gold",
        "card_name": "Amex Gold",
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 4.0, "groceries": 4.0},
        },
        "annual_fee": 250,
    },
    {
        "card_id": "citi_double",
        "card_name": "Citi Double Cash",
        "reward_rates": {"universal_base_rate": 2.0},
        "annual_fee": 0,
    },
]

SAMPLE_TRANSACTION = {
    "amount": 100.0,
    "category": "dining",
    "merchant": "Olive Garden",
    "mcc_code": 5812,
}


class _FakePersonalizationModel:
    """Minimal sklearn-compatible stub that returns a fixed prediction."""

    def __init__(self, value: float = 0.02):
        self._value = value

    def predict(self, X):
        return np.array([self._value] * len(X))


@pytest.fixture()
def fake_model():
    return _FakePersonalizationModel(value=0.02)


@pytest.fixture()
def user_features():
    """Single-row DataFrame mimicking a feature-engineered user."""
    return pd.DataFrame(
        {
            "monthly_budget": [5000.0],
            "annual_budget": [60000.0],
            "num_cards": [3],
            "monthly_budget_log": [8.5172],
            "age_group_ordinal": [3],
            "total_spending": [12000.0],
            "total_transactions": [150.0],
            "avg_transaction_amount": [80.0],
            "median_transaction_amount": [65.0],
            "transaction_amount_std": [45.0],
            "spending_diversity": [1.8],
            "weekend_spending_ratio": [0.3],
            "card_switch_rate": [0.4],
            "num_cards_used": [3],
            "num_unique_mccs": [12],
            "num_unique_merchants": [25],
            "repeat_merchant_ratio": [0.5],
        }
    )


# ── 1. Personalized output differs from unpersonalized ───────────────


class TestPersonalizedDiffersFromUnpersonalized:
    def test_reward_amounts_differ(self, fake_model, user_features):
        scorer_plain = PersonalizedScorer(model=None)
        scorer_personal = PersonalizedScorer(model=fake_model)

        plain_result = scorer_plain.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=None
        )
        personal_result = scorer_personal.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features
        )

        assert plain_result["is_personalized"] is False
        assert personal_result["is_personalized"] is True
        assert plain_result["point_value"] != personal_result["point_value"]

        plain_amounts = [c["reward_amount"] for c in plain_result["ranked"]]
        personal_amounts = [c["reward_amount"] for c in personal_result["ranked"]]
        assert plain_amounts != personal_amounts

    def test_raw_scores_preserved(self, fake_model, user_features):
        scorer = PersonalizedScorer(model=fake_model)
        result = scorer.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features
        )
        for card in result["ranked"]:
            assert "raw_reward_amount" in card
            assert card["raw_reward_amount"] > 0


# ── 2. Determinism ───────────────────────────────────────────────────


class TestDeterminism:
    def test_same_inputs_same_output(self, fake_model, user_features):
        scorer = PersonalizedScorer(model=fake_model)

        r1 = scorer.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features
        )
        r2 = scorer.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features
        )

        assert r1["best_card_id"] == r2["best_card_id"]
        assert r1["point_value"] == r2["point_value"]

        amounts_1 = [c["reward_amount"] for c in r1["ranked"]]
        amounts_2 = [c["reward_amount"] for c in r2["ranked"]]
        assert amounts_1 == amounts_2

    def test_batch_deterministic(self, fake_model, user_features):
        scorer = PersonalizedScorer(model=fake_model)
        txns = [SAMPLE_TRANSACTION, {**SAMPLE_TRANSACTION, "amount": 200.0}]

        b1 = scorer.score_batch(SAMPLE_PORTFOLIO, txns, user_features=user_features)
        b2 = scorer.score_batch(SAMPLE_PORTFOLIO, txns, user_features=user_features)

        for r1, r2 in zip(b1, b2):
            assert r1["best_card_id"] == r2["best_card_id"]


# ── 3. Cold-start fallback ───────────────────────────────────────────


class TestColdStartFallback:
    def test_no_features_returns_valid_scores(self, fake_model):
        scorer = PersonalizedScorer(model=fake_model)
        result = scorer.score(SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=None)

        assert result["is_personalized"] is False
        assert result["point_value"] == DEFAULT_POINT_VALUE
        assert result["best_card_id"] is not None
        assert len(result["ranked"]) == len(SAMPLE_PORTFOLIO)

    def test_no_model_returns_valid_scores(self, user_features):
        scorer = PersonalizedScorer(model=None)
        result = scorer.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features
        )

        assert result["is_personalized"] is False
        assert result["best_card_id"] is not None

    def test_fallback_ranking_is_deterministic(self, fake_model):
        scorer = PersonalizedScorer(model=fake_model)
        r1 = scorer.score(SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=None)
        r2 = scorer.score(SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=None)

        ids_1 = [c["card_id"] for c in r1["ranked"]]
        ids_2 = [c["card_id"] for c in r2["ranked"]]
        assert ids_1 == ids_2

    def test_bad_prediction_falls_back(self, user_features):
        """Model that returns NaN should trigger fallback."""

        class _NanModel:
            def predict(self, X):
                return np.array([float("nan")])

        scorer = PersonalizedScorer(model=_NanModel())
        result = scorer.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features
        )
        assert result["is_personalized"] is False
        assert result["point_value"] == DEFAULT_POINT_VALUE


# ── 4. Latency ───────────────────────────────────────────────────────


class TestLatency:
    def test_single_recommendation_under_200ms(self, fake_model, user_features):
        scorer = PersonalizedScorer(model=fake_model)

        # Warm-up call
        scorer.score(SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features)

        start = time.perf_counter()
        scorer.score(SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"Latency {elapsed_ms:.1f}ms exceeds 200ms"


# ── 5. Serialization round-trip ──────────────────────────────────────


class TestSerialization:
    def test_joblib_roundtrip_identical(self, fake_model, user_features, tmp_path):
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(fake_model, model_path)

        scorer_mem = PersonalizedScorer(model=fake_model)
        scorer_disk = PersonalizedScorer.from_artifact(model_path)

        r_mem = scorer_mem.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features
        )
        r_disk = scorer_disk.score(
            SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION, user_features=user_features
        )

        assert r_mem["best_card_id"] == r_disk["best_card_id"]
        assert r_mem["point_value"] == r_disk["point_value"]

        amounts_mem = [c["reward_amount"] for c in r_mem["ranked"]]
        amounts_disk = [c["reward_amount"] for c in r_disk["ranked"]]
        assert amounts_mem == amounts_disk

    def test_from_artifact_creates_scorer(self, fake_model, tmp_path):
        model_path = tmp_path / "model.joblib"
        joblib.dump(fake_model, model_path)

        scorer = PersonalizedScorer.from_artifact(model_path)
        assert scorer.model is not None
        result = scorer.score(SAMPLE_PORTFOLIO, SAMPLE_TRANSACTION)
        assert result["best_card_id"] is not None
