from contextlib import contextmanager
import sys
from types import SimpleNamespace

import pandas as pd

# Local test runner may not have optional runtime deps installed.
if "loguru" not in sys.modules:
    sys.modules["loguru"] = SimpleNamespace(
        logger=SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
            success=lambda *args, **kwargs: None,
        )
    )

from src.app.server import RewardSenseService
from src.model_pipeline.llm.explanation_generator import (
    ExplanationGenerator,
    ExplanationQualityFilter,
)

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
    "amount": 120.0,
    "category": "dining",
    "merchant": "Chipotle",
    "mcc_code": 5812,
}


class _FakeLLMClient:
    def generate(self, system_message: str, user_message: str, **kwargs) -> str:
        return (
            '{"summary":"Use Amex Gold for Chipotle.",'
            '"pros":["Amex Gold earns 4x on dining","Highest expected reward for this category"],'
            '"cons":["Annual fee may offset rewards for light spenders","Limited cashback flexibility"],'
            '"best_for":"Frequent diners",'
            '"confidence":0.96}'
        )


class _FakeTracker:
    def __init__(self):
        self.logged_metrics = None
        self.logged_params = None
        self.logged_dict = None

    @contextmanager
    def start_run(self, run_name=None, tags=None, nested=False):
        yield object()

    def log_metrics(self, metrics, step=None):
        self.logged_metrics = metrics

    def log_params(self, params):
        self.logged_params = params

    def log_dict(self, payload, filename):
        self.logged_dict = {"payload": payload, "filename": filename}


def test_recommend_without_explanations_returns_scoring_only():
    service = RewardSenseService(enable_llm_explanations=False)

    result = service.recommend(
        portfolio=SAMPLE_PORTFOLIO,
        transaction=SAMPLE_TRANSACTION,
        user_features=pd.DataFrame(),
    )

    assert result["recommendation"]["best_card_id"] is not None
    assert result["explanation"] is None
    assert result["llm_explanations_enabled"] is False


def test_recommend_with_explanations_uses_real_scoring_output():
    generator = ExplanationGenerator(
        llm_client=_FakeLLMClient(),
        quality_filter=ExplanationQualityFilter(max_chars=500, min_rationale_points=2),
    )
    service = RewardSenseService(
        enable_llm_explanations=True,
        explanation_generator=generator,
    )

    result = service.recommend(
        portfolio=SAMPLE_PORTFOLIO,
        transaction=SAMPLE_TRANSACTION,
        personalization_signals={"user_segment": "foodie"},
        user_features=pd.DataFrame(),
    )

    assert result["recommendation"]["best_card_id"] == "amex_gold"
    assert result["explanation"] is not None
    assert result["explanation"]["quality_checks"]["relevance_ok"] is True
    assert result["explanation"]["factual_accuracy"]["passed"] is True
    assert result["explanation"]["used_fallback"] is False
    assert result["explanation"]["latency_ms"] >= 0.0
    assert "readability" in result["explanation"]


def test_recommend_with_tracker_logs_llm_metrics():
    generator = ExplanationGenerator(
        llm_client=_FakeLLMClient(),
        quality_filter=ExplanationQualityFilter(max_chars=500, min_rationale_points=2),
    )
    tracker = _FakeTracker()
    service = RewardSenseService(
        enable_llm_explanations=True,
        explanation_generator=generator,
        tracker=tracker,
    )

    service.recommend(
        portfolio=SAMPLE_PORTFOLIO,
        transaction=SAMPLE_TRANSACTION,
        personalization_signals={"user_segment": "foodie"},
        user_features=pd.DataFrame(),
    )

    assert tracker.logged_metrics is not None
    assert "factual_accuracy_score" in tracker.logged_metrics
    assert "explanation_latency_ms" in tracker.logged_metrics
    assert (
        tracker.logged_params["explanation_type"] == "single_transaction_recommendation"
    )
    assert tracker.logged_dict is not None
    assert (
        tracker.logged_dict["filename"]
        == "llm_explanation_single_transaction_recommendation.json"
    )


def test_recommend_fallback_when_llm_output_invalid():
    class _BadLLMClient:
        def generate(self, system_message: str, user_message: str, **kwargs) -> str:
            return "non-json"

    generator = ExplanationGenerator(
        llm_client=_BadLLMClient(),
        quality_filter=ExplanationQualityFilter(max_chars=500, min_rationale_points=2),
    )
    service = RewardSenseService(
        enable_llm_explanations=True,
        explanation_generator=generator,
    )

    result = service.recommend(
        portfolio=SAMPLE_PORTFOLIO,
        transaction=SAMPLE_TRANSACTION,
        personalization_signals={"user_segment": "foodie"},
        user_features=pd.DataFrame(),
    )
    assert result["explanation"] is not None
    assert result["explanation"]["used_fallback"] is True
