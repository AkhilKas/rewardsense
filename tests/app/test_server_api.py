# ruff: noqa: E402
import sys
from types import SimpleNamespace

import pytest

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

fastapi = pytest.importorskip("fastapi")

from src.app.server import (
    RecommendRequest,
    RecommendResponse,
    RewardSenseService,
    create_app,
)
from src.model_pipeline.llm.explanation_generator import (
    ExplanationGenerator,
    ExplanationQualityFilter,
)

SAMPLE_PORTFOLIO = [
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
    "amount": 80.0,
    "category": "dining",
    "merchant": "Sweetgreen",
    "mcc_code": 5812,
}


class _FakeLLMClient:
    def generate(self, system_message: str, user_message: str, **kwargs) -> str:
        return (
            '{"summary":"Use Amex Gold for Sweetgreen.",'
            '"pros":["Earns 4x on dining purchases","Highest expected return for this merchant"],'
            '"cons":["Annual fee may offset rewards for light spenders","No cashback option available"],'
            '"best_for":"Frequent diners",'
            '"confidence":0.94}'
        )


def test_health_endpoint_ok():
    app = create_app(service=RewardSenseService(enable_llm_explanations=False))
    health_route = next(r for r in app.routes if getattr(r, "path", "") == "/health")
    response = health_route.endpoint()
    assert response["status"] == "healthy"


def test_recommend_endpoint_returns_scoring_and_explanation():
    generator = ExplanationGenerator(
        llm_client=_FakeLLMClient(),
        quality_filter=ExplanationQualityFilter(max_chars=500, min_rationale_points=2),
    )
    service = RewardSenseService(
        enable_llm_explanations=True,
        explanation_generator=generator,
    )

    app = create_app(service=service)
    recommend_route = next(
        r
        for r in app.routes
        if getattr(r, "path", "") == "/recommend"
        and "POST" in getattr(r, "methods", set())
    )

    payload = RecommendRequest(
        portfolio=SAMPLE_PORTFOLIO,
        transaction=SAMPLE_TRANSACTION,
        personalization_signals={"user_segment": "foodie"},
        user_features=[{"monthly_budget": 4000.0, "num_cards": 2}],
        explanation_type="single_transaction_recommendation",
    )

    body = recommend_route.endpoint(payload)
    assert isinstance(body, RecommendResponse)
    body = body.model_dump()
    assert body["recommendation"]["best_card_id"] == "amex_gold"
    assert body["explanation"] is not None
    assert body["explanation"]["factual_accuracy"]["passed"] is True
