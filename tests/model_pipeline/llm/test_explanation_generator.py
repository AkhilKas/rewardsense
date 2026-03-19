from src.model_pipeline.llm.explanation_generator import (
    ExplanationGenerator,
    ExplanationQualityFilter,
)
from src.model_pipeline.llm.prompt_builder import ExplanationType


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def generate(self, system_message: str, user_message: str, **kwargs) -> str:
        self.last_prompt = (system_message, user_message, kwargs)
        return self.response


def test_explanation_generator_happy_path(scoring_output_fixture):
    raw = '{"summary":"Use Amex Gold.","rationale":["4x dining","best expected value"],"confidence":0.95}'
    client = _FakeLLMClient(raw)
    quality = ExplanationQualityFilter(max_chars=500, min_rationale_points=2)
    generator = ExplanationGenerator(llm_client=client, quality_filter=quality)

    out = generator.generate(
        explanation_type=ExplanationType.SINGLE_TRANSACTION,
        scoring_output=scoring_output_fixture,
        personalization_signals={"user_segment": "foodie"},
    )

    assert out.summary == "Use Amex Gold."
    assert out.quality_checks["length_ok"] is True
    assert out.quality_checks["relevance_ok"] is True
    assert out.used_fallback is False
    assert out.latency_ms >= 0.0


def test_quality_filter_flags_low_relevance():
    quality = ExplanationQualityFilter(max_chars=500, min_rationale_points=2)
    checks = quality.evaluate(
        summary="Use card A.",
        rationale=["good"],
        context={"scoring": {"best_card": {"card_name": "Card A"}}},
    )
    assert checks["relevance_ok"] is False


def test_quality_filter_flags_hallucination():
    quality = ExplanationQualityFilter(max_chars=500, min_rationale_points=1)
    checks = quality.evaluate(
        summary="Use Amex Gold with 12x rewards.",
        rationale=["12x at dining"],
        context={
            "scoring": {"best_card": {"card_name": "Amex Gold", "reward_rate": 4.0}}
        },
    )
    assert checks["hallucination_guard_ok"] is False


def test_generator_falls_back_on_parse_error(scoring_output_fixture):
    client = _FakeLLMClient("not-json-response")
    generator = ExplanationGenerator(llm_client=client)

    out = generator.generate(
        explanation_type=ExplanationType.SINGLE_TRANSACTION,
        scoring_output=scoring_output_fixture,
        personalization_signals={"user_segment": "foodie"},
    )

    assert out.used_fallback is True
    assert "fallback" in " ".join(out.disclaimers).lower()
    assert out.fallback_reason is not None


def test_generator_falls_back_when_quality_fails(scoring_output_fixture):
    raw = (
        '{"summary":"Use Amex Gold with 12x rewards.",'
        '"rationale":["12x on dining"],'
        '"confidence":0.9}'
    )
    client = _FakeLLMClient(raw)
    generator = ExplanationGenerator(llm_client=client)

    out = generator.generate(
        explanation_type=ExplanationType.SINGLE_TRANSACTION,
        scoring_output=scoring_output_fixture,
        personalization_signals={"user_segment": "foodie"},
    )

    assert out.used_fallback is True
    assert out.fallback_reason == "quality_filter_failed"
