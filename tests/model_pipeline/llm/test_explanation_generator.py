from src.model_pipeline.llm.explanation_generator import (
    ExplanationGenerator,
    ExplanationQualityFilter,
)
from src.model_pipeline.llm.prompt_builder import ExplanationType
from src.model_pipeline.llm.response_parser import ParsedExplanation


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def generate(self, system_message: str, user_message: str, **kwargs) -> str:
        self.last_prompt = (system_message, user_message, kwargs)
        return self.response


def test_explanation_generator_happy_path(scoring_output_fixture):
    raw = (
        '{"summary":"Use Amex Gold.",'
        '"pros":["4x dining rewards","Best expected value in this category"],'
        '"cons":["$250 annual fee requires high spend","No cashback option"],'
        '"best_for":"Frequent diners spending $500+/month on restaurants",'
        '"confidence":0.95}'
    )
    client = _FakeLLMClient(raw)
    quality = ExplanationQualityFilter(max_chars=500, min_rationale_points=2)
    generator = ExplanationGenerator(llm_client=client, quality_filter=quality)

    out = generator.generate(
        explanation_type=ExplanationType.SINGLE_TRANSACTION,
        scoring_output=scoring_output_fixture,
        personalization_signals={"user_segment": "foodie"},
    )

    assert out.summary == "Use Amex Gold."
    assert len(out.pros) == 2
    assert len(out.cons) == 2
    assert out.best_for == "Frequent diners spending $500+/month on restaurants"
    assert out.quality_checks["length_ok"] is True
    assert out.quality_checks["relevance_ok"] is True
    assert out.quality_checks["pros_cons_ok"] is True
    assert out.used_fallback is False
    assert out.latency_ms >= 0.0
    # Telemetry fields
    assert out.prompt_hash != ""
    assert out.model_name == "gemini-2.5-flash"


def test_quality_filter_flags_low_relevance():
    """Relevance fails when the card name is not mentioned in the explanation text."""
    quality = ExplanationQualityFilter(max_chars=500, min_rationale_points=2)
    parsed = ParsedExplanation(
        summary="Use this great card.",
        pros=["good rate", "nice bonus"],
        cons=["annual fee", "limited categories"],
        confidence=0.8,
    )
    checks = quality.evaluate(
        parsed=parsed,
        context={"scoring": {"best_card": {"card_name": "Amex Gold"}}},
    )
    # "Amex Gold" not mentioned anywhere in summary or rationale
    assert checks["relevance_ok"] is False


def test_quality_filter_flags_hallucination():
    quality = ExplanationQualityFilter(max_chars=500, min_rationale_points=1)
    parsed = ParsedExplanation(
        summary="Use Amex Gold with 12x rewards.",
        pros=["12x at dining", "great value"],
        cons=["high fee", "limited network"],
        confidence=0.9,
    )
    checks = quality.evaluate(
        parsed=parsed,
        context={
            "scoring": {"best_card": {"card_name": "Amex Gold", "reward_rate": 4.0}}
        },
    )
    assert checks["hallucination_guard_ok"] is False


def test_quality_filter_flags_wrong_pros_cons_count():
    quality = ExplanationQualityFilter(max_chars=500, min_rationale_points=1)
    parsed = ParsedExplanation(
        summary="Use Amex Gold.",
        pros=["only one pro"],
        cons=["one con", "two con"],
        confidence=0.8,
    )
    checks = quality.evaluate(
        parsed=parsed,
        context={"scoring": {"best_card": {"card_name": "Amex Gold"}}},
    )
    assert checks["pros_cons_ok"] is False


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
    assert len(out.pros) == 2
    assert len(out.cons) == 2


def test_generator_falls_back_when_quality_fails(scoring_output_fixture):
    raw = (
        '{"summary":"Use Amex Gold with 12x rewards.",'
        '"pros":["12x on dining","great value"],'
        '"cons":["annual fee","limited"],'
        '"best_for":"diners",'
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
    assert len(out.pros) == 2
    assert len(out.cons) == 2


def test_fallback_template_produces_structured_output(scoring_output_fixture):
    """Fallback templates must produce the same 2-pro/2-con structure."""
    client = _FakeLLMClient("not-json")
    generator = ExplanationGenerator(llm_client=client)

    out = generator.generate(
        explanation_type=ExplanationType.SINGLE_TRANSACTION,
        scoring_output=scoring_output_fixture,
        personalization_signals={"active_personas": ["traveler"]},
    )

    assert out.used_fallback is True
    assert len(out.pros) == 2
    assert len(out.cons) == 2
    assert all(isinstance(p, str) and p for p in out.pros)
    assert all(isinstance(c, str) and c for c in out.cons)
    # rationale property still works for backward compat
    assert len(out.rationale) == 4
