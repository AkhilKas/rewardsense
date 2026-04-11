import json

import pytest

from src.model_pipeline.llm.prompt_builder import (
    ExplanationType,
    PromptBuilder,
    PromptRenderError,
)
from src.model_pipeline.llm.response_parser import ResponseParser, ResponseParseError


@pytest.fixture
def scoring_output():
    return {
        "transaction": {
            "amount": 120.0,
            "category": "dining",
            "merchant": "Chipotle",
        },
        "best_card": {
            "card_id": "amex_gold",
            "card_name": "Amex Gold",
            "reward_rate": 4.0,
            "reward_amount": 4.8,
        },
        "alternatives": [
            {"card_name": "Citi Double Cash", "reward_rate": 2.0, "reward_amount": 2.4}
        ],
    }


@pytest.fixture
def personalization_signals():
    return {
        "user_segment": "foodie_traveler",
        "point_value": 0.018,
        "preferences": {
            "avoid_annual_fee": False,
            "prefers_transferable_points": True,
        },
    }


class TestPromptBuilderTemplates:
    def test_single_transaction_template_renders(
        self, scoring_output, personalization_signals
    ):
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            explanation_type=ExplanationType.SINGLE_TRANSACTION,
            scoring_output=scoring_output,
            personalization_signals=personalization_signals,
        )

        assert "single transaction recommendation" in prompt.system_message.lower()
        assert "pros" in prompt.system_message.lower()
        assert "cons" in prompt.system_message.lower()
        assert "Chipotle" in prompt.user_message
        assert "Amex Gold" in prompt.user_message

    def test_prompt_includes_pros_cons_instructions(
        self, scoring_output, personalization_signals
    ):
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            explanation_type=ExplanationType.SINGLE_TRANSACTION,
            scoring_output=scoring_output,
            personalization_signals=personalization_signals,
        )
        assert "exactly 2 pros" in prompt.user_message.lower()
        assert "exactly 2 cons" in prompt.user_message.lower()

    def test_portfolio_optimization_template_renders(self):
        builder = PromptBuilder()
        scoring_output = {
            "portfolio_summary": {
                "current_cards": ["Citi Double Cash", "Amex Gold"],
                "annual_fee_total": 250,
                "optimization_goal": "maximize travel rewards",
            },
            "suggested_actions": [
                {
                    "action": "shift_spend",
                    "detail": "Move flights to Sapphire Preferred",
                }
            ],
        }

        prompt = builder.build_prompt(
            explanation_type=ExplanationType.PORTFOLIO_OPTIMIZATION,
            scoring_output=scoring_output,
            personalization_signals={"user_segment": "traveler"},
        )

        assert "portfolio optimization suggestion" in prompt.system_message.lower()
        assert "maximize travel rewards" in prompt.user_message

    def test_new_card_template_renders(self):
        builder = PromptBuilder()
        scoring_output = {
            "candidate_card": {
                "card_name": "Chase Sapphire Preferred",
                "annual_fee": 95,
                "welcome_bonus": "60,000 points",
            },
            "expected_gain": 320.0,
        }

        prompt = builder.build_prompt(
            explanation_type=ExplanationType.NEW_CARD_RECOMMENDATION,
            scoring_output=scoring_output,
            personalization_signals={"credit_profile": "good"},
        )

        assert "new card recommendation" in prompt.system_message.lower()
        assert "Chase Sapphire Preferred" in prompt.user_message

    def test_unsupported_type_raises(self, scoring_output):
        builder = PromptBuilder()
        with pytest.raises(PromptRenderError):
            builder.build_prompt(
                explanation_type="unknown",  # type: ignore[arg-type]
                scoring_output=scoring_output,
                personalization_signals={},
            )


class TestPersonaBlock:
    def test_persona_context_included_in_prompt(self, scoring_output):
        builder = PromptBuilder()
        signals = {
            "active_personas": ["traveler", "student"],
            "fee_sensitivity": "high",
            "saved_cards": ["Citi Double Cash", "Discover it"],
        }
        prompt = builder.build_prompt(
            explanation_type=ExplanationType.SINGLE_TRANSACTION,
            scoring_output=scoring_output,
            personalization_signals=signals,
        )
        assert "traveler" in prompt.user_message
        assert "Fee sensitivity: high" in prompt.user_message
        assert "Citi Double Cash" in prompt.user_message

    def test_empty_persona_block_when_no_signals(self, scoring_output):
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            explanation_type=ExplanationType.SINGLE_TRANSACTION,
            scoring_output=scoring_output,
            personalization_signals={},
        )
        assert "User context:" not in prompt.user_message


class TestContextInjection:
    def test_context_contains_scoring_and_personalization(
        self, scoring_output, personalization_signals
    ):
        builder = PromptBuilder()
        context = builder.build_context(scoring_output, personalization_signals)

        assert context["scoring"]["best_card"]["card_name"] == "Amex Gold"
        assert context["personalization"]["point_value"] == 0.018

    def test_missing_required_scoring_fields_raise(self, personalization_signals):
        builder = PromptBuilder()
        with pytest.raises(PromptRenderError):
            builder.build_prompt(
                explanation_type=ExplanationType.SINGLE_TRANSACTION,
                scoring_output={"best_card": {"card_name": "Amex Gold"}},
                personalization_signals=personalization_signals,
            )


class TestResponseParser:
    def test_parse_json_block(self):
        parser = ResponseParser()
        raw = """
Here is the explanation.
```json
{
  "summary": "Use Amex Gold for dining.",
  "pros": ["4x dining rewards", "User values transferable points"],
  "cons": ["$250 annual fee", "Limited cashback options"],
  "best_for": "Frequent diners",
  "confidence": 0.93,
  "disclaimers": ["Rates may change"]
}
```
"""
        parsed = parser.parse(raw)
        assert parsed.summary == "Use Amex Gold for dining."
        assert len(parsed.pros) == 2
        assert parsed.pros[0] == "4x dining rewards"
        assert len(parsed.cons) == 2
        assert parsed.best_for == "Frequent diners"
        assert parsed.confidence == pytest.approx(0.93)
        # backward compat
        assert len(parsed.rationale) == 4

    def test_parse_plain_json(self):
        parser = ResponseParser()
        raw = json.dumps(
            {
                "summary": "Pick Citi Double Cash.",
                "pros": ["No annual fee", "Strong baseline return"],
                "cons": ["No category bonuses", "No sign-up bonus"],
                "best_for": "Everyday spenders",
                "confidence": 0.88,
            }
        )
        parsed = parser.parse(raw)
        assert parsed.summary.startswith("Pick Citi")
        assert len(parsed.pros) == 2
        assert len(parsed.cons) == 2

    def test_invalid_json_raises(self):
        parser = ResponseParser()
        with pytest.raises(ResponseParseError):
            parser.parse("This is not JSON")

    def test_missing_required_fields_raises(self):
        parser = ResponseParser()
        with pytest.raises(ResponseParseError):
            parser.parse('{"summary": "x"}')

    def test_wrong_pros_count_raises(self):
        parser = ResponseParser()
        raw = json.dumps(
            {
                "summary": "Pick card.",
                "pros": ["only one"],
                "cons": ["a", "b"],
                "confidence": 0.8,
            }
        )
        with pytest.raises(ResponseParseError, match="exactly 2"):
            parser.parse(raw)

    def test_wrong_cons_count_raises(self):
        parser = ResponseParser()
        raw = json.dumps(
            {
                "summary": "Pick card.",
                "pros": ["a", "b"],
                "cons": ["only one"],
                "confidence": 0.8,
            }
        )
        with pytest.raises(ResponseParseError, match="exactly 2"):
            parser.parse(raw)
