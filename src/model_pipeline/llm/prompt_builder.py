"""Prompt engineering framework for LLM explanation generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict


class PromptRenderError(ValueError):
    """Raised when a prompt cannot be rendered from provided context."""


class ExplanationType(str, Enum):
    """Supported explanation types for Epic 4."""

    SINGLE_TRANSACTION = "single_transaction_recommendation"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization_suggestion"
    NEW_CARD_RECOMMENDATION = "new_card_recommendation"


@dataclass(frozen=True)
class BuiltPrompt:
    """Rendered prompt payload sent to the LLM client."""

    explanation_type: ExplanationType
    system_message: str
    user_message: str
    context: Dict[str, Any]


_OUTPUT_SCHEMA_INSTRUCTION = (
    "Return strict JSON with keys: "
    "summary (string), pros (array of exactly 2 strings), cons (array of exactly 2 strings), "
    "best_for (string — one sentence describing who this card is best for, or empty string), "
    "confidence (number 0-1), disclaimers (array of strings, optional)."
)


class PromptBuilder:
    """Build structured prompts from scoring + personalization context."""

    _SYSTEM_BY_TYPE = {
        ExplanationType.SINGLE_TRANSACTION: (
            "You are a financial assistant generating a single transaction "
            "recommendation explanation. " + _OUTPUT_SCHEMA_INSTRUCTION
        ),
        ExplanationType.PORTFOLIO_OPTIMIZATION: (
            "You are a financial assistant generating a portfolio optimization "
            "suggestion explanation. " + _OUTPUT_SCHEMA_INSTRUCTION
        ),
        ExplanationType.NEW_CARD_RECOMMENDATION: (
            "You are a financial assistant generating a new card recommendation "
            "explanation. " + _OUTPUT_SCHEMA_INSTRUCTION
        ),
    }

    def build_context(
        self,
        scoring_output: Dict[str, Any],
        personalization_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compose a structured context object for prompt templates."""
        return {
            "scoring": scoring_output,
            "personalization": personalization_signals,
        }

    def build_prompt(
        self,
        explanation_type: ExplanationType,
        scoring_output: Dict[str, Any],
        personalization_signals: Dict[str, Any],
    ) -> BuiltPrompt:
        """Render a prompt for the requested explanation type."""
        if explanation_type not in self._SYSTEM_BY_TYPE:
            raise PromptRenderError(f"Unsupported explanation type: {explanation_type}")

        self._validate_required_fields(explanation_type, scoring_output)
        context = self.build_context(scoring_output, personalization_signals)

        # Build enriched persona / fee / wallet context block
        persona_block = self._build_persona_block(personalization_signals)

        user_message = (
            f"Generate a {explanation_type.value.replace('_', ' ')} using only this context.\n"
            f"Context JSON:\n{json.dumps(context, indent=2, sort_keys=True)}\n"
            f"{persona_block}"
            "Rules:\n"
            "1. Do not invent rates, fees, or bonuses not present in context.\n"
            "2. Keep summary concise and actionable.\n"
            "3. Mention key trade-offs when relevant.\n"
            "4. Return exactly 2 pros and exactly 2 cons.\n"
            "5. Pros should highlight specific, concrete benefits for this user.\n"
            "6. Cons should mention genuine trade-offs (fees, limitations, opportunity costs).\n"
            '7. Include a "best_for" line describing the ideal user profile for this card.\n'
        )

        return BuiltPrompt(
            explanation_type=explanation_type,
            system_message=self._SYSTEM_BY_TYPE[explanation_type],
            user_message=user_message,
            context=context,
        )

    def _build_persona_block(self, personalization_signals: Dict[str, Any]) -> str:
        """Build a human-readable block of persona, fee, and wallet context."""
        parts: list[str] = []

        personas = personalization_signals.get("active_personas")
        if personas:
            parts.append(f"User personas: {', '.join(personas)}")

        fee_sensitivity = personalization_signals.get("fee_sensitivity")
        if fee_sensitivity:
            parts.append(f"Fee sensitivity: {fee_sensitivity}")

        saved_cards = personalization_signals.get("saved_cards")
        if saved_cards:
            parts.append(f"Cards already in wallet: {', '.join(saved_cards)}")

        score_breakdown = personalization_signals.get("score_breakdown")
        if score_breakdown:
            parts.append(f"Score breakdown: {json.dumps(score_breakdown)}")

        if not parts:
            return ""
        return "User context:\n" + "\n".join(f"- {p}" for p in parts) + "\n"

    def _validate_required_fields(
        self,
        explanation_type: ExplanationType,
        scoring_output: Dict[str, Any],
    ) -> None:
        if explanation_type == ExplanationType.SINGLE_TRANSACTION:
            txn = scoring_output.get("transaction") or {}
            best = scoring_output.get("best_card") or {}
            required = [
                (txn, "merchant"),
                (txn, "category"),
                (best, "card_name"),
            ]
            missing = [key for container, key in required if key not in container]
            if missing:
                raise PromptRenderError(
                    "Missing required scoring fields for single transaction: "
                    + ", ".join(missing)
                )
        elif explanation_type == ExplanationType.PORTFOLIO_OPTIMIZATION:
            if "portfolio_summary" not in scoring_output:
                raise PromptRenderError(
                    "Missing required scoring field for portfolio optimization: "
                    "portfolio_summary"
                )
        elif explanation_type == ExplanationType.NEW_CARD_RECOMMENDATION:
            if "candidate_card" not in scoring_output:
                raise PromptRenderError(
                    "Missing required scoring field for new card recommendation: "
                    "candidate_card"
                )
