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


class PromptBuilder:
    """Build structured prompts from scoring + personalization context."""

    _SYSTEM_BY_TYPE = {
        ExplanationType.SINGLE_TRANSACTION: (
            "You are a financial assistant generating a single transaction "
            "recommendation explanation. Return strict JSON with keys: "
            "summary (string), rationale (array of strings), confidence (0-1), "
            "disclaimers (array of strings, optional)."
        ),
        ExplanationType.PORTFOLIO_OPTIMIZATION: (
            "You are a financial assistant generating a portfolio optimization "
            "suggestion explanation. Return strict JSON with keys: summary, "
            "rationale, confidence, disclaimers."
        ),
        ExplanationType.NEW_CARD_RECOMMENDATION: (
            "You are a financial assistant generating a new card recommendation "
            "explanation. Return strict JSON with keys: summary, rationale, "
            "confidence, disclaimers."
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

        user_message = (
            f"Generate a {explanation_type.value.replace('_', ' ')} using only this context.\n"
            f"Context JSON:\n{json.dumps(context, indent=2, sort_keys=True)}\n"
            "Rules:\n"
            "1. Do not invent rates, fees, or bonuses not present in context.\n"
            "2. Keep summary concise and actionable.\n"
            "3. Mention key trade-offs when relevant."
        )

        return BuiltPrompt(
            explanation_type=explanation_type,
            system_message=self._SYSTEM_BY_TYPE[explanation_type],
            user_message=user_message,
            context=context,
        )

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
