"""LLM explanation orchestration and quality filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, List, Optional, Protocol

from src.model_pipeline.llm.prompt_builder import (
    ExplanationType,
    PromptBuilder,
)
from src.model_pipeline.llm.response_parser import (
    ParsedExplanation,
    ResponseParseError,
    ResponseParser,
)


class LLMClient(Protocol):
    """Protocol for pluggable LLM backends."""

    def generate(self, system_message: str, user_message: str, **kwargs: Any) -> str:
        """Return raw text from an LLM call."""


@dataclass(frozen=True)
class GeneratedExplanation(ParsedExplanation):
    """Structured explanation with quality metadata."""

    quality_checks: Dict[str, bool] = field(default_factory=dict)
    raw_response: str = ""
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0


class ExplanationQualityFilter:
    """Lightweight quality guardrails for LLM explanation outputs."""

    def __init__(self, max_chars: int = 800, min_rationale_points: int = 2) -> None:
        self.max_chars = max_chars
        self.min_rationale_points = min_rationale_points

    def evaluate(
        self,
        summary: str,
        rationale: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Evaluate quality checks for length, relevance, and hallucination risk."""
        length_ok = len(summary) + sum(len(r) for r in rationale) <= self.max_chars

        best_card_name = (
            (context.get("scoring") or {}).get("best_card", {}).get("card_name", "")
        )
        relevance_ok = len(rationale) >= self.min_rationale_points
        if best_card_name:
            corpus = f"{summary} {' '.join(rationale)}".lower()
            relevance_ok = relevance_ok and best_card_name.lower() in corpus

        hallucination_guard_ok = self._hallucination_guard(summary, rationale, context)

        return {
            "length_ok": length_ok,
            "relevance_ok": relevance_ok,
            "hallucination_guard_ok": hallucination_guard_ok,
        }

    def _hallucination_guard(
        self, summary: str, rationale: List[str], context: Dict[str, Any]
    ) -> bool:
        """Flag obviously impossible reward multipliers against known context."""
        text = f"{summary} {' '.join(rationale)}"
        claimed_multipliers = [
            int(v) for v in re.findall(r"\b(\d{1,2})x\b", text.lower())
        ]

        best_rate = (
            (context.get("scoring") or {}).get("best_card", {}).get("reward_rate")
        )
        if best_rate is None:
            return True

        try:
            max_allowed = float(best_rate)
        except (TypeError, ValueError):
            return True

        return all(multiplier <= max_allowed for multiplier in claimed_multipliers)


class TemplateFallbackGenerator:
    """Deterministic fallback explanations when model output is unavailable/unsafe."""

    def generate(
        self,
        explanation_type: ExplanationType,
        context: Dict[str, Any],
        reason: Optional[str] = None,
    ) -> ParsedExplanation:
        scoring = context.get("scoring", {})
        best = scoring.get("best_card", {})
        txn = scoring.get("transaction", {})
        best_card_name = best.get("card_name", "the recommended card")
        reward_rate = best.get("reward_rate")
        merchant = txn.get("merchant", "this purchase")
        category = txn.get("category", "this category")

        if explanation_type == ExplanationType.SINGLE_TRANSACTION:
            summary = f"Use {best_card_name} for {merchant}."
            rationale = [
                f"It is the top-ranked option for {category} in current scoring output.",
            ]
            if reward_rate is not None:
                rationale.append(
                    f"It provides up to {reward_rate}x rewards in this context."
                )
        elif explanation_type == ExplanationType.PORTFOLIO_OPTIMIZATION:
            summary = f"Prioritize {best_card_name} in your portfolio strategy."
            rationale = [
                "It appears as the strongest option in the current optimization context.",
                "Rebalance spend toward top-ranked cards to improve expected value.",
            ]
        else:
            summary = f"{best_card_name} is the strongest next-card candidate."
            rationale = [
                "Current scoring indicates better projected rewards versus alternatives.",
                "Review annual fee and benefit fit against your spending profile.",
            ]

        disclaimers = ["Generated from deterministic fallback template."]
        if reason:
            disclaimers.append(f"Fallback reason: {reason}")

        return ParsedExplanation(
            summary=summary,
            rationale=rationale,
            confidence=0.6,
            disclaimers=disclaimers,
        )


class ExplanationGenerator:
    """Coordinates prompt building, model invocation, and structured parsing."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: Optional[PromptBuilder] = None,
        parser: Optional[ResponseParser] = None,
        quality_filter: Optional[ExplanationQualityFilter] = None,
        fallback_generator: Optional[TemplateFallbackGenerator] = None,
        enforce_quality: bool = True,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.parser = parser or ResponseParser()
        self.quality_filter = quality_filter or ExplanationQualityFilter()
        self.fallback_generator = fallback_generator or TemplateFallbackGenerator()
        self.enforce_quality = enforce_quality
        self.model_name = model_name

    def generate(
        self,
        explanation_type: ExplanationType,
        scoring_output: Dict[str, Any],
        personalization_signals: Dict[str, Any],
        llm_params: Optional[Dict[str, Any]] = None,
    ) -> GeneratedExplanation:
        """Generate a validated explanation for a recommendation scenario."""
        start = time.perf_counter()
        prompt = self.prompt_builder.build_prompt(
            explanation_type=explanation_type,
            scoring_output=scoring_output,
            personalization_signals=personalization_signals,
        )

        request_kwargs = {"model": self.model_name}
        if llm_params:
            request_kwargs.update(llm_params)

        try:
            raw_response = self.llm_client.generate(
                system_message=prompt.system_message,
                user_message=prompt.user_message,
                **request_kwargs,
            )
            parsed = self.parser.parse(raw_response)

            checks = self.quality_filter.evaluate(
                summary=parsed.summary,
                rationale=parsed.rationale,
                context=prompt.context,
            )

            if self.enforce_quality and not all(checks.values()):
                reason = "quality_filter_failed"
                fallback = self.fallback_generator.generate(
                    explanation_type=explanation_type,
                    context=prompt.context,
                    reason=reason,
                )
                fallback_checks = self.quality_filter.evaluate(
                    summary=fallback.summary,
                    rationale=fallback.rationale,
                    context=prompt.context,
                )
                return GeneratedExplanation(
                    summary=fallback.summary,
                    rationale=fallback.rationale,
                    confidence=fallback.confidence,
                    disclaimers=fallback.disclaimers,
                    quality_checks=fallback_checks,
                    raw_response=raw_response,
                    used_fallback=True,
                    fallback_reason=reason,
                    latency_ms=round((time.perf_counter() - start) * 1000, 3),
                )

            return GeneratedExplanation(
                summary=parsed.summary,
                rationale=parsed.rationale,
                confidence=parsed.confidence,
                disclaimers=parsed.disclaimers,
                quality_checks=checks,
                raw_response=raw_response,
                used_fallback=False,
                fallback_reason=None,
                latency_ms=round((time.perf_counter() - start) * 1000, 3),
            )
        except (ResponseParseError, RuntimeError, ValueError) as exc:
            fallback = self.fallback_generator.generate(
                explanation_type=explanation_type,
                context=prompt.context,
                reason=str(exc),
            )
            checks = self.quality_filter.evaluate(
                summary=fallback.summary,
                rationale=fallback.rationale,
                context=prompt.context,
            )
            return GeneratedExplanation(
                summary=fallback.summary,
                rationale=fallback.rationale,
                confidence=fallback.confidence,
                disclaimers=fallback.disclaimers,
                quality_checks=checks,
                raw_response="",
                used_fallback=True,
                fallback_reason=str(exc),
                latency_ms=round((time.perf_counter() - start) * 1000, 3),
            )
