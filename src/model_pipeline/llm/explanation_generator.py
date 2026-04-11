"""LLM explanation orchestration and quality filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
import time
from typing import Any, Dict, Optional, Protocol

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
    """Structured explanation with quality and telemetry metadata."""

    quality_checks: Dict[str, bool] = field(default_factory=dict)
    raw_response: str = ""
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0
    # --- Telemetry fields (Story 4.3) ---
    prompt_hash: str = ""
    model_name: str = ""
    temperature: float = 0.0
    token_estimate: Optional[int] = None


class ExplanationQualityFilter:
    """Lightweight quality guardrails for LLM explanation outputs."""

    def __init__(self, max_chars: int = 800, min_rationale_points: int = 2) -> None:
        self.max_chars = max_chars
        self.min_rationale_points = min_rationale_points

    def evaluate(
        self,
        parsed: ParsedExplanation,
        context: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Evaluate quality checks for length, relevance, structure, and hallucination risk."""
        total_len = len(parsed.summary) + sum(len(r) for r in parsed.rationale)
        length_ok = total_len <= self.max_chars

        best_card_name = (
            (context.get("scoring") or {}).get("best_card", {}).get("card_name", "")
        )
        relevance_ok = len(parsed.rationale) >= self.min_rationale_points
        if best_card_name:
            corpus = f"{parsed.summary} {' '.join(parsed.rationale)}".lower()
            relevance_ok = relevance_ok and best_card_name.lower() in corpus

        # v2 structure check: exactly 2 pros and 2 cons
        pros_cons_ok = len(parsed.pros) == 2 and len(parsed.cons) == 2

        hallucination_guard_ok = self._hallucination_guard(parsed, context)

        return {
            "length_ok": length_ok,
            "relevance_ok": relevance_ok,
            "pros_cons_ok": pros_cons_ok,
            "hallucination_guard_ok": hallucination_guard_ok,
        }

    def _hallucination_guard(
        self, parsed: ParsedExplanation, context: Dict[str, Any]
    ) -> bool:
        """Flag obviously impossible reward multipliers against known context."""
        text = f"{parsed.summary} {' '.join(parsed.rationale)}"
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
        annual_fee = best.get("annual_fee")
        merchant = txn.get("merchant", "this purchase")
        category = txn.get("category", "this category")

        # Derive persona info from personalization context
        personalization = context.get("personalization", {})
        personas = personalization.get("active_personas", [])

        if explanation_type == ExplanationType.SINGLE_TRANSACTION:
            summary = f"Use {best_card_name} for {merchant}."
            pros = [
                f"Top-ranked option for {category} in current scoring output.",
                (
                    f"Provides up to {reward_rate}x rewards in this category."
                    if reward_rate is not None
                    else "Strong reward rate for this spending category."
                ),
            ]
            cons = [
                (
                    f"Annual fee of ${annual_fee} may offset rewards for low spenders."
                    if annual_fee and annual_fee > 0
                    else "Reward rates may vary by merchant within this category."
                ),
                "Other cards may offer better rates in different spending categories.",
            ]
            best_for = (
                f"Frequent {category} spenders" if category != "this category" else ""
            )
        elif explanation_type == ExplanationType.PORTFOLIO_OPTIMIZATION:
            summary = f"Prioritize {best_card_name} in your portfolio strategy."
            pros = [
                "Strongest option in the current portfolio optimization context.",
                "Rebalancing spend toward this card improves expected reward value.",
            ]
            cons = [
                "Concentrating spend on one card may miss category-specific bonuses elsewhere.",
                (
                    f"Annual fee of ${annual_fee} requires sufficient spend to break even."
                    if annual_fee and annual_fee > 0
                    else "Reward structure may change with card issuer updates."
                ),
            ]
            best_for = f"Users focused on {', '.join(personas)}" if personas else ""
        else:
            summary = f"{best_card_name} is the strongest next-card candidate."
            pros = [
                "Current scoring indicates better projected rewards versus alternatives.",
                (
                    f"Competitive reward rate of {reward_rate}x in key categories."
                    if reward_rate is not None
                    else "Well-rounded reward structure across spending categories."
                ),
            ]
            cons = [
                "Review annual fee and benefit fit against your spending profile.",
                "Adding a new card affects your credit profile temporarily.",
            ]
            best_for = f"Users focused on {', '.join(personas)}" if personas else ""

        disclaimers = ["Generated from deterministic fallback template."]
        if reason:
            disclaimers.append(f"Fallback reason: {reason}")

        return ParsedExplanation(
            summary=summary,
            pros=pros,
            cons=cons,
            best_for=best_for,
            confidence=0.6,
            disclaimers=disclaimers,
        )


def _compute_prompt_hash(system_message: str, user_message: str) -> str:
    """SHA-256 hash of the prompt text for drift tracking."""
    content = f"{system_message}\n---\n{user_message}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


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
        temperature: float = 0.0,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.parser = parser or ResponseParser()
        self.quality_filter = quality_filter or ExplanationQualityFilter()
        self.fallback_generator = fallback_generator or TemplateFallbackGenerator()
        self.enforce_quality = enforce_quality
        self.model_name = model_name
        self.temperature = temperature

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

        prompt_hash = _compute_prompt_hash(prompt.system_message, prompt.user_message)
        effective_temp = self.temperature
        request_kwargs: Dict[str, Any] = {"model": self.model_name}
        if llm_params:
            request_kwargs.update(llm_params)
            effective_temp = llm_params.get("temperature", self.temperature)

        try:
            raw_response = self.llm_client.generate(
                system_message=prompt.system_message,
                user_message=prompt.user_message,
                **request_kwargs,
            )
            parsed = self.parser.parse(raw_response)

            checks = self.quality_filter.evaluate(
                parsed=parsed,
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
                    parsed=fallback,
                    context=prompt.context,
                )
                return GeneratedExplanation(
                    summary=fallback.summary,
                    pros=fallback.pros,
                    cons=fallback.cons,
                    best_for=fallback.best_for,
                    confidence=fallback.confidence,
                    disclaimers=fallback.disclaimers,
                    quality_checks=fallback_checks,
                    raw_response=raw_response,
                    used_fallback=True,
                    fallback_reason=reason,
                    latency_ms=round((time.perf_counter() - start) * 1000, 3),
                    prompt_hash=prompt_hash,
                    model_name=self.model_name,
                    temperature=effective_temp,
                )

            return GeneratedExplanation(
                summary=parsed.summary,
                pros=parsed.pros,
                cons=parsed.cons,
                best_for=parsed.best_for,
                confidence=parsed.confidence,
                disclaimers=parsed.disclaimers,
                quality_checks=checks,
                raw_response=raw_response,
                used_fallback=False,
                fallback_reason=None,
                latency_ms=round((time.perf_counter() - start) * 1000, 3),
                prompt_hash=prompt_hash,
                model_name=self.model_name,
                temperature=effective_temp,
            )
        except (ResponseParseError, RuntimeError, ValueError) as exc:
            fallback = self.fallback_generator.generate(
                explanation_type=explanation_type,
                context=prompt.context,
                reason=str(exc),
            )
            checks = self.quality_filter.evaluate(
                parsed=fallback,
                context=prompt.context,
            )
            return GeneratedExplanation(
                summary=fallback.summary,
                pros=fallback.pros,
                cons=fallback.cons,
                best_for=fallback.best_for,
                confidence=fallback.confidence,
                disclaimers=fallback.disclaimers,
                quality_checks=checks,
                raw_response="",
                used_fallback=True,
                fallback_reason=str(exc),
                latency_ms=round((time.perf_counter() - start) * 1000, 3),
                prompt_hash=prompt_hash,
                model_name=self.model_name,
                temperature=effective_temp,
            )
