"""Validation utilities for LLM explanation quality (Story 4.3)."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class FactualAccuracyResult:
    """Validation result for factual consistency checks."""

    score: float
    passed: bool
    total_claims: int
    supported_claims: int
    unsupported_claims: List[str]


class FactualAccuracyChecker:
    """Check that numeric claims in explanation are grounded in context."""

    def __init__(
        self, min_score: float = 0.95, numeric_tolerance: float = 1e-2
    ) -> None:
        self.min_score = min_score
        self.numeric_tolerance = numeric_tolerance

    def evaluate(
        self, summary: str, rationale: Sequence[str], context: Dict[str, Any]
    ) -> FactualAccuracyResult:
        """Compute factual score from extracted numeric claims."""
        text = f"{summary} {' '.join(rationale)}"

        multiplier_claims = re.findall(r"\b(\d+(?:\.\d+)?)x\b", text.lower())
        amount_claims = re.findall(r"\$\s*(\d+(?:\.\d+)?)\b", text)

        total = len(multiplier_claims) + len(amount_claims)
        if total == 0:
            return FactualAccuracyResult(
                score=1.0,
                passed=True,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
            )

        valid_multipliers = self._valid_multipliers(context)
        valid_amounts = self._valid_amounts(context)

        supported = 0
        unsupported: List[str] = []

        for claim in multiplier_claims:
            val = float(claim)
            if any(abs(val - m) <= self.numeric_tolerance for m in valid_multipliers):
                supported += 1
            else:
                unsupported.append(f"{claim}x")

        for claim in amount_claims:
            val = float(claim)
            if any(abs(val - a) <= self.numeric_tolerance for a in valid_amounts):
                supported += 1
            else:
                unsupported.append(f"${claim}")

        score = supported / total
        return FactualAccuracyResult(
            score=score,
            passed=score >= self.min_score,
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
        )

    def _valid_multipliers(self, context: Dict[str, Any]) -> List[float]:
        scoring = context.get("scoring") or {}
        values: List[float] = []

        best = scoring.get("best_card") or {}
        if "reward_rate" in best:
            values.append(float(best["reward_rate"]))

        for alt in scoring.get("alternatives", []) or []:
            if isinstance(alt, dict) and "reward_rate" in alt:
                values.append(float(alt["reward_rate"]))

        return values

    def _valid_amounts(self, context: Dict[str, Any]) -> List[float]:
        scoring = context.get("scoring") or {}
        values: List[float] = []

        txn = scoring.get("transaction") or {}
        if "amount" in txn:
            values.append(float(txn["amount"]))

        best = scoring.get("best_card") or {}
        if "reward_amount" in best:
            values.append(float(best["reward_amount"]))

        for alt in scoring.get("alternatives", []) or []:
            if isinstance(alt, dict) and "reward_amount" in alt:
                values.append(float(alt["reward_amount"]))

        candidate = scoring.get("candidate_card") or {}
        if "annual_fee" in candidate:
            values.append(float(candidate["annual_fee"]))

        if "expected_gain" in scoring:
            values.append(float(scoring["expected_gain"]))

        return values


@dataclass(frozen=True)
class ReadabilityResult:
    """Readability scoring output."""

    flesch_reading_ease: float
    grade_level: float
    passed: bool


class ReadabilityScorer:
    """Compute readability scores with lightweight heuristics."""

    def __init__(
        self, min_flesch_score: float = 50.0, max_grade_level: float = 10.0
    ) -> None:
        self.min_flesch_score = min_flesch_score
        self.max_grade_level = max_grade_level

    def evaluate(self, summary: str, rationale: Sequence[str]) -> ReadabilityResult:
        """Score explanation readability."""
        text = f"{summary}. {' '.join(rationale)}"

        words = re.findall(r"[A-Za-z0-9']+", text)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

        word_count = max(len(words), 1)
        sentence_count = max(len(sentences), 1)
        syllables = max(sum(self._count_syllables(w) for w in words), 1)

        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllables / word_count

        flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59

        passed = flesch >= self.min_flesch_score and grade <= self.max_grade_level
        return ReadabilityResult(
            flesch_reading_ease=round(flesch, 2),
            grade_level=round(grade, 2),
            passed=passed,
        )

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        if not word:
            return 1

        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        if word.endswith("e") and count > 1:
            count -= 1

        return max(count, 1)
