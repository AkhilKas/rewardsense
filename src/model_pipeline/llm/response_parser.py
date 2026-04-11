"""Parses LLM outputs into structured explanation components."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List


class ResponseParseError(ValueError):
    """Raised when the LLM response cannot be parsed safely."""


@dataclass(frozen=True)
class ParsedExplanation:
    """Structured explanation extracted from raw LLM output.

    The v2 contract requires exactly 2 pros, 2 cons, and an optional
    ``best_for`` line.  ``rationale`` is kept for backward compatibility
    and is derived from ``pros + cons`` when not provided directly.
    """

    summary: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    best_for: str = ""
    confidence: float = 0.0
    disclaimers: List[str] = field(default_factory=list)

    @property
    def rationale(self) -> List[str]:
        """Backward-compatible rationale derived from pros + cons."""
        return list(self.pros) + list(self.cons)


class ResponseParser:
    """Extract and validate JSON explanation payloads from model text."""

    _JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)

    def parse(self, raw_output: str) -> ParsedExplanation:
        """Parse raw LLM output into a validated ParsedExplanation."""
        payload = self._parse_json_payload(raw_output)
        return self._validate(payload)

    def _parse_json_payload(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        block_match = self._JSON_BLOCK_RE.search(text)
        if block_match:
            candidate = block_match.group(1)
            return self._safe_json_load(candidate)

        if text.startswith("{") and text.endswith("}"):
            return self._safe_json_load(text)

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return self._safe_json_load(text[first : last + 1])

        raise ResponseParseError("No JSON object found in LLM response")

    def _safe_json_load(self, raw_json: str) -> Dict[str, Any]:
        try:
            out = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ResponseParseError(f"Invalid JSON from LLM: {exc}") from exc

        if not isinstance(out, dict):
            raise ResponseParseError("Parsed explanation payload must be a JSON object")
        return out

    def _validate(self, payload: Dict[str, Any]) -> ParsedExplanation:
        summary = payload.get("summary")
        pros = payload.get("pros")
        cons = payload.get("cons")
        best_for = payload.get("best_for", "")
        confidence = payload.get("confidence")
        disclaimers = payload.get("disclaimers", [])

        if not isinstance(summary, str) or not summary.strip():
            raise ResponseParseError("Missing or invalid required field: summary")

        # --- Pros validation (exactly 2) ---
        if not isinstance(pros, list) or len(pros) != 2:
            raise ResponseParseError("Field 'pros' must be a list of exactly 2 strings")
        if not all(isinstance(item, str) and item.strip() for item in pros):
            raise ResponseParseError("Each pro must be a non-empty string")

        # --- Cons validation (exactly 2) ---
        if not isinstance(cons, list) or len(cons) != 2:
            raise ResponseParseError("Field 'cons' must be a list of exactly 2 strings")
        if not all(isinstance(item, str) and item.strip() for item in cons):
            raise ResponseParseError("Each con must be a non-empty string")

        # --- best_for (optional string) ---
        if not isinstance(best_for, str):
            raise ResponseParseError("Field 'best_for' must be a string")

        # --- confidence ---
        if not isinstance(confidence, (int, float)):
            raise ResponseParseError("Missing or invalid required field: confidence")
        confidence = float(confidence)
        if confidence < 0.0 or confidence > 1.0:
            raise ResponseParseError("Confidence must be between 0 and 1")

        # --- disclaimers ---
        if not isinstance(disclaimers, list) or not all(
            isinstance(item, str) for item in disclaimers
        ):
            raise ResponseParseError("Disclaimers must be a list of strings")

        return ParsedExplanation(
            summary=summary.strip(),
            pros=[item.strip() for item in pros],
            cons=[item.strip() for item in cons],
            best_for=best_for.strip(),
            confidence=confidence,
            disclaimers=disclaimers,
        )
