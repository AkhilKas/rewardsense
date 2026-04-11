"""Persist per-explanation telemetry for prompt drift tracking.

Story 4.3 — internal logging separate from user-facing history.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sqlalchemy.orm import Session

from src.app.db.models import LLMTelemetryEvent

logger = logging.getLogger(__name__)


def log_llm_telemetry(
    db: Session,
    *,
    recommendation_event_id: Optional[int] = None,
    card_id: Optional[str] = None,
    prompt_version_hash: str,
    model_name: str,
    temperature: float = 0.0,
    latency_ms: float,
    used_fallback: bool = False,
    fallback_reason: Optional[str] = None,
    token_estimate: Optional[int] = None,
    cost_estimate_usd: Optional[float] = None,
    output_quality_score: Optional[float] = None,
) -> LLMTelemetryEvent:
    """Persist a telemetry row and return it."""
    event = LLMTelemetryEvent(
        recommendation_event_id=recommendation_event_id,
        card_id=card_id,
        prompt_version_hash=prompt_version_hash,
        model_name=model_name,
        temperature=temperature,
        latency_ms=latency_ms,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        token_estimate=token_estimate,
        cost_estimate_usd=cost_estimate_usd,
        output_quality_score=output_quality_score,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    logger.debug(
        "LLM telemetry logged: id=%s prompt_hash=%s fallback=%s",
        event.id,
        prompt_version_hash,
        used_fallback,
    )
    return event


def compute_quality_score(quality_checks: Dict[str, bool]) -> float:
    """Derive a 0-1 quality score from the quality check dict."""
    if not quality_checks:
        return 0.0
    passed = sum(1 for v in quality_checks.values() if v)
    return round(passed / len(quality_checks), 3)
