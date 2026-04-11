"""Pydantic request/response shapes for feedback endpoints (Story 4.1)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

VALID_REASON_TAGS = {
    "too_expensive",
    "not_relevant",
    "already_have",
    "explanation_unclear",
}


class FeedbackCreateRequest(BaseModel):
    card_id: str = Field(..., min_length=1)
    recommendation_event_id: Optional[int] = None
    reaction: Literal["like", "dislike"]
    reason_tag: Optional[str] = None
    target: Literal["card", "explanation"] = "card"

    @field_validator("reason_tag")
    @classmethod
    def validate_reason_tag(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_REASON_TAGS:
            raise ValueError(
                f"Invalid reason_tag '{v}'. Must be one of: {', '.join(sorted(VALID_REASON_TAGS))}"
            )
        return v


class FeedbackResponse(BaseModel):
    ok: bool
    feedback_id: int
