"""Pydantic request/response shapes for user profile and card wallet endpoints."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Valid persona keys — must match config/personas.yaml (added in Story 1.3)
# ---------------------------------------------------------------------------
VALID_PERSONAS = {"student", "traveler", "family", "cashback-focused"}

# Valid reward preferences
VALID_REWARD_PREFERENCES = {"cashback", "points", "miles"}


class UserProfileResponse(BaseModel):
    user_id: int
    email: str
    display_name: str
    personas: List[str]
    reward_preference: str
    transaction_logging_enabled: bool
    dark_mode: bool
    saved_card_ids: List[str]


class ProfilePatchRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    personas: Optional[List[str]] = None
    reward_preference: Optional[str] = None
    transaction_logging_enabled: Optional[bool] = None
    dark_mode: Optional[bool] = None


class CardListRequest(BaseModel):
    card_ids: List[str]


class CardCatalogItem(BaseModel):
    card_id: str
    card_name: str
    issuer: str
    annual_fee: float
    reward_highlights: List[str]
    image_url: Optional[str] = None
