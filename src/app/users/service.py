"""User profile and saved-card CRUD operations."""

from __future__ import annotations

from typing import List

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from src.app.db.models import SavedCard, User, UserPersona, UserSettings
from src.app.users.schemas import (
    VALID_PERSONAS,
    VALID_REWARD_PREFERENCES,
    ProfilePatchRequest,
    UserProfileResponse,
)


def _to_profile_response(user: User) -> UserProfileResponse:
    settings = user.settings
    return UserProfileResponse(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        personas=[p.persona_key for p in user.personas],
        reward_preference=settings.reward_preference if settings else "cashback",
        transaction_logging_enabled=(
            settings.transaction_logging_enabled if settings else False
        ),
        dark_mode=settings.dark_mode if settings else False,
        saved_card_ids=[c.card_id for c in user.saved_cards],
    )


def get_profile(db: Session, user: User) -> UserProfileResponse:
    return _to_profile_response(user)


def update_profile(
    db: Session, user: User, patch: ProfilePatchRequest
) -> UserProfileResponse:
    if patch.display_name is not None:
        user.display_name = patch.display_name

    if patch.personas is not None:
        unknown = set(patch.personas) - VALID_PERSONAS
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown persona(s): {sorted(unknown)}. "
                f"Valid options: {sorted(VALID_PERSONAS)}",
            )
        db.query(UserPersona).filter(UserPersona.user_id == user.id).delete()
        for key in set(patch.personas):
            db.add(UserPersona(user_id=user.id, persona_key=key))

    if user.settings is None:
        db.add(UserSettings(user_id=user.id))
        db.flush()

    if patch.reward_preference is not None:
        if patch.reward_preference not in VALID_REWARD_PREFERENCES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid reward_preference '{patch.reward_preference}'. "
                f"Valid options: {sorted(VALID_REWARD_PREFERENCES)}",
            )
        user.settings.reward_preference = patch.reward_preference

    if patch.transaction_logging_enabled is not None:
        user.settings.transaction_logging_enabled = patch.transaction_logging_enabled

    if patch.dark_mode is not None:
        user.settings.dark_mode = patch.dark_mode

    db.commit()
    db.refresh(user)
    return _to_profile_response(user)


def replace_saved_cards(
    db: Session, user: User, card_ids: List[str]
) -> UserProfileResponse:
    db.query(SavedCard).filter(SavedCard.user_id == user.id).delete()
    for card_id in card_ids:
        db.add(SavedCard(user_id=user.id, card_id=card_id))
    db.commit()
    db.refresh(user)
    return _to_profile_response(user)
