"""User profile, settings, saved-cards, and card catalog endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.app.auth.dependencies import get_current_user
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.users import service
from src.app.users.schemas import (
    CardCatalogItem,
    CardListRequest,
    ProfilePatchRequest,
    UserProfileResponse,
)

router = APIRouter(tags=["users"])

# ---------------------------------------------------------------------------
# Card catalog — static for now; image_url populated in Story 2.4
# ---------------------------------------------------------------------------
_CATALOG: List[CardCatalogItem] = [
    CardCatalogItem(
        card_id="chase_sapphire_preferred",
        card_name="Chase Sapphire Preferred",
        issuer="Chase",
        annual_fee=95,
        reward_highlights=["3x dining", "2x travel", "1x everything else"],
        image_url=None,
    ),
    CardCatalogItem(
        card_id="amex_gold",
        card_name="Amex Gold",
        issuer="American Express",
        annual_fee=250,
        reward_highlights=["4x dining", "4x groceries", "3x travel"],
        image_url=None,
    ),
    CardCatalogItem(
        card_id="citi_double_cash",
        card_name="Citi Double Cash",
        issuer="Citi",
        annual_fee=0,
        reward_highlights=["2% cash back on everything"],
        image_url=None,
    ),
    CardCatalogItem(
        card_id="capital_one_venture",
        card_name="Capital One Venture",
        issuer="Capital One",
        annual_fee=95,
        reward_highlights=["5x travel via Capital One", "2x everything else"],
        image_url=None,
    ),
    CardCatalogItem(
        card_id="discover_it",
        card_name="Discover it Cash Back",
        issuer="Discover",
        annual_fee=0,
        reward_highlights=["5% rotating categories", "1% everything else"],
        image_url=None,
    ),
]


@router.get("/cards/catalog", response_model=List[CardCatalogItem])
def get_card_catalog() -> List[CardCatalogItem]:
    """Public endpoint — returns the curated card list."""
    return _CATALOG


@router.get("/me", response_model=UserProfileResponse)
def get_me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfileResponse:
    return service.get_profile(db, current_user)


@router.patch("/me/profile", response_model=UserProfileResponse)
def patch_profile(
    payload: ProfilePatchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfileResponse:
    return service.update_profile(db, current_user, payload)


@router.put("/me/cards", response_model=UserProfileResponse)
def put_saved_cards(
    payload: CardListRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfileResponse:
    return service.replace_saved_cards(db, current_user, payload.card_ids)
