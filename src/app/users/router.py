"""User profile, settings, saved-cards, card catalog, and recommendation endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.app.auth.dependencies import get_current_user
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.personas.modifier import PersonaModifier
from src.app.users import service
from src.app.users.schemas import (
    CardCatalogItem,
    CardListRequest,
    PersonaRecommendResponse,
    PortfolioRecommendRequest,
    ProfilePatchRequest,
    ScoredCard,
    TransactionRecommendRequest,
    UserProfileResponse,
)

router = APIRouter(tags=["users"])

# Loaded once at import time — reads config/personas.yaml
_persona_modifier = PersonaModifier()

# Simple keyword → category resolver for the transaction endpoint
_MERCHANT_CATEGORY_HINTS: Dict[str, List[str]] = {
    "dining": [
        "mcdonald",
        "starbucks",
        "chipotle",
        "restaurant",
        "cafe",
        "pizza",
        "burger",
        "sushi",
        "taco",
        "subway",
        "domino",
        "kfc",
    ],
    "groceries": [
        "walmart",
        "whole foods",
        "trader joe",
        "kroger",
        "safeway",
        "costco",
        "aldi",
        "publix",
        "wegmans",
        "target",
    ],
    "travel": [
        "delta",
        "united",
        "american airlines",
        "southwest",
        "marriott",
        "hilton",
        "hyatt",
        "airbnb",
        "expedia",
        "booking",
        "uber",
        "lyft",
    ],
    "gas": ["shell", "bp", "chevron", "exxon", "mobil", "texaco", "citgo"],
    "entertainment": [
        "netflix",
        "spotify",
        "hulu",
        "disney",
        "cinema",
        "amc",
        "regal",
        "theater",
        "ticketmaster",
    ],
    "online_shopping": ["amazon", "ebay", "etsy", "shopify", "wayfair"],
}

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Catalog keyed by card_id for O(1) lookup
_CATALOG_BY_ID: Dict[str, CardCatalogItem] = {c.card_id: c for c in _CATALOG}

# Scoring engine portfolio format for a given card_id
_SCORING_RATES: Dict[str, Dict[str, Any]] = {
    "chase_sapphire_preferred": {
        "reward_rates": {"dining": 3.0, "travel": 2.0, "universal_base_rate": 1.0}
    },
    "amex_gold": {
        "reward_rates": {
            "dining": 4.0,
            "groceries": 4.0,
            "travel": 3.0,
            "universal_base_rate": 1.0,
        }
    },
    "citi_double_cash": {"reward_rates": {"universal_base_rate": 2.0}},
    "capital_one_venture": {
        "reward_rates": {"travel": 5.0, "universal_base_rate": 2.0}
    },
    "discover_it": {"reward_rates": {"universal_base_rate": 1.0}},
}


def _build_portfolio(card_ids: List[str]) -> List[Dict[str, Any]]:
    """Build scorer-compatible portfolio dicts from card IDs."""
    portfolio = []
    for cid in card_ids:
        catalog_card = _CATALOG_BY_ID.get(cid)
        if catalog_card is None:
            continue
        rates = _SCORING_RATES.get(cid, {"reward_rates": {"universal_base_rate": 1.0}})
        portfolio.append(
            {
                "card_id": cid,
                "card_name": catalog_card.card_name,
                "annual_fee": catalog_card.annual_fee,
                **rates,
            }
        )
    return portfolio


def _resolve_category(merchant: str, hint: Optional[str]) -> str:
    """Return a category string from a merchant name or explicit hint."""
    if hint:
        return hint.lower()
    lower = merchant.lower()
    for category, keywords in _MERCHANT_CATEGORY_HINTS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "other"


def _run_recommendation(
    portfolio: List[Dict[str, Any]],
    transaction: Dict[str, Any],
    active_personas: List[str],
    is_generic: bool,
) -> PersonaRecommendResponse:
    """Score portfolio, apply persona modifier, return response."""
    from src.model_pipeline.personalization.personalized_scorer import (
        PersonalizedScorer,
    )

    scorer = PersonalizedScorer()
    try:
        result = scorer.score(portfolio=portfolio, transaction=transaction)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {exc}",
        ) from exc

    ranked_raw: List[Dict[str, Any]] = result.get("ranked", [])
    category = str(transaction.get("category", "other"))

    modifier_result = _persona_modifier.apply(ranked_raw, active_personas, category)
    ranked_adjusted = modifier_result["ranked"]
    persona_context = modifier_result["persona_context"]

    scored_cards = [
        ScoredCard(
            card_id=c.get("card_id"),
            card_name=c.get("card_name", ""),
            reward_amount=float(c.get("reward_amount", 0.0)),
            annual_fee=float(c.get("annual_fee", 0.0)),
            rank=int(c.get("rank", 0)),
            persona_adjustments=c.get("persona_adjustments"),
        )
        for c in ranked_adjusted
    ]

    best_card_id = scored_cards[0].card_id if scored_cards else None

    return PersonaRecommendResponse(
        ranked=scored_cards,
        best_card_id=best_card_id,
        is_personalized=result.get("is_personalized", False),
        is_generic=is_generic,
        active_personas=active_personas,
        persona_context=persona_context,
    )


# ---------------------------------------------------------------------------
# Recommendation endpoints
# ---------------------------------------------------------------------------


@router.post("/recommendations/portfolio", response_model=PersonaRecommendResponse)
def recommend_portfolio(
    payload: PortfolioRecommendRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PersonaRecommendResponse:
    """Recommend using the user's saved wallet and active personas."""
    profile = service.get_profile(db, current_user)

    # Build portfolio — fall back to full catalog if wallet is empty
    is_generic = len(profile.saved_card_ids) == 0
    card_ids = [c.card_id for c in _CATALOG] if is_generic else profile.saved_card_ids
    portfolio = _build_portfolio(card_ids)

    # Derive transaction from dominant spending category
    categories: Dict[str, float] = payload.spending_categories or {}
    if categories:
        dominant = max(categories, key=lambda k: categories[k])
        amount = float(categories[dominant])
    else:
        dominant = "other"
        amount = float(payload.monthly_spend) if payload.monthly_spend else 100.0

    transaction: Dict[str, Any] = {
        "amount": amount,
        "category": dominant,
        "merchant": f"{dominant}-merchant",
    }

    return _run_recommendation(
        portfolio=portfolio,
        transaction=transaction,
        active_personas=profile.personas,
        is_generic=is_generic,
    )


@router.post("/recommendations/transaction", response_model=PersonaRecommendResponse)
def recommend_transaction(
    payload: TransactionRecommendRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PersonaRecommendResponse:
    """Recommend best card for a single merchant transaction."""
    profile = service.get_profile(db, current_user)

    is_generic = len(profile.saved_card_ids) == 0
    card_ids = [c.card_id for c in _CATALOG] if is_generic else profile.saved_card_ids
    portfolio = _build_portfolio(card_ids)

    category = _resolve_category(payload.merchant, payload.category)
    transaction: Dict[str, Any] = {
        "amount": float(payload.amount),
        "category": category,
        "merchant": payload.merchant,
    }

    return _run_recommendation(
        portfolio=portfolio,
        transaction=transaction,
        active_personas=profile.personas,
        is_generic=is_generic,
    )
