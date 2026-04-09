"""User profile, settings, saved-cards, card catalog, and recommendation endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.app.auth.dependencies import get_current_user
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.personas.modifier import PersonaModifier
from src.model_pipeline.personalization.personalized_scorer import PersonalizedScorer
from src.app.users import service
from src.app.users.schemas import (
    CardCatalogItem,
    CardDisplayInfo,
    CardListRequest,
    CardSavingsDetail,
    CategorySavings,
    PersonaRecommendResponse,
    PortfolioRecommendRequest,
    ProfilePatchRequest,
    QuickTransactionRequest,
    QuickTransactionResponse,
    SavingsCalculatorRequest,
    SavingsCalculatorResponse,
    ScoreBreakdown,
    ScoredCard,
    TransactionRecommendRequest,
    UserProfileResponse,
)

router = APIRouter(tags=["users"])

logger = logging.getLogger(__name__)

# Loaded once at import time — reads config/personas.yaml
try:
    _persona_modifier: Optional[PersonaModifier] = PersonaModifier()
except Exception:
    logger.warning("PersonaModifier failed to load; persona adjustments disabled")
    _persona_modifier = None

# Scorer cached at module scope to avoid per-request instantiation
_scorer = PersonalizedScorer()

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
# Card catalog with image URLs keyed by stable card_id slug
# ---------------------------------------------------------------------------
_CARD_IMAGE_PATH = "/cards/{card_id}.svg"

_CATALOG: List[CardCatalogItem] = [
    CardCatalogItem(
        card_id="chase_sapphire_preferred",
        card_name="Chase Sapphire Preferred",
        issuer="Chase",
        annual_fee=95,
        reward_highlights=["3x dining", "2x travel", "1x everything else"],
        image_url=_CARD_IMAGE_PATH.format(card_id="chase_sapphire_preferred"),
    ),
    CardCatalogItem(
        card_id="amex_gold",
        card_name="Amex Gold",
        issuer="American Express",
        annual_fee=250,
        reward_highlights=["4x dining", "4x groceries", "3x travel"],
        image_url=_CARD_IMAGE_PATH.format(card_id="amex_gold"),
    ),
    CardCatalogItem(
        card_id="citi_double_cash",
        card_name="Citi Double Cash",
        issuer="Citi",
        annual_fee=0,
        reward_highlights=["2% cash back on everything"],
        image_url=_CARD_IMAGE_PATH.format(card_id="citi_double_cash"),
    ),
    CardCatalogItem(
        card_id="capital_one_venture",
        card_name="Capital One Venture",
        issuer="Capital One",
        annual_fee=95,
        reward_highlights=["5x travel via Capital One", "2x everything else"],
        image_url=_CARD_IMAGE_PATH.format(card_id="capital_one_venture"),
    ),
    CardCatalogItem(
        card_id="discover_it",
        card_name="Discover it Cash Back",
        issuer="Discover",
        annual_fee=0,
        reward_highlights=["5% rotating categories", "1% everything else"],
        image_url=_CARD_IMAGE_PATH.format(card_id="discover_it"),
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
    unknown = [cid for cid in payload.card_ids if cid not in _CATALOG_BY_ID]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown card_id(s): {unknown}. "
            f"Valid options: {sorted(_CATALOG_BY_ID)}",
        )
    return service.replace_saved_cards(db, current_user, payload.card_ids)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Catalog keyed by card_id for O(1) lookup
_CATALOG_BY_ID: Dict[str, CardCatalogItem] = {c.card_id: c for c in _CATALOG}


def _card_display_for(card_id: Optional[str]) -> Optional[CardDisplayInfo]:
    """Build a CardDisplayInfo from the catalog, or None if unknown."""
    if card_id is None:
        return None
    cat = _CATALOG_BY_ID.get(card_id)
    if cat is None:
        return None
    return CardDisplayInfo(
        card_id=cat.card_id,
        card_name=cat.card_name,
        issuer=cat.issuer,
        annual_fee=cat.annual_fee,
        reward_highlights=cat.reward_highlights,
        image_url=cat.image_url,
    )


# Scoring engine portfolio format for a given card_id
_SCORING_RATES: Dict[str, Dict[str, Any]] = {
    "chase_sapphire_preferred": {
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 3.0, "travel": 2.0},
        }
    },
    "amex_gold": {
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 4.0, "groceries": 4.0, "travel": 3.0},
        }
    },
    "citi_double_cash": {"reward_rates": {"universal_base_rate": 2.0}},
    "capital_one_venture": {
        "reward_rates": {
            "universal_base_rate": 2.0,
            "category_bonuses": {"travel": 5.0},
        }
    },
    "discover_it": {"reward_rates": {"universal_base_rate": 1.0}},
}


def _build_portfolio(card_ids: List[str]) -> List[Dict[str, Any]]:
    """Build scorer-compatible portfolio dicts from card IDs."""
    portfolio = []
    for cid in card_ids:
        catalog_card = _CATALOG_BY_ID.get(cid)
        if catalog_card is None:
            logger.warning("Saved card_id %r not in catalog — skipped", cid)
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


def _resolve_category_heuristic_first(merchant: str, hint: Optional[str]) -> str:
    """Heuristic lookup first; fall back to user-supplied category, then 'other'."""
    lower = merchant.lower()
    for category, keywords in _MERCHANT_CATEGORY_HINTS.items():
        if any(kw in lower for kw in keywords):
            return category
    if hint:
        return hint.lower()
    return "other"


def _run_recommendation(
    portfolio: List[Dict[str, Any]],
    transaction: Dict[str, Any],
    active_personas: List[str],
    is_generic: bool,
    monthly_spend: float = 0.0,
) -> PersonaRecommendResponse:
    """Score portfolio, apply persona modifier, return enriched response."""
    try:
        result = _scorer.score(portfolio=portfolio, transaction=transaction)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {exc}",
        ) from exc

    point_value: float = float(result.get("point_value", 0.01))
    ranked_raw: List[Dict[str, Any]] = result.get("ranked", [])
    category = str(transaction.get("category", "other"))

    if _persona_modifier is not None:
        modifier_result = _persona_modifier.apply(ranked_raw, active_personas, category)
        ranked_adjusted = modifier_result["ranked"]
        persona_context = modifier_result["persona_context"]
    else:
        ranked_adjusted = ranked_raw
        persona_context = ""

    # Build enriched ScoredCard list
    scored_cards: List[ScoredCard] = []
    for c in ranked_adjusted:
        raw_reward_amount = float(c.get("raw_reward_amount", 0.0))
        reward_rate = float(c.get("reward_rate", 0.0))
        adj = c.get("persona_adjustments") or {}

        breakdown = ScoreBreakdown(
            raw_reward_rate=reward_rate,
            raw_reward_amount=raw_reward_amount,
            personalization_multiplier=point_value,
            persona_category_boost=float(adj.get("category_boost_applied", 1.0)),
            persona_fee_penalty=float(adj.get("extra_fee_penalty", 0.0)),
        )

        # Projected annual savings based on actual dollar reward
        annual_factor = 12.0
        if monthly_spend > 0 and transaction.get("amount", 0):
            annual_factor = monthly_spend / float(transaction["amount"]) * 12.0
        projected_savings = round(raw_reward_amount * annual_factor, 2)

        reason = (
            _persona_modifier.card_persona_reason(c, active_personas, category)
            if _persona_modifier is not None
            else "No active persona \u2014 ranked by raw reward value."
        )

        cid = c.get("card_id")
        scored_cards.append(
            ScoredCard(
                card_id=cid,
                card_name=c.get("card_name", ""),
                reward_amount=float(c.get("reward_amount", 0.0)),
                annual_fee=float(c.get("annual_fee", 0.0)),
                rank=int(c.get("rank", 0)),
                persona_adjustments=c.get("persona_adjustments"),
                score_breakdown=breakdown,
                persona_match_reason=reason,
                projected_savings=projected_savings,
                card_display=_card_display_for(cid),
            )
        )

    best_card_id = scored_cards[0].card_id if scored_cards else None
    top_card = scored_cards[0] if scored_cards else None
    alternatives = scored_cards[1:] if len(scored_cards) > 1 else []

    return PersonaRecommendResponse(
        top_card=top_card,
        alternatives=alternatives,
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

    monthly_spend = float(payload.monthly_spend) if payload.monthly_spend else amount

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
        monthly_spend=monthly_spend,
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
        monthly_spend=float(payload.amount),
    )


@router.post(
    "/recommendations/quick-transaction",
    response_model=QuickTransactionResponse,
)
def recommend_quick_transaction(
    payload: QuickTransactionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QuickTransactionResponse:
    """Quick single-purchase recommendation using only saved cards."""
    from datetime import datetime

    profile = service.get_profile(db, current_user)

    has_saved_cards = len(profile.saved_card_ids) > 0
    if not has_saved_cards:
        return QuickTransactionResponse(
            top_card=None,
            alternatives=[],
            estimated_reward=0.0,
            money_saved=0.0,
            category_used="unknown",
            is_personalized=False,
            has_saved_cards=False,
            active_personas=profile.personas,
            persona_context="Add cards to your wallet to get recommendations.",
        )

    portfolio = _build_portfolio(profile.saved_card_ids)
    category = _resolve_category_heuristic_first(payload.merchant, payload.category)

    transaction: Dict[str, Any] = {
        "amount": float(payload.amount),
        "category": category,
        "merchant": payload.merchant,
    }
    if payload.date:
        try:
            transaction["date"] = datetime.fromisoformat(payload.date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid date format '{payload.date}'. Use ISO-8601 (YYYY-MM-DD).",
            )

    try:
        result = _scorer.score(portfolio=portfolio, transaction=transaction)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {exc}",
        ) from exc

    point_value: float = float(result.get("point_value", 0.01))
    ranked_raw: List[Dict[str, Any]] = result.get("ranked", [])

    if _persona_modifier is not None:
        modifier_result = _persona_modifier.apply(
            ranked_raw, profile.personas, category
        )
        ranked_adjusted = modifier_result["ranked"]
        persona_context = modifier_result["persona_context"]
    else:
        ranked_adjusted = ranked_raw
        persona_context = ""

    scored_cards: List[ScoredCard] = []
    for c in ranked_adjusted:
        raw_reward_amount = float(c.get("raw_reward_amount", 0.0))
        reward_rate = float(c.get("reward_rate", 0.0))
        adj = c.get("persona_adjustments") or {}

        breakdown = ScoreBreakdown(
            raw_reward_rate=reward_rate,
            raw_reward_amount=raw_reward_amount,
            personalization_multiplier=point_value,
            persona_category_boost=float(adj.get("category_boost_applied", 1.0)),
            persona_fee_penalty=float(adj.get("extra_fee_penalty", 0.0)),
        )
        reason = (
            _persona_modifier.card_persona_reason(c, profile.personas, category)
            if _persona_modifier is not None
            else "No active persona \u2014 ranked by raw reward value."
        )

        cid = c.get("card_id")
        scored_cards.append(
            ScoredCard(
                card_id=cid,
                card_name=c.get("card_name", ""),
                reward_amount=float(c.get("reward_amount", 0.0)),
                annual_fee=float(c.get("annual_fee", 0.0)),
                rank=int(c.get("rank", 0)),
                persona_adjustments=c.get("persona_adjustments"),
                score_breakdown=breakdown,
                persona_match_reason=reason,
                projected_savings=round(raw_reward_amount * 12.0, 2),
                card_display=_card_display_for(cid),
            )
        )

    top_card = scored_cards[0] if scored_cards else None
    alternatives = scored_cards[1:] if len(scored_cards) > 1 else []

    # estimated_reward / money_saved = raw dollar reward from the top card
    estimated_reward = 0.0
    if top_card and top_card.score_breakdown:
        estimated_reward = round(top_card.score_breakdown.raw_reward_amount, 2)

    return QuickTransactionResponse(
        top_card=top_card,
        alternatives=alternatives,
        estimated_reward=estimated_reward,
        money_saved=estimated_reward,
        category_used=category,
        is_personalized=result.get("is_personalized", False),
        has_saved_cards=True,
        active_personas=profile.personas,
        persona_context=persona_context,
    )


# ---------------------------------------------------------------------------
# Savings calculator
# ---------------------------------------------------------------------------

_GENERIC_BASELINE: Dict[str, Any] = {
    "card_id": None,
    "card_name": "Generic 1% Cashback",
    "annual_fee": 0,
    "reward_rates": {"universal_base_rate": 1.0},
}

_DEFAULT_CATEGORIES: Dict[str, float] = {
    "dining": 200.0,
    "groceries": 400.0,
    "travel": 150.0,
    "gas": 100.0,
    "other": 150.0,
}


def _is_catch_all(card_id: str) -> bool:
    """True when the card has no category bonuses (flat-rate only)."""
    rates = _SCORING_RATES.get(card_id, {}).get("reward_rates", {})
    bonuses = rates.get("category_bonuses")
    return not bonuses


def _find_baseline(saved_card_ids: List[str]) -> Dict[str, Any]:
    """Return the first saved catch-all card, or the generic 1% baseline."""
    for cid in saved_card_ids:
        if _is_catch_all(cid) and cid in _CATALOG_BY_ID:
            cat = _CATALOG_BY_ID[cid]
            rates = _SCORING_RATES.get(
                cid, {"reward_rates": {"universal_base_rate": 1.0}}
            )
            return {
                "card_id": cid,
                "card_name": cat.card_name,
                "annual_fee": cat.annual_fee,
                **rates,
            }
    return dict(_GENERIC_BASELINE)


def _reward_for_category(card: Dict[str, Any], category: str, amount: float) -> float:
    """Compute raw dollar reward for *card* in *category* at *amount*."""
    rates = card.get("reward_rates", {})
    bonuses = rates.get("category_bonuses", {})
    rate = bonuses.get(category, rates.get("universal_base_rate", 1.0))
    return amount * float(rate) / 100.0


@router.post(
    "/recommendations/savings-calculator",
    response_model=SavingsCalculatorResponse,
)
def savings_calculator(
    payload: SavingsCalculatorRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SavingsCalculatorResponse:
    """Category-by-category savings breakdown for every card in the wallet."""
    profile = service.get_profile(db, current_user)

    # Build spending profile
    spending = payload.spending_by_category
    if not spending:
        total = payload.monthly_spend if payload.monthly_spend > 0 else 1000.0
        weight_sum = sum(_DEFAULT_CATEGORIES.values())
        spending = {
            cat: round(total * amt / weight_sum, 2)
            for cat, amt in _DEFAULT_CATEGORIES.items()
        }
    total_monthly = sum(spending.values())

    # Resolve baseline
    baseline = _find_baseline(profile.saved_card_ids)
    baseline_annual_fee = float(baseline.get("annual_fee", 0))

    # Compute baseline rewards per category
    baseline_rewards: Dict[str, float] = {
        cat: _reward_for_category(baseline, cat, amt) for cat, amt in spending.items()
    }

    # Determine card set — saved cards, or full catalog if wallet empty
    card_ids = (
        profile.saved_card_ids
        if profile.saved_card_ids
        else [c.card_id for c in _CATALOG]
    )
    portfolio = _build_portfolio(card_ids)

    cards_out: List[CardSavingsDetail] = []
    for card in portfolio:
        cid = card["card_id"]
        cat_rows: List[CategorySavings] = []
        monthly_reward = 0.0
        monthly_uplift = 0.0

        for cat, amt in spending.items():
            reward = _reward_for_category(card, cat, amt)
            bl = baseline_rewards[cat]
            up = reward - bl
            cat_rows.append(
                CategorySavings(
                    category=cat,
                    monthly_spend=amt,
                    reward_amount=round(reward, 2),
                    baseline_reward=round(bl, 2),
                    uplift=round(up, 2),
                )
            )
            monthly_reward += reward
            monthly_uplift += up

        annual_reward = monthly_reward * 12
        annual_uplift = monthly_uplift * 12
        fee_diff = float(card.get("annual_fee", 0)) - baseline_annual_fee
        net_benefit = annual_uplift - fee_diff

        cards_out.append(
            CardSavingsDetail(
                card_id=cid,
                card_name=card.get("card_name", ""),
                annual_fee=float(card.get("annual_fee", 0)),
                card_display=_card_display_for(cid),
                categories=cat_rows,
                monthly_reward_total=round(monthly_reward, 2),
                annual_reward_total=round(annual_reward, 2),
                monthly_uplift_vs_baseline=round(monthly_uplift, 2),
                annual_uplift_vs_baseline=round(annual_uplift, 2),
                net_annual_benefit=round(net_benefit, 2),
            )
        )

    # Sort by net_annual_benefit descending
    cards_out.sort(key=lambda c: c.net_annual_benefit, reverse=True)

    return SavingsCalculatorResponse(
        cards=cards_out,
        baseline_card_id=baseline.get("card_id"),
        baseline_card_name=baseline["card_name"],
        baseline_annual_fee=baseline_annual_fee,
        spending_profile=spending,
        total_monthly_spend=round(total_monthly, 2),
    )
