"""User profile, settings, saved-cards, card catalog, and recommendation endpoints."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.app.auth.dependencies import get_current_user
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.personas.modifier import PersonaModifier
from src.model_pipeline.personalization.personalized_scorer import PersonalizedScorer
from src.app.users import service
from src.app.cards.catalog import (
    DISPLAY_CATALOG,
    DISPLAY_CATALOG_BY_ID,
    get_scoring_rates,
)
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

# Scorer cached at module scope to avoid per-request instantiation.
# Tries to load a trained model from the local cache written by fetch_catalog.sh
# at container startup. Falls back to cold-start (point_value=0.01) if absent.
_MODEL_CACHE_PATH = os.getenv("MODEL_ARTIFACT_PATH", "/tmp/model_cache/model.joblib")
try:
    _scorer = PersonalizedScorer.from_artifact(_MODEL_CACHE_PATH)
    logger.info("Loaded personalization model from %s", _MODEL_CACHE_PATH)
except Exception as _model_exc:
    logger.warning(
        "Personalization model not found at %s (%s) — cold-start mode.",
        _MODEL_CACHE_PATH,
        _model_exc,
    )
    _scorer = PersonalizedScorer()

# Simple keyword → category resolver for the transaction endpoint
_MERCHANT_CATEGORY_HINTS: Dict[str, List[str]] = {
    # Check dining before travel so "uber eats" / "doordash" resolve correctly
    "dining": [
        # Delivery platforms
        "doordash",
        "door dash",
        "uber eats",
        "ubereats",
        "grubhub",
        "grub hub",
        "instacart",
        "seamless",
        "caviar",
        "postmates",
        "gopuff",
        # Fast food
        "mcdonald",
        "chick-fil-a",
        "chickfila",
        "wendy",
        "burger king",
        "taco bell",
        "popeyes",
        "five guys",
        "in-n-out",
        "shake shack",
        "sonic",
        "dairy queen",
        "kfc",
        "papa john",
        "domino",
        "little caesar",
        # Casual / chain
        "chipotle",
        "starbucks",
        "dunkin",
        "panera",
        "subway",
        "panda express",
        "olive garden",
        "applebee",
        "chili",
        "texas roadhouse",
        "outback",
        "cheesecake factory",
        "denny",
        "ihop",
        "waffle house",
        # Generic keywords
        "restaurant",
        "cafe",
        "coffee",
        "bakery",
        "diner",
        "eatery",
        "steakhouse",
        "grill",
        "pizzeria",
        "pizza",
        "sushi",
        "ramen",
        "taqueria",
        "bar & grill",
        "gastropub",
        "bistro",
        "brasserie",
        "cantina",
        "trattoria",
    ],
    "groceries": [
        "whole foods",
        "trader joe",
        "kroger",
        "safeway",
        "publix",
        "wegmans",
        "aldi",
        "sprouts",
        "meijer",
        "stop & shop",
        "stop and shop",
        "giant",
        "food lion",
        "heb",
        "grocery",
        "supermarket",
        "market basket",
        "winco",
        "piggly wiggly",
        "fresh market",
        "albertson",
        "vons",
        "ralphs",
        "tom thumb",
        "randall",
        "winn-dixie",
        "shoprite",
        # Wholesale clubs (grocery spend, not travel/general)
        "costco",
        "sam's club",
        "bj's wholesale",
    ],
    "travel": [
        # Airlines
        "delta",
        "united airlines",
        "american airlines",
        "southwest",
        "jetblue",
        "alaska airlines",
        "spirit airlines",
        "frontier airlines",
        "hawaiian airlines",
        # Hotels
        "marriott",
        "hilton",
        "hyatt",
        "ihg",
        "wyndham",
        "best western",
        "radisson",
        "holiday inn",
        "sheraton",
        "westin",
        "ritz-carlton",
        "four seasons",
        "kimpton",
        "motel",
        "hotel",
        "resort",
        # Booking platforms
        "airbnb",
        "vrbo",
        "expedia",
        "booking.com",
        "hotels.com",
        "priceline",
        "kayak",
        "orbitz",
        "hotwire",
        "travelocity",
        # Rideshare (not delivery — "uber eats" is caught by dining first)
        "uber",
        "lyft",
        # Car rental
        "enterprise",
        "hertz",
        "avis",
        "budget rent",
        "national car",
        "alamo",
        "dollar rent",
        # Transit
        "amtrak",
        "greyhound",
        "megabus",
    ],
    "gas": [
        "shell",
        "bp",
        "chevron",
        "exxon",
        "mobil",
        "texaco",
        "citgo",
        "marathon",
        "sunoco",
        "valero",
        "arco",
        "76 gas",
        "casey",
        "kwik trip",
        "wawa",
        "sheetz",
        "speedway",
        "circle k",
        "gas station",
        "fuel",
        "gasoline",
    ],
    # "streaming" matches the category key used in card reward_rates (e.g. Blue Cash Preferred 6x)
    "streaming": [
        "netflix",
        "hulu",
        "disney+",
        "disney plus",
        "hbo",
        "max",
        "peacock",
        "paramount+",
        "paramount plus",
        "apple tv",
        "amazon prime video",
        "prime video",
        "crunchyroll",
        "fubo",
        "sling",
        "youtube premium",
        "spotify",
        "apple music",
        "tidal",
        "pandora",
        "sirius",
        "deezer",
        "audible",
    ],
    "entertainment": [
        "amc",
        "regal",
        "cinemark",
        "imax",
        "cinema",
        "theater",
        "ticketmaster",
        "stubhub",
        "eventbrite",
        "livenation",
        "live nation",
        "seatgeek",
        "gamestop",
        "playstation",
        "xbox",
        "nintendo",
        "steam",
        "twitch",
        "bowling",
        "topgolf",
        "dave & buster",
        "escape room",
        "laser tag",
        "miniature golf",
        "arcade",
    ],
    "online_shopping": [
        "amazon",
        "ebay",
        "etsy",
        "shopify",
        "wayfair",
        "chewy",
        "zappos",
        "overstock",
        "newegg",
        "adorama",
        "b&h photo",
        "bestbuy.com",
        "best buy",
        "apple store",
        "apple.com",
        "walmart.com",
        "target.com",
    ],
    "drugstore": [
        "cvs",
        "walgreens",
        "rite aid",
        "duane reade",
        "pharmacy",
        "drug mart",
        "health mart",
    ],
}

# ---------------------------------------------------------------------------
# Gemini merchant classifier — used as fallback for merchants not in the
# keyword map above. Cached per process so the same merchant is never
# classified twice. Disabled gracefully if GCP creds are unavailable.
# ---------------------------------------------------------------------------
_VALID_CATEGORIES = frozenset(_MERCHANT_CATEGORY_HINTS.keys()) | {"other"}

_CATEGORY_CACHE: Dict[str, str] = {}

_GEMINI_SYSTEM_PROMPT = (
    "You are a credit card reward category classifier. "
    "Given a merchant name, reply with exactly one word — the spending category it belongs to. "
    "Valid categories: dining, groceries, travel, gas, streaming, entertainment, "
    "online_shopping, drugstore, other. "
    "Rules: delivery apps (DoorDash, Uber Eats, Grubhub) → dining. "
    "Rideshare without food delivery context → travel. "
    "Reply with only the category word, no punctuation, no explanation."
)


def _classify_merchant_gemini(merchant: str) -> Optional[str]:
    """Return a category from Gemini, or None if unavailable/disabled."""
    if os.getenv("ENABLE_LLM_CATEGORY", "true").lower() not in ("1", "true", "yes"):
        return None

    key = merchant.lower().strip()
    if key in _CATEGORY_CACHE:
        return _CATEGORY_CACHE[key]

    try:
        from src.model_pipeline.llm.vertex_gemini_client import VertexGeminiClient

        client = VertexGeminiClient(temperature=0.0, timeout_sec=3.0)
        raw = client.generate(
            system_message=_GEMINI_SYSTEM_PROMPT,
            user_message=merchant,
        )
        category = raw.strip().lower().split()[0]  # take first word only
        if category not in _VALID_CATEGORIES:
            category = "other"
        _CATEGORY_CACHE[key] = category
        logger.debug("Gemini classified %r → %s", merchant, category)
        return category
    except Exception as exc:
        logger.debug(
            "Gemini merchant classification unavailable for %r: %s", merchant, exc
        )
        return None


# ---------------------------------------------------------------------------
# Card catalog — loaded from shared module (pipeline + curated cards)
# ---------------------------------------------------------------------------
_CATALOG: List[CardCatalogItem] = DISPLAY_CATALOG
_CATALOG_BY_ID: Dict[str, CardCatalogItem] = DISPLAY_CATALOG_BY_ID


@router.get("/cards/catalog", response_model=List[CardCatalogItem])
def get_card_catalog() -> List[CardCatalogItem]:
    """Public endpoint — returns the full card catalog."""
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


def _build_portfolio(card_ids: List[str]) -> List[Dict[str, Any]]:
    """Build scorer-compatible portfolio dicts from card IDs."""
    portfolio = []
    for cid in card_ids:
        catalog_card = _CATALOG_BY_ID.get(cid)
        if catalog_card is None:
            logger.warning("Saved card_id %r not in catalog — skipped", cid)
            continue
        rates = get_scoring_rates(cid)
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
    """Return a category from hint → keyword map → Gemini → 'other'."""
    if hint:
        return hint.lower()
    lower = merchant.lower()
    for category, keywords in _MERCHANT_CATEGORY_HINTS.items():
        if any(kw in lower for kw in keywords):
            return category
    return _classify_merchant_gemini(merchant) or "other"


def _resolve_category_heuristic_first(merchant: str, hint: Optional[str]) -> str:
    """Keyword map first, then Gemini for unknowns, then hint, then 'other'."""
    lower = merchant.lower()
    for category, keywords in _MERCHANT_CATEGORY_HINTS.items():
        if any(kw in lower for kw in keywords):
            return category
    gemini_category = _classify_merchant_gemini(merchant)
    if gemini_category:
        return gemini_category
    if hint:
        return hint.lower()
    return "other"


def _build_card_explanation(
    card_name: str,
    rank: int,
    reward_rate: float,
    annual_fee: float,
    projected_savings: float,
    category: str,
    active_personas: List[str],
    category_boost: float,
    card_display: Optional[CardDisplayInfo],
    spending_categories: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Generate card-specific explanation, pros, cons, and best_for."""
    # --- Determine the card's top bonus category from display highlights ---
    highlights = card_display.reward_highlights if card_display else []
    has_category_bonuses = any(
        "x " in h.lower() or "%" in h
        for h in highlights
        if "everything" not in h.lower()
    )

    # --- Explanation ---
    rate_str = f"{reward_rate:g}x" if reward_rate >= 1 else f"{reward_rate:g}%"
    savings_str = (
        f"${projected_savings:,.0f}"
        if projected_savings >= 1
        else f"${projected_savings:.2f}"
    )

    if rank == 1:
        explanation = (
            f"Top pick for your spending profile. "
            f"{card_name} earns {rate_str} on {category}, "
            f"projecting {savings_str}/year in rewards."
        )
    else:
        explanation = (
            f"{card_name} earns {rate_str} on {category}. "
            f"Projected annual reward: {savings_str}."
        )

    if category_boost > 1.0 and active_personas:
        explanation += f" Boosted by your {', '.join(active_personas)} profile."

    # --- Pros (exactly 2) ---
    if has_category_bonuses:
        # Find the best highlight to feature
        top_highlight = highlights[0] if highlights else f"{rate_str} on {category}"
        pro1 = f"{top_highlight} — strong for your spending pattern"
    else:
        pro1 = f"Flat {rate_str} on all purchases — no category tracking needed"

    if annual_fee == 0:
        pro2 = "No annual fee keeps your net rewards positive from day one"
    elif projected_savings > annual_fee:
        pro2 = (
            f"Projected {savings_str}/year in rewards "
            f"more than offsets the ${annual_fee:,.0f} annual fee"
        )
    else:
        pro2 = (
            f"Projected annual reward of {savings_str} across your spending categories"
        )

    # --- Cons (exactly 2) ---
    if annual_fee > 0:
        con1 = (
            f"${annual_fee:,.0f}/year annual fee requires consistent spending to offset"
        )
    else:
        if has_category_bonuses:
            con1 = "Lower base rate on purchases outside bonus categories"
        else:
            con1 = "No bonus categories — specialized cards may earn more in your top areas"

    # Find a category the user spends in where this card has no bonus
    if spending_categories and has_category_bonuses:
        # Use the card's highlights to identify gaps
        highlight_text = " ".join(highlights).lower()
        missing_cats = [
            cat
            for cat, amt in sorted(spending_categories.items(), key=lambda x: -x[1])
            if cat not in highlight_text and amt > 0
        ]
        if missing_cats:
            label = "#2" if len(missing_cats) > 1 else ""
            con2 = (
                f"No bonus rate for {missing_cats[0]}"
                f" — your {label} spending category"
            )
        else:
            con2 = "Rewards value depends on how you redeem points"
    else:
        con2 = "Other cards may offer higher rates in specific spending categories"

    # --- Best for ---
    if active_personas:
        persona_label = ", ".join(active_personas)
        if has_category_bonuses and highlights:
            best_for = f"{persona_label.title()} spenders focused on {category}"
        else:
            best_for = (
                f"{persona_label.title()} spenders looking for simple flat-rate rewards"
            )
    elif has_category_bonuses:
        best_for = f"Spenders with high {category} purchases"
    else:
        best_for = "Everyday spenders who prefer simplicity over category optimization"

    return {
        "explanation": explanation,
        "pros": [pro1, pro2],
        "cons": [con1, con2],
        "best_for": best_for,
    }


def _run_recommendation(
    portfolio: List[Dict[str, Any]],
    transaction: Dict[str, Any],
    active_personas: List[str],
    is_generic: bool,
    monthly_spend: float = 0.0,
    spending_categories: Optional[Dict[str, float]] = None,
) -> PersonaRecommendResponse:
    """Score portfolio, apply persona modifier, return enriched response."""
    try:
        if spending_categories:
            # Multi-category scoring: aggregate reward across every category
            # weighted by spend amount so the ranking reflects the user's full
            # spending profile rather than just the single dominant category.
            card_totals: Dict[str, Dict[str, Any]] = {}
            for cat, amt in spending_categories.items():
                if amt <= 0:
                    continue
                txn = {"amount": amt, "category": cat, "merchant": f"{cat}-merchant"}
                for score in _scorer.scorer.score_portfolio(portfolio, txn):
                    cid = score["card_id"]
                    if cid not in card_totals:
                        card_totals[cid] = dict(score)
                        card_totals[cid]["reward_amount"] = 0.0
                    card_totals[cid]["reward_amount"] += score["reward_amount"]

            for entry in card_totals.values():
                entry["raw_reward_amount"] = entry["reward_amount"]

            ranked_raw_agg = _scorer.ranker.rank(list(card_totals.values()))
            result = {
                "ranked": ranked_raw_agg,
                "best_card_id": (
                    ranked_raw_agg[0]["card_id"] if ranked_raw_agg else None
                ),
                "point_value": _scorer.default_point_value,
                "is_personalized": False,
            }
        else:
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
        display = _card_display_for(cid)
        card_annual_fee = float(c.get("annual_fee", 0.0))

        expl = _build_card_explanation(
            card_name=c.get("card_name", ""),
            rank=int(c.get("rank", 0)),
            reward_rate=reward_rate,
            annual_fee=card_annual_fee,
            projected_savings=projected_savings,
            category=category,
            active_personas=active_personas,
            category_boost=float(adj.get("category_boost_applied", 1.0)),
            card_display=display,
            spending_categories=spending_categories,
        )

        scored_cards.append(
            ScoredCard(
                card_id=cid,
                card_name=c.get("card_name", ""),
                reward_amount=float(c.get("reward_amount", 0.0)),
                annual_fee=card_annual_fee,
                rank=int(c.get("rank", 0)),
                explanation=expl["explanation"],
                pros=expl["pros"],
                cons=expl["cons"],
                best_for=expl["best_for"],
                persona_adjustments=c.get("persona_adjustments"),
                score_breakdown=breakdown,
                persona_match_reason=reason,
                projected_savings=projected_savings,
                card_display=display,
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
    """Recommend using the user's saved wallet and active personas.

    When ``use_full_catalog`` is true (card-finder mode), the full catalog
    is scored and cards already in the wallet are excluded so that only
    *new* card suggestions are returned.
    """
    profile = service.get_profile(db, current_user)
    saved_ids = set(profile.saved_card_ids)

    if payload.use_full_catalog or len(saved_ids) == 0:
        # Card-finder mode: score full catalog, exclude wallet cards
        all_ids = [c.card_id for c in _CATALOG]
        card_ids = (
            [cid for cid in all_ids if cid not in saved_ids] if saved_ids else all_ids
        )
        is_generic = len(saved_ids) == 0
    else:
        # Wallet mode: score only saved cards
        card_ids = list(saved_ids)
        is_generic = False

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
        spending_categories=categories or None,
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
    card_ids = [c.card_id for c in _CATALOG]
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
    rates = get_scoring_rates(card_id).get("reward_rates", {})
    bonuses = rates.get("category_bonuses")
    return not bonuses


def _find_baseline(saved_card_ids: List[str]) -> Dict[str, Any]:
    """Return the first saved catch-all card, or the generic 1% baseline."""
    for cid in saved_card_ids:
        if _is_catch_all(cid) and cid in _CATALOG_BY_ID:
            cat = _CATALOG_BY_ID[cid]
            rates = get_scoring_rates(cid)
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
