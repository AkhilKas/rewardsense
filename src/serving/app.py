"""FastAPI scaffold and deterministic scoring API for RewardSense serving."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.model_pipeline.scoring.card_ranker import CardRanker
from src.model_pipeline.scoring.merchant_mapper import MerchantCategoryMapper
from src.model_pipeline.scoring.transaction_scorer import TransactionScorer
from src.serving.model_loader import get_model, get_model_version

logger = logging.getLogger(__name__)

KNOWN_SPENDING_CATEGORIES = {
    "groceries",
    "dining",
    "travel",
    "gas",
    "online_shopping",
    "entertainment",
    "utilities",
    "streaming",
    "drugstores",
    "general",
    "other",
}
DEFAULT_MONTHLY_SPEND = 1000.0
MAX_RECOMMENDATIONS = int(os.getenv("PREDICT_TOP_K", "10"))
DEFAULT_DETERMINISTIC_WEIGHT = 0.6
DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "current"
    / "offers"
    / "merged_cards.json"
)

CURATED_CARD_CATALOG: List[Dict[str, Any]] = [
    {
        "card_id": "amex_gold",
        "card_name": "Amex Gold Card",
        "annual_fee": 250.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 4.0, "groceries": 4.0},
        },
    },
    {
        "card_id": "chase_sapphire_preferred",
        "card_name": "Chase Sapphire Preferred",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"travel": 3.0, "dining": 3.0},
        },
    },
    {
        "card_id": "capital_one_venture_x",
        "card_name": "Capital One Venture X",
        "annual_fee": 395.0,
        "reward_rates": {
            "universal_base_rate": 2.0,
            "category_bonuses": {"travel": 5.0},
        },
    },
    {
        "card_id": "citi_double_cash",
        "card_name": "Citi Double Cash",
        "annual_fee": 0.0,
        "reward_rates": {"universal_base_rate": 2.0},
    },
    {
        "card_id": "blue_cash_preferred",
        "card_name": "Blue Cash Preferred",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"groceries": 6.0, "streaming": 6.0, "gas": 3.0},
        },
    },
    {
        "card_id": "capital_one_savor",
        "card_name": "Capital One Savor",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 3.0, "entertainment": 3.0, "groceries": 3.0},
        },
    },
    {
        "card_id": "chase_freedom_unlimited",
        "card_name": "Chase Freedom Unlimited",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 1.5,
            "category_bonuses": {"dining": 3.0, "travel": 2.0},
        },
    },
    {
        "card_id": "wells_fargo_autograph",
        "card_name": "Wells Fargo Autograph",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {
                "dining": 3.0,
                "travel": 3.0,
                "gas": 3.0,
                "streaming": 3.0,
            },
        },
    },
    {
        "card_id": "discover_it_cash_back",
        "card_name": "Discover it Cash Back",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"gas": 5.0, "online_shopping": 5.0},
        },
    },
]
MCC_MAPPER = MerchantCategoryMapper()


def _parse_cors_origins() -> List[str]:
    """Parse CORS origins from env (JSON list or comma-separated string)."""
    raw = os.getenv("CORS_ORIGINS")
    if not raw:
        return ["http://localhost:5173", "http://localhost:3000"]

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            origins = [str(item).strip() for item in parsed if str(item).strip()]
            if origins:
                return origins
    except json.JSONDecodeError:
        pass

    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost:5173", "http://localhost:3000"]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown_card"


def _parse_rate(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        match = re.search(r"(\d+(?:\.\d+)?)", str(raw))
        if not match:
            return None
        value = float(match.group(1))

    if value < 0 or value > 10:
        return None
    return value


def _load_catalog_from_offers(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read card catalog from %s: %s", path, exc)
        return []

    if not isinstance(data, list):
        return []

    cards_by_id: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue

        raw_name = str(item.get("card_name") or item.get("name") or "").strip()
        if not raw_name or raw_name.lower().startswith("best for:"):
            continue

        rate = _parse_rate(item.get("base_reward_rate"))
        if rate is None:
            continue

        card_id = str(item.get("card_id") or _slugify(raw_name))
        annual_fee = item.get("annual_fee", 0.0)
        try:
            annual_fee = float(annual_fee)
        except (TypeError, ValueError):
            annual_fee = 0.0

        candidate = {
            "card_id": card_id,
            "card_name": raw_name,
            "annual_fee": max(annual_fee, 0.0),
            "reward_rates": {"universal_base_rate": rate},
        }

        existing = cards_by_id.get(card_id)
        if existing is None:
            cards_by_id[card_id] = candidate
            continue

        existing_rate = float(
            existing.get("reward_rates", {}).get("universal_base_rate", 0.0)
        )
        if rate > existing_rate:
            cards_by_id[card_id] = candidate

    return list(cards_by_id.values())


def _load_card_catalog() -> List[Dict[str, Any]]:
    path = Path(os.getenv("CARD_CATALOG_PATH", str(DEFAULT_CATALOG_PATH)))
    loaded_cards = _load_catalog_from_offers(path)

    cards_by_id: Dict[str, Dict[str, Any]] = {
        card["card_id"]: dict(card) for card in loaded_cards
    }
    for card in CURATED_CARD_CATALOG:
        cards_by_id[card["card_id"]] = dict(card)

    catalog = list(cards_by_id.values())
    if not catalog:
        return list(CURATED_CARD_CATALOG)
    return catalog


CARD_CATALOG = _load_card_catalog()


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class TransactionRecord(StrictModel):
    merchant: Optional[str] = None
    merchant_category: Optional[str] = None
    category: Optional[str] = None
    mcc_code: Optional[int] = None
    amount: float = Field(..., gt=0)
    date: str = Field(..., min_length=1)


class PredictionRequest(StrictModel):
    user_id: str = Field(..., min_length=1)
    spending_categories: Dict[str, float] = Field(default_factory=dict)
    monthly_spend: Optional[float] = Field(default=None, ge=0)
    preferred_rewards: List[str] = Field(default_factory=list)
    transaction_history: List[TransactionRecord] = Field(default_factory=list)


class RecommendedCard(StrictModel):
    card_name: str = Field(..., min_length=1)
    score: float
    rank: int = Field(..., ge=1)
    explanation: str = Field(..., min_length=1)
    deterministic_score: float
    personalization_score: float


class PredictionResponse(StrictModel):
    recommended_cards: List[RecommendedCard]
    model_version: str
    inference_latency_ms: float


class HealthResponse(StrictModel):
    status: str
    model_version: str
    uptime_seconds: float


app = FastAPI(title="RewardSense Inference API", version="0.2.0")
app.state.started_at = time.monotonic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-request-id"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log request_id, latency, and status code for every request."""
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "request_id=%s method=%s path=%s status_code=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            latency_ms,
        )


def _parse_deterministic_weight() -> float:
    raw_weight = os.getenv("PERSONALIZATION_DETERMINISTIC_WEIGHT")
    if raw_weight is None:
        return DEFAULT_DETERMINISTIC_WEIGHT
    try:
        value = float(raw_weight)
    except ValueError:
        return DEFAULT_DETERMINISTIC_WEIGHT
    return min(1.0, max(0.0, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_user_features_frame(
    payload: PredictionRequest,
    normalized_categories: Dict[str, float],
    transactions: List[Dict[str, Any]],
) -> pd.DataFrame:
    amounts = [max(_safe_float(txn.get("amount")), 0.0) for txn in transactions]
    total_spending = sum(amounts)
    monthly_budget = (
        payload.monthly_spend
        if payload.monthly_spend is not None and payload.monthly_spend > 0
        else (total_spending if total_spending > 0 else DEFAULT_MONTHLY_SPEND)
    )

    txn_count = max(len(amounts), 1)
    avg_amount = total_spending / txn_count
    median_amount = sorted(amounts)[len(amounts) // 2] if amounts else 0.0
    if amounts:
        mean = avg_amount
        variance = sum((value - mean) ** 2 for value in amounts) / txn_count
        std_amount = math.sqrt(variance)
    else:
        std_amount = 0.0

    merchant_counts: Dict[str, int] = {}
    for txn in transactions:
        merchant = str(txn.get("merchant", "")).strip().lower()
        if not merchant:
            continue
        merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1

    repeat_merchants = sum(1 for count in merchant_counts.values() if count > 1)
    repeat_ratio = repeat_merchants / max(len(merchant_counts), 1)

    category_shares = [
        amount / total_spending
        for amount in normalized_categories.values()
        if total_spending
    ]
    entropy = 0.0
    for share in category_shares:
        entropy -= share * math.log(share, 2)
    max_entropy = math.log(max(len(normalized_categories), 2), 2)
    spending_diversity = entropy if entropy > 0 else 0.0
    affinity_score = (entropy / max_entropy) if max_entropy > 0 else 0.0

    features: Dict[str, float] = {
        "monthly_budget": monthly_budget,
        "annual_budget": monthly_budget * 12.0,
        "num_cards": float(len(CARD_CATALOG)),
        "monthly_budget_log": math.log1p(monthly_budget),
        "age_group_ordinal": 3.0,
        "total_spending": total_spending,
        "total_transactions": float(len(payload.transaction_history)),
        "avg_transaction_amount": avg_amount,
        "median_transaction_amount": median_amount,
        "transaction_amount_std": std_amount,
        "spending_diversity": spending_diversity,
        "weekend_spending_ratio": 0.0,
        "card_switch_rate": 0.0,
        "num_cards_used": float(min(max(len(payload.preferred_rewards), 1), 5)),
        "num_unique_mccs": float(
            len(
                {
                    txn.mcc_code
                    for txn in payload.transaction_history
                    if txn.mcc_code is not None
                }
            )
        ),
        "num_unique_merchants": float(len(merchant_counts)),
        "repeat_merchant_ratio": repeat_ratio,
        "peak_spending_day": 0.0,
        "peak_spending_month": 0.0,
        "spending_velocity": (std_amount / avg_amount) if avg_amount > 0 else 0.0,
        "category_affinity_score": affinity_score,
    }

    for reward in payload.preferred_rewards:
        key = f"redemption_{_slugify(str(reward))}"
        features[key] = 1.0

    return pd.DataFrame([features])


def _align_features_for_model(model: Any, user_features: pd.DataFrame) -> pd.DataFrame:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return user_features

    aligned: Dict[str, float] = {}
    source_row = user_features.iloc[0].to_dict()
    for name in feature_names:
        aligned[str(name)] = _safe_float(source_row.get(str(name), 0.0), default=0.0)
    return pd.DataFrame([aligned])


def _predict_point_value(
    user_features: pd.DataFrame,
) -> Tuple[float, bool, float, Optional[str]]:
    default_point_value = 0.01
    try:
        personalization_scorer = get_model()
    except Exception as exc:
        return default_point_value, False, default_point_value, str(exc)

    default_point_value = _safe_float(
        getattr(personalization_scorer, "default_point_value", 0.01), 0.01
    )

    # Use scorer's own fallback logic when available.
    if hasattr(personalization_scorer, "_get_point_value"):
        try:
            value, is_personalized = personalization_scorer._get_point_value(  # type: ignore[attr-defined]
                user_features
            )
            point_value = _safe_float(value, default_point_value)
            if point_value <= 0:
                return default_point_value, False, default_point_value, None
            return point_value, bool(is_personalized), default_point_value, None
        except Exception as exc:
            return default_point_value, False, default_point_value, str(exc)

    model = getattr(personalization_scorer, "model", None)
    if model is None or not hasattr(model, "predict"):
        return default_point_value, False, default_point_value, "model_unavailable"

    try:
        aligned = _align_features_for_model(model, user_features)
        prediction = model.predict(aligned)
        point_value = _safe_float(prediction[0], default_point_value)
        if math.isnan(point_value) or point_value <= 0:
            return default_point_value, False, default_point_value, None
        return point_value, True, default_point_value, None
    except Exception as exc:
        return default_point_value, False, default_point_value, str(exc)


def _card_affinity_multiplier(
    card: Dict[str, Any],
    normalized_categories: Dict[str, float],
    preferred_rewards: List[str],
) -> float:
    total_spend = sum(normalized_categories.values())
    if total_spend <= 0:
        return 1.0

    reward_rates = card.get("reward_rates", {})
    base_rate = _safe_float(reward_rates.get("universal_base_rate"), 1.0)
    category_bonuses = reward_rates.get("category_bonuses", {})
    if not isinstance(category_bonuses, dict):
        category_bonuses = {}

    multiplier = 1.0
    for category, spend in normalized_categories.items():
        spend_share = spend / total_spend
        category_rate = _safe_float(category_bonuses.get(category), base_rate)
        uplift = max(category_rate - base_rate, 0.0)
        multiplier += spend_share * uplift * 0.75

    preferred = {str(value).strip().lower() for value in preferred_rewards}
    if "travel" in preferred or "travel_points" in preferred:
        travel_bonus = _safe_float(category_bonuses.get("travel"), base_rate)
        if travel_bonus > base_rate:
            multiplier += 0.15
    if "cashback" in preferred:
        if _safe_float(card.get("annual_fee"), 0.0) <= 0:
            multiplier += 0.1

    return max(multiplier, 0.1)


def _normalize_spending_categories(
    raw_categories: Dict[str, float],
) -> Tuple[Dict[str, float], List[str]]:
    normalized: Dict[str, float] = {}
    unknown_categories: List[str] = []

    for raw_name, raw_amount in raw_categories.items():
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            continue

        if amount <= 0:
            continue

        category = str(raw_name).strip().lower()
        if not category:
            continue
        if category not in KNOWN_SPENDING_CATEGORIES:
            unknown_categories.append(category)
            category = "other"

        normalized[category] = normalized.get(category, 0.0) + amount

    return normalized, unknown_categories


def _normalize_transaction_category(txn: TransactionRecord) -> str:
    category = (
        (txn.category or "").strip().lower()
        or (txn.merchant_category or "").strip().lower()
        or ""
    )
    if category in KNOWN_SPENDING_CATEGORIES:
        return category

    if txn.mcc_code is not None:
        mapped = MCC_MAPPER.map_mcc_to_category(txn.mcc_code)
        if mapped in KNOWN_SPENDING_CATEGORIES:
            return mapped

    return "other"


def _build_transactions(
    payload: PredictionRequest,
    normalized_categories: Dict[str, float],
) -> List[Dict[str, Any]]:
    transactions: List[Dict[str, Any]] = []

    for category, amount in normalized_categories.items():
        transactions.append(
            {
                "amount": amount,
                "category": category,
                "merchant": "monthly_profile",
            }
        )

    for txn in payload.transaction_history:
        if txn.amount <= 0:
            continue
        transactions.append(
            {
                "amount": float(txn.amount),
                "category": _normalize_transaction_category(txn),
                "merchant": txn.merchant or "transaction_history",
                "date": txn.date,
            }
        )

    explicit_spend = sum(normalized_categories.values())
    monthly_spend = payload.monthly_spend if payload.monthly_spend is not None else 0.0
    if monthly_spend > explicit_spend:
        transactions.append(
            {
                "amount": monthly_spend - explicit_spend,
                "category": "other",
                "merchant": "monthly_spend_remainder",
            }
        )

    if not transactions:
        fallback_amount = monthly_spend if monthly_spend > 0 else DEFAULT_MONTHLY_SPEND
        transactions.append(
            {
                "amount": fallback_amount,
                "category": "other",
                "merchant": "default_spend",
            }
        )

    return transactions


def _anonymize_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def _build_explanation(
    card: Dict[str, Any],
    spending_categories: Dict[str, float],
) -> str:
    reward_rates = card.get("reward_rates", {})
    category_bonuses = reward_rates.get("category_bonuses", {})

    if isinstance(category_bonuses, dict):
        for category, _amount in sorted(
            spending_categories.items(), key=lambda item: item[1], reverse=True
        ):
            if category in category_bonuses:
                return (
                    f"Strong match for {category} spend at "
                    f"{float(category_bonuses[category]):.1f}% deterministic reward rate."
                )

    base_rate = float(reward_rates.get("universal_base_rate", 1.0))
    return f"Consistent base rewards card with {base_rate:.1f}% return."


def _score_profile(
    payload: PredictionRequest,
) -> Tuple[
    List[RecommendedCard],
    Dict[str, float],
    float,
    Dict[str, Any],
]:
    total_start = time.perf_counter()

    normalize_start = time.perf_counter()
    normalized_categories, unknown_categories = _normalize_spending_categories(
        payload.spending_categories
    )
    transactions = _build_transactions(payload, normalized_categories)
    normalize_ms = (time.perf_counter() - normalize_start) * 1000.0

    deterministic_start = time.perf_counter()
    deterministic_scorer = TransactionScorer()
    aggregate: Dict[str, Dict[str, Any]] = {
        card["card_id"]: {
            "card_id": card["card_id"],
            "card_name": card.get("card_name", ""),
            "annual_fee": float(card.get("annual_fee", 0.0)),
            "deterministic_score": 0.0,
            "reward_rates": card.get("reward_rates", {}),
        }
        for card in CARD_CATALOG
    }

    for transaction in transactions:
        scored_cards = deterministic_scorer.score_portfolio(CARD_CATALOG, transaction)
        for scored in scored_cards:
            card_id = scored["card_id"]
            if card_id not in aggregate:
                continue
            aggregate[card_id]["deterministic_score"] += float(scored["reward_amount"])
    deterministic_ms = (time.perf_counter() - deterministic_start) * 1000.0

    personalization_start = time.perf_counter()
    user_features = _build_user_features_frame(
        payload, normalized_categories, transactions
    )
    point_value, is_personalized, default_point_value, personalization_error = (
        _predict_point_value(user_features)
    )
    point_value_factor = (
        point_value / default_point_value if default_point_value > 0 else 1.0
    )
    deterministic_weight = _parse_deterministic_weight()

    for card in aggregate.values():
        deterministic_score = float(card["deterministic_score"])
        if is_personalized:
            affinity = _card_affinity_multiplier(
                card=card,
                normalized_categories=normalized_categories,
                preferred_rewards=payload.preferred_rewards,
            )
            personalization_score = deterministic_score * point_value_factor * affinity
            blended_score = (
                deterministic_weight * deterministic_score
                + (1.0 - deterministic_weight) * personalization_score
            )
        else:
            personalization_score = deterministic_score
            blended_score = deterministic_score

        card["personalization_score"] = personalization_score
        card["blended_score"] = blended_score
        # CardRanker sorts on reward_amount and annual_fee.
        card["reward_amount"] = blended_score
    personalization_ms = (time.perf_counter() - personalization_start) * 1000.0

    ranking_start = time.perf_counter()
    ranked_cards = CardRanker().rank(list(aggregate.values()))
    ranking_ms = (time.perf_counter() - ranking_start) * 1000.0

    recommendations = [
        RecommendedCard(
            card_name=card.get("card_name", ""),
            score=round(float(card.get("blended_score", 0.0)), 4),
            rank=int(card.get("rank", index + 1)),
            explanation=_build_explanation(card, normalized_categories),
            deterministic_score=round(float(card.get("deterministic_score", 0.0)), 4),
            personalization_score=round(
                float(card.get("personalization_score", 0.0)), 4
            ),
        )
        for index, card in enumerate(ranked_cards[: max(1, MAX_RECOMMENDATIONS)])
    ]

    total_ms = (time.perf_counter() - total_start) * 1000.0
    stage_latency_ms = {
        "normalize": round(normalize_ms, 3),
        "deterministic": round(deterministic_ms, 3),
        "personalization": round(personalization_ms, 3),
        "rank": round(ranking_ms, 3),
        "total": round(total_ms, 3),
    }
    telemetry = {
        "unknown_categories": unknown_categories,
        "is_personalized": is_personalized,
        "point_value": round(point_value, 6),
        "default_point_value": round(default_point_value, 6),
        "point_value_factor": round(point_value_factor, 6),
        "deterministic_weight": round(deterministic_weight, 3),
        "personalization_error": personalization_error,
        "scores": [
            {
                "card_name": card.get("card_name", ""),
                "rank": int(card.get("rank", 0)),
                "deterministic_score": round(
                    float(card.get("deterministic_score", 0.0)), 4
                ),
                "personalization_score": round(
                    float(card.get("personalization_score", 0.0)), 4
                ),
                "blended_score": round(float(card.get("blended_score", 0.0)), 4),
            }
            for card in ranked_cards[: max(1, MAX_RECOMMENDATIONS)]
        ],
    }
    return recommendations, stage_latency_ms, total_ms, telemetry


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    uptime = max(time.monotonic() - app.state.started_at, 0.0)
    return HealthResponse(
        status="healthy",
        model_version=get_model_version() or "unloaded",
        uptime_seconds=round(uptime, 3),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Run deterministic scoring across all available cards for one user profile."""
    recommendations, stage_latency_ms, total_ms, telemetry = _score_profile(payload)
    request_id = getattr(request.state, "request_id", "unknown")
    user_hash = _anonymize_user_id(payload.user_id)

    logger.info(
        "predict_scoring request_id=%s user_hash=%s categories=%s unknown_categories=%s "
        "monthly_spend=%s preferred_rewards=%s transaction_history_count=%s "
        "is_personalized=%s point_value=%s deterministic_weight=%s "
        "stage_latency_ms=%s score_components=%s personalization_error=%s",
        request_id,
        user_hash,
        payload.spending_categories,
        telemetry["unknown_categories"],
        payload.monthly_spend,
        payload.preferred_rewards,
        len(payload.transaction_history),
        telemetry["is_personalized"],
        telemetry["point_value"],
        telemetry["deterministic_weight"],
        stage_latency_ms,
        telemetry["scores"],
        telemetry["personalization_error"],
    )

    return PredictionResponse(
        recommended_cards=recommendations,
        model_version=get_model_version() or "unloaded",
        inference_latency_ms=round(total_ms, 3),
    )
