"""FastAPI scaffold and deterministic scoring API for RewardSense serving."""

from __future__ import annotations

import asyncio
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

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.model_pipeline.scoring.card_ranker import CardRanker
from src.model_pipeline.scoring.merchant_mapper import MerchantCategoryMapper
from src.model_pipeline.scoring.transaction_scorer import TransactionScorer
from src.serving.inference_logger import build_log_record, log_inference
from src.serving.model_loader import get_model, get_model_version

# ---------------------------------------------------------------------------
# Lazy LLM imports — only needed when ENABLE_LLM_EXPLANATIONS is set
# ---------------------------------------------------------------------------
try:
    from src.model_pipeline.llm.explanation_generator import ExplanationGenerator
    from src.model_pipeline.llm.prompt_builder import ExplanationType
    from src.model_pipeline.llm.vertex_gemini_client import VertexGeminiClient

    LLM_MODULES_AVAILABLE = True
except ImportError:
    LLM_MODULES_AVAILABLE = False

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

# LLM explanation configuration (Story 2.4)
ENABLE_LLM_EXPLANATIONS: bool = os.getenv(
    "ENABLE_LLM_EXPLANATIONS", "false"
).lower() in (
    "1",
    "true",
    "yes",
)
LLM_EXPLANATION_TIMEOUT_SEC: float = float(
    os.getenv("LLM_EXPLANATION_TIMEOUT_SEC", "5.0")
)
LLM_TOP_N_EXPLANATIONS: int = int(os.getenv("LLM_TOP_N_EXPLANATIONS", "3"))

# Monitoring endpoint configuration
GCS_MONITORING_BUCKET: str = os.getenv("MONITORING_BUCKET", "rewardsense-monitoring")
DRIFT_REPORT_PREFIX: str = os.getenv("DRIFT_REPORT_PREFIX", "drift-reports")
PERFORMANCE_SNAPSHOT_PREFIX: str = os.getenv("PERFORMANCE_PREFIX", "performance-snapshots")
LOCAL_DRIFT_DIR: Path = Path(
    os.getenv("LOCAL_DRIFT_DIR", "data/monitoring/drift-reports")
)
LOCAL_PERFORMANCE_DIR: Path = Path(
    os.getenv("LOCAL_SNAPSHOT_DIR", "data/monitoring/performance")
)

# Module-level LLM singleton (initialised lazily)
_explanation_generator: Optional[ExplanationGenerator] = None  # type: ignore[type-arg]

CURATED_CARD_CATALOG: List[Dict[str, Any]] = [
    {
        "card_id": "amex_gold",
        "card_name": "Amex Gold Card",
        "issuer": "American Express",
        "annual_fee": 250.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 4.0, "groceries": 4.0},
        },
        "key_benefits": [
            "4x on dining",
            "4x on groceries",
            "$120 dining credit",
            "$120 Uber credit",
        ],
    },
    {
        "card_id": "chase_sapphire_preferred",
        "card_name": "Chase Sapphire Preferred",
        "issuer": "Chase",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"travel": 3.0, "dining": 3.0},
        },
        "key_benefits": [
            "3x on dining",
            "2x on travel",
            "$50 hotel credit",
            "Trip cancellation insurance",
        ],
    },
    {
        "card_id": "capital_one_venture_x",
        "card_name": "Capital One Venture X",
        "issuer": "Capital One",
        "annual_fee": 395.0,
        "reward_rates": {
            "universal_base_rate": 2.0,
            "category_bonuses": {"travel": 5.0},
        },
        "key_benefits": [
            "2x on everything",
            "5x on travel",
            "$300 travel credit",
            "Airport lounge access",
        ],
    },
    {
        "card_id": "citi_double_cash",
        "card_name": "Citi Double Cash",
        "issuer": "Citi",
        "annual_fee": 0.0,
        "reward_rates": {"universal_base_rate": 2.0},
        "key_benefits": [
            "2% on everything",
            "No annual fee",
            "0% intro APR",
            "Citi Entertainment access",
        ],
    },
    {
        "card_id": "blue_cash_preferred",
        "card_name": "Blue Cash Preferred",
        "issuer": "American Express",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"groceries": 6.0, "streaming": 6.0, "gas": 3.0},
        },
        "key_benefits": [
            "6% on groceries",
            "6% on streaming",
            "3% on gas",
            "$0 intro annual fee first year",
        ],
    },
    {
        "card_id": "capital_one_savor",
        "card_name": "Capital One Savor",
        "issuer": "Capital One",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 3.0, "entertainment": 3.0, "groceries": 3.0},
        },
        "key_benefits": [
            "3% on dining",
            "3% on entertainment",
            "3% on groceries",
            "No annual fee",
        ],
    },
    {
        "card_id": "chase_freedom_unlimited",
        "card_name": "Chase Freedom Unlimited",
        "issuer": "Chase",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 1.5,
            "category_bonuses": {"dining": 3.0, "travel": 2.0},
        },
        "key_benefits": [
            "1.5% on everything",
            "3% on dining",
            "No annual fee",
            "0% intro APR 15 months",
        ],
    },
    {
        "card_id": "wells_fargo_autograph",
        "card_name": "Wells Fargo Autograph",
        "issuer": "Wells Fargo",
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
        "key_benefits": [
            "3x on dining",
            "3x on travel",
            "3x on gas and streaming",
            "No annual fee",
        ],
    },
    {
        "card_id": "discover_it_cash_back",
        "card_name": "Discover it Cash Back",
        "issuer": "Discover",
        "annual_fee": 0.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"gas": 5.0, "online_shopping": 5.0},
        },
        "key_benefits": [
            "5% rotating categories",
            "1% on everything else",
            "Cashback Match first year",
            "No annual fee",
        ],
    },
]
MCC_MAPPER = MerchantCategoryMapper()


def _parse_cors_origins() -> List[str]:
    """Parse CORS origins from env (JSON list or comma-separated string)."""
    raw = os.getenv("CORS_ORIGINS")
    if not raw:
        return [
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:3000",
        ]

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
    issuer: str = Field(default="")
    score: float
    rank: int = Field(..., ge=1)
    explanation: str = Field(..., min_length=1)
    deterministic_score: float
    personalization_score: float
    annual_fee: float = Field(default=0.0)
    reward_rate: float = Field(default=0.0)
    key_benefits: List[str] = Field(default_factory=list)


class PredictionResponse(StrictModel):
    recommended_cards: List[RecommendedCard]
    model_version: str
    inference_latency_ms: float


class HealthResponse(StrictModel):
    status: str
    model_version: str
    uptime_seconds: float


class MonitoringDriftCheck(StrictModel):
    detected: bool = False
    timestamp: str = ""
    feature_drift: Dict[str, float] = Field(default_factory=dict)


class MonitoringServingMetrics(StrictModel):
    request_count: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    p95_latency_ms: float = 0.0


class MonitoringRetrainEvent(StrictModel):
    timestamp: str
    trigger_reason: str
    model_version: str
    status: str


class MonitoringResponse(StrictModel):
    model_version: str
    last_deployment_time: str
    drift_check: MonitoringDriftCheck = Field(default_factory=MonitoringDriftCheck)
    serving_metrics: MonitoringServingMetrics = Field(
        default_factory=MonitoringServingMetrics
    )
    retrain_history: List[MonitoringRetrainEvent] = Field(default_factory=list)


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


def _build_template_explanation(
    card: Dict[str, Any],
    spending_categories: Dict[str, float],
) -> str:
    """Deterministic template-based explanation fallback."""
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


def _get_explanation_generator() -> Optional[Any]:
    """Lazily initialise the LLM ExplanationGenerator singleton."""
    global _explanation_generator
    if _explanation_generator is not None:
        return _explanation_generator
    if not LLM_MODULES_AVAILABLE or not ENABLE_LLM_EXPLANATIONS:
        return None
    try:
        client = VertexGeminiClient(
            timeout_sec=LLM_EXPLANATION_TIMEOUT_SEC,
        )
        _explanation_generator = ExplanationGenerator(
            llm_client=client,
            enforce_quality=True,
        )
        logger.info("LLM ExplanationGenerator initialised.")
        return _explanation_generator
    except Exception as exc:
        logger.warning("Could not initialise LLM ExplanationGenerator: %s", exc)
        return None


async def _generate_single_llm_explanation(
    generator: Any,
    card: Dict[str, Any],
    spending_categories: Dict[str, float],
    user_profile: Dict[str, Any],
) -> Tuple[str, float]:
    """Generate one LLM explanation in a thread-pool executor; returns (text, latency_ms)."""
    start = time.perf_counter()

    scoring_output = {
        "best_card": {
            "card_name": card.get("card_name", ""),
            "card_id": card.get("card_id", ""),
            "reward_rate": card.get("reward_rates", {}).get("universal_base_rate", 1.0),
            "deterministic_score": card.get("deterministic_score", 0.0),
            "personalization_score": card.get("personalization_score", 0.0),
            "blended_score": card.get("blended_score", 0.0),
        },
        "transaction": {
            "merchant": "user_profile",
            "category": (
                max(spending_categories, key=spending_categories.get)
                if spending_categories
                else "general"
            ),
        },
        "candidate_card": {
            "card_name": card.get("card_name", ""),
        },
    }
    personalization_signals = {
        "spending_categories": spending_categories,
        "preferred_rewards": user_profile.get("preferred_rewards", []),
    }

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: generator.generate(
                    explanation_type=ExplanationType.NEW_CARD_RECOMMENDATION,
                    scoring_output=scoring_output,
                    personalization_signals=personalization_signals,
                ),
            ),
            timeout=LLM_EXPLANATION_TIMEOUT_SEC,
        )
        explanation_text = result.summary
        if result.rationale:
            explanation_text += " " + " ".join(result.rationale)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return explanation_text, latency_ms
    except (asyncio.TimeoutError, Exception) as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.warning(
            "LLM explanation failed for %s (%.1fms): %s",
            card.get("card_name", "unknown"),
            latency_ms,
            exc,
        )
        # Return template fallback
        fallback = _build_template_explanation(card, spending_categories)
        return fallback, latency_ms


async def _generate_llm_explanations(
    cards: List[Dict[str, Any]],
    spending_categories: Dict[str, float],
    user_profile: Dict[str, Any],
    top_n: int = 3,
) -> Tuple[Dict[str, str], float]:
    """Generate LLM explanations for the top-N cards concurrently.

    Returns a dict mapping card_name -> explanation text, and total LLM latency.
    Falls back to template explanations on failure.
    """
    generator = _get_explanation_generator()
    if generator is None:
        return {}, 0.0

    top_cards = cards[:top_n]
    tasks = [
        _generate_single_llm_explanation(
            generator, card, spending_categories, user_profile
        )
        for card in top_cards
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    explanations: Dict[str, str] = {}
    total_latency = 0.0
    for card, result in zip(top_cards, results):
        card_name = card.get("card_name", "")
        if isinstance(result, Exception):
            logger.warning("LLM explanation exception for %s: %s", card_name, result)
            explanations[card_name] = _build_template_explanation(
                card, spending_categories
            )
        else:
            text, latency = result
            explanations[card_name] = text
            total_latency = max(total_latency, latency)

    return explanations, total_latency


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
            "issuer": card.get("issuer", ""),
            "annual_fee": float(card.get("annual_fee", 0.0)),
            "deterministic_score": 0.0,
            "reward_rates": card.get("reward_rates", {}),
            "key_benefits": card.get("key_benefits", []),
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
            issuer=card.get("issuer", ""),
            score=round(float(card.get("blended_score", 0.0)), 4),
            rank=int(card.get("rank", index + 1)),
            explanation=_build_template_explanation(card, normalized_categories),
            deterministic_score=round(float(card.get("deterministic_score", 0.0)), 4),
            personalization_score=round(
                float(card.get("personalization_score", 0.0)), 4
            ),
            annual_fee=round(float(card.get("annual_fee", 0.0)), 2),
            reward_rate=round(
                float(
                    card.get("reward_rates", {}).get("universal_base_rate", 0.0)
                ),
                2,
            ),
            key_benefits=card.get("key_benefits", []),
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


# ---------------------------------------------------------------------------
# Monitoring endpoint helpers
# ---------------------------------------------------------------------------


def _find_latest_json(directory: Path, prefix: str) -> Optional[Dict[str, Any]]:
    """Load the most recent JSON file matching ``prefix*.json`` from *directory*."""
    if not directory.exists():
        return None
    files = sorted(directory.glob(f"{prefix}*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read %s: %s", files[0], exc)
        return None


def _find_latest_json_gcs(prefix: str) -> Optional[Dict[str, Any]]:
    """Load the most recent JSON blob from GCS under *prefix*."""
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        return None

    try:
        client = gcs_storage.Client()
        bucket = client.bucket(GCS_MONITORING_BUCKET)
        blobs = list(bucket.list_blobs(prefix=prefix))
        json_blobs = [b for b in blobs if b.name.endswith(".json")]
        if not json_blobs:
            return None
        latest = sorted(json_blobs, key=lambda b: b.name, reverse=True)[0]
        content = latest.download_as_text(encoding="utf-8")
        return json.loads(content)
    except Exception as exc:
        logger.warning("GCS monitoring read failed (prefix=%s): %s", prefix, exc)
        return None


def _load_monitoring_data() -> MonitoringResponse:
    """Build monitoring response from latest drift report and performance snapshot."""
    from datetime import datetime, timedelta, timezone

    model_version = get_model_version() or "unloaded"
    uptime = max(time.monotonic() - app.state.started_at, 0.0)
    deploy_time = datetime.now(timezone.utc) - timedelta(seconds=uptime)

    # --- Drift data ---
    drift_data = (
        _find_latest_json(LOCAL_DRIFT_DIR, "data_drift_")
        or _find_latest_json_gcs(f"{DRIFT_REPORT_PREFIX}/")
    )

    drift_check = MonitoringDriftCheck()
    if drift_data:
        summary = drift_data.get("summary", drift_data)
        drift_check.detected = bool(summary.get("drift_detected", False))
        drift_check.timestamp = str(summary.get("timestamp", ""))

        per_feature = drift_data.get("per_feature_drift", {})
        drift_check.feature_drift = {
            feature: float(data.get("drift_score", 0.0))
            if isinstance(data, dict)
            else float(data)
            for feature, data in per_feature.items()
        }

    # --- Performance data ---
    perf_data = (
        _find_latest_json(LOCAL_PERFORMANCE_DIR, "performance_")
        or _find_latest_json_gcs(f"{PERFORMANCE_SNAPSHOT_PREFIX}/")
    )

    serving_metrics = MonitoringServingMetrics()
    if perf_data:
        serving_metrics.request_count = int(perf_data.get("total_requests", 0))
        latency = perf_data.get("latency", {})
        serving_metrics.avg_latency_ms = float(latency.get("mean_ms", 0.0))
        serving_metrics.p95_latency_ms = float(latency.get("p95_ms", 0.0))

    return MonitoringResponse(
        model_version=model_version,
        last_deployment_time=deploy_time.isoformat(),
        drift_check=drift_check,
        serving_metrics=serving_metrics,
        retrain_history=[],
    )


@app.get("/monitoring", response_model=MonitoringResponse)
def monitoring() -> MonitoringResponse:
    return _load_monitoring_data()


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """Run scoring, LLM explanations, and inference logging for one user profile."""
    recommendations, stage_latency_ms, total_ms, telemetry = _score_profile(payload)
    request_id = getattr(request.state, "request_id", "unknown")
    user_hash = _anonymize_user_id(payload.user_id)

    # --- Story 2.4: LLM Explanation Integration ---
    explanation_latency_ms: float = 0.0
    if ENABLE_LLM_EXPLANATIONS and LLM_MODULES_AVAILABLE:
        # Re-compute normalized categories for explanation context
        normalized_categories, _ = _normalize_spending_categories(
            payload.spending_categories
        )
        user_profile = {
            "preferred_rewards": payload.preferred_rewards,
            "monthly_spend": payload.monthly_spend,
        }
        # Get the ranked card aggregates for explanation context
        ranked_card_dicts = telemetry.get("scores", [])
        # Build card dicts with full info from telemetry scores
        explanation_cards = []
        for score_entry in ranked_card_dicts[:LLM_TOP_N_EXPLANATIONS]:
            card_dict = dict(score_entry)
            # Find full card info from catalog
            for catalog_card in CARD_CATALOG:
                if catalog_card.get("card_name") == score_entry.get("card_name"):
                    card_dict["reward_rates"] = catalog_card.get("reward_rates", {})
                    card_dict["card_id"] = catalog_card.get("card_id", "")
                    card_dict["annual_fee"] = catalog_card.get("annual_fee", 0.0)
                    break
            explanation_cards.append(card_dict)

        llm_explanations, explanation_latency_ms = await _generate_llm_explanations(
            cards=explanation_cards,
            spending_categories=normalized_categories,
            user_profile=user_profile,
            top_n=LLM_TOP_N_EXPLANATIONS,
        )

        # Overlay LLM explanations onto recommendations
        if llm_explanations:
            for rec in recommendations:
                if rec.card_name in llm_explanations:
                    rec = rec.model_copy(
                        update={"explanation": llm_explanations[rec.card_name]}
                    )
                    # Update in-place by index
                    for idx, r in enumerate(recommendations):
                        if r.card_name == rec.card_name:
                            recommendations[idx] = rec
                            break

        total_ms += explanation_latency_ms
        stage_latency_ms["llm_explanation"] = round(explanation_latency_ms, 3)

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

    # --- Story 2.5: Inference Logging for Monitoring ---
    model_version = get_model_version() or "unloaded"
    log_record = build_log_record(
        request_id=request_id,
        user_hash=user_hash,
        input_features={
            "spending_categories": payload.spending_categories,
            "monthly_spend": payload.monthly_spend,
            "preferred_rewards": payload.preferred_rewards,
            "transaction_history_count": len(payload.transaction_history),
        },
        scores=telemetry.get("scores", []),
        top_card=(recommendations[0].card_name if recommendations else "none"),
        model_version=model_version,
        latency_breakdown=stage_latency_ms,
        is_personalized=telemetry.get("is_personalized", False),
        explanation_latency_ms=(
            explanation_latency_ms if explanation_latency_ms > 0 else None
        ),
    )
    background_tasks.add_task(log_inference, log_record)

    return PredictionResponse(
        recommended_cards=recommendations,
        model_version=model_version,
        inference_latency_ms=round(total_ms, 3),
    )
