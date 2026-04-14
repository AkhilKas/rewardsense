"""Shared card catalog loader.

Loads the full card catalog from the data pipeline output
(``merged_cards.json``) and merges it with curated cards that carry
richer reward-rate detail (category bonuses, key benefits, etc.).

Both the inference API (``src/serving/app.py``) and the frontend-facing
app server (``src/app/server.py``, ``src/app/users/router.py``) import
from this module so there is a single source of truth.

Exposed singletons (loaded once at import time):
    CARD_CATALOG            – scoring-format dicts (all cards)
    CARD_CATALOG_BY_ID      – {card_id: card_dict}
    DISPLAY_CATALOG         – List[CardCatalogItem] for the UI
    DISPLAY_CATALOG_BY_ID   – {card_id: CardCatalogItem}
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.app.users.schemas import CardCatalogItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default path to the pipeline output
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/app/cards -> repo root
DEFAULT_CATALOG_PATH = (
    _REPO_ROOT / "data" / "processed" / "current" / "offers" / "merged_cards.json"
)

# ---------------------------------------------------------------------------
# 9 Curated cards — hand-verified reward structures with category bonuses
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _slugify(value: str) -> str:
    """Turn a card name into a URL/ID-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown_card"


def _parse_rate(raw: Any) -> Optional[float]:
    """Parse a reward rate from various formats ('1.5%', '2x', 2.0)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        cleaned = raw.strip().lower().replace("%", "").replace("x", "")
        match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
        if not match:
            return None
        value = float(match.group(1))
    else:
        match = re.search(r"(\d+(?:\.\d+)?)", str(raw))
        if not match:
            return None
        value = float(match.group(1))

    if value < 0 or value > 10:
        return None
    return value


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_catalog_from_offers(path: Path) -> List[Dict[str, Any]]:
    """Load and deduplicate cards from a merged_cards.json file."""
    if not path.exists():
        logger.info("Card catalog file not found at %s", path)
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

        # Try to get reward_rates from enriched data first, fall back to base_reward_rate
        reward_rates = item.get("reward_rates")
        if isinstance(reward_rates, dict) and "universal_base_rate" in reward_rates:
            # Already in scoring format (e.g. from creditcardbonuses API enrichment)
            pass
        else:
            rate = _parse_rate(item.get("base_reward_rate"))
            if rate is None:
                continue
            reward_rates = {"universal_base_rate": rate}

        card_id = str(item.get("card_id") or _slugify(raw_name))
        annual_fee = item.get("annual_fee", 0.0)
        try:
            annual_fee = float(annual_fee)
        except (TypeError, ValueError):
            annual_fee = 0.0

        candidate: Dict[str, Any] = {
            "card_id": card_id,
            "card_name": raw_name,
            "issuer": item.get("issuer") or "",
            "annual_fee": max(annual_fee, 0.0),
            "reward_rates": reward_rates,
        }
        # Carry through optional fields if present
        if item.get("key_benefits"):
            candidate["key_benefits"] = item["key_benefits"]
        if item.get("image_url"):
            candidate["image_url"] = item["image_url"]

        existing = cards_by_id.get(card_id)
        if existing is None:
            cards_by_id[card_id] = candidate
            continue

        # Keep the version with the higher base rate
        existing_rate = float(
            existing.get("reward_rates", {}).get("universal_base_rate", 0.0)
        )
        new_rate = float(reward_rates.get("universal_base_rate", 0.0))
        if new_rate > existing_rate:
            cards_by_id[card_id] = candidate

    return list(cards_by_id.values())


_GCS_BUCKET = "us-central1-rewardsense-com-8e7127ac-bucket"
_GCS_CATALOG_OBJECT = "data/processed/current/offers/merged_cards.json"


def _fetch_catalog_from_gcs(dest: Path) -> bool:
    """Download merged_cards.json from GCS into dest.

    Returns True on success, False on any error (missing creds, network, etc.).
    Silently skips so callers fall back to curated cards.
    """
    try:
        from google.cloud import storage  # type: ignore[import]

        client = storage.Client()
        bucket = client.bucket(_GCS_BUCKET)
        blob = bucket.blob(_GCS_CATALOG_OBJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        logger.info(
            "Fetched card catalog from GCS: %s (%d bytes)", dest, dest.stat().st_size
        )
        return True
    except Exception as exc:
        logger.debug(
            "GCS catalog fetch skipped (%s) — using local/curated fallback", exc
        )
        return False


def _norm(s: str) -> str:
    """Strip non-alphanumeric characters and lowercase for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _pipeline_card_is_covered_by_curated(
    pipeline_card: Dict[str, Any],
    curated_index: Dict[str, Dict[str, Any]],
) -> bool:
    """Return True if a pipeline card is a duplicate of a curated card.

    Matches on same normalised issuer AND one name being a substring of
    the other (e.g. pipeline "Gold" / issuer AMERICAN_EXPRESS is covered by
    curated "Amex Gold Card" / issuer American Express).
    """
    p_name = _norm(str(pipeline_card.get("card_name", "")))
    if not p_name:
        return False

    p_tokens = set(re.split(r"[^a-z0-9]+", _norm(str(pipeline_card.get("issuer", "")))))

    for curated in curated_index.values():
        c_name = _norm(str(curated.get("card_name", "")))

        # Issuer must share a common token (handles "AMERICAN_EXPRESS" vs "American Express")
        c_tokens = set(re.split(r"[^a-z0-9]+", _norm(str(curated.get("issuer", "")))))
        issuer_overlap = bool(p_tokens & c_tokens - {""})

        if not issuer_overlap:
            continue

        # Name match: one is a substring of the other
        if p_name in c_name or c_name in p_name:
            return True

    return False


def load_card_catalog(
    catalog_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load the full card catalog (pipeline + curated).

    Resolution order:
    1. Local merged_cards.json (CARD_CATALOG_PATH env var or default path)
    2. GCS fetch if local file is absent and google-cloud-storage is available
    3. 9 curated hardcoded cards as final fallback

    Curated cards always take precedence. Pipeline cards that are fuzzy-matched
    to a curated card (same issuer + name is substring of the other) are dropped
    so the correct hand-verified reward rates are shown rather than the
    incomplete API data (e.g. the API stores Amex Gold as "Gold" with only
    universal_base_rate=1.0, missing the 4x dining/groceries bonuses).
    """
    path = Path(
        catalog_path or os.getenv("CARD_CATALOG_PATH", str(DEFAULT_CATALOG_PATH))
    )

    if not path.exists():
        _fetch_catalog_from_gcs(path)

    loaded_cards = _load_catalog_from_offers(path)

    # Build curated index for duplicate detection
    curated_index: Dict[str, Dict[str, Any]] = {
        c["card_id"]: c for c in CURATED_CARD_CATALOG
    }

    cards_by_id: Dict[str, Dict[str, Any]] = {}
    skipped = 0
    for card in loaded_cards:
        if _pipeline_card_is_covered_by_curated(card, curated_index):
            skipped += 1
            continue
        cards_by_id[card["card_id"]] = dict(card)

    if skipped:
        logger.debug("Dropped %d pipeline cards shadowed by curated entries", skipped)

    # Curated cards override any remaining pipeline entries with the same card_id
    for card in CURATED_CARD_CATALOG:
        cards_by_id[card["card_id"]] = dict(card)

    catalog = list(cards_by_id.values())
    if not catalog:
        return list(CURATED_CARD_CATALOG)
    return catalog


# ---------------------------------------------------------------------------
# Display format conversion
# ---------------------------------------------------------------------------


def _generate_reward_highlights(card: Dict[str, Any]) -> List[str]:
    """Generate human-readable reward highlights from reward_rates."""
    rates = card.get("reward_rates", {})
    highlights: List[str] = []

    bonuses = rates.get("category_bonuses", {})
    for cat, rate in sorted(bonuses.items(), key=lambda x: -x[1]):
        highlights.append(f"{rate:g}x {cat}")

    base = rates.get("universal_base_rate", 0.0)
    if base > 0:
        highlights.append(
            f"{base:g}% on everything else" if bonuses else f"{base:g}% on everything"
        )

    fee = card.get("annual_fee", 0)
    if fee == 0:
        highlights.append("No annual fee")

    return highlights[:4]  # Cap at 4 highlights for UI


def _to_display_item(card: Dict[str, Any]) -> CardCatalogItem:
    """Convert a scoring-format card dict to a CardCatalogItem for the UI."""
    return CardCatalogItem(
        card_id=card["card_id"],
        card_name=card["card_name"],
        issuer=card.get("issuer") or "Unknown",
        annual_fee=card.get("annual_fee", 0.0),
        reward_highlights=card.get("key_benefits") or _generate_reward_highlights(card),
        image_url=card.get("image_url"),
    )


def _build_display_catalog(
    scoring_catalog: List[Dict[str, Any]],
) -> List[CardCatalogItem]:
    """Convert the full scoring catalog to display format."""
    return [_to_display_item(card) for card in scoring_catalog]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_scoring_rates(card_id: str) -> Dict[str, Any]:
    """Return ``{"reward_rates": {...}}`` for a card, suitable for unpacking
    into a portfolio entry.  Falls back to a 1% base rate for unknown cards.
    """
    card = CARD_CATALOG_BY_ID.get(card_id)
    if card is None:
        return {"reward_rates": {"universal_base_rate": 1.0}}
    return {"reward_rates": card["reward_rates"]}


# ---------------------------------------------------------------------------
# Module-level singletons — loaded once on first import
# ---------------------------------------------------------------------------

CARD_CATALOG: List[Dict[str, Any]] = load_card_catalog()
CARD_CATALOG_BY_ID: Dict[str, Dict[str, Any]] = {c["card_id"]: c for c in CARD_CATALOG}

DISPLAY_CATALOG: List[CardCatalogItem] = _build_display_catalog(CARD_CATALOG)
DISPLAY_CATALOG_BY_ID: Dict[str, CardCatalogItem] = {
    c.card_id: c for c in DISPLAY_CATALOG
}

logger.info(
    "Card catalog loaded: %d cards (%d curated)",
    len(CARD_CATALOG),
    len(CURATED_CARD_CATALOG),
)
