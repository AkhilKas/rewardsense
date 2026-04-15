"""
RewardSense - API Fetcher Module

This module provides clients for fetching credit card data from external APIs.

Why this module exists:
----------------------
Web scraping (Epic 2.1) gives us card details from aggregator websites, but:
1. Websites can block scrapers or change HTML structure
2. APIs provide more reliable, structured data
3. APIs often have data not on public websites (e.g., real-time bonus values)

This module provides a fallback/complement to scraping.

Usage:
------
    from data_pipeline.api_fetcher import CreditCardBonusesClient, CardOffer

    # Fetch raw offers
    client = CreditCardBonusesClient()
    raw_offers = client.fetch_current_offers()

    # Or fetch already normalized (recommended)
    normalized_offers = client.fetch_normalized_offers()

Configuration:
--------------
Set these environment variables (optional):

    CREDITCARDBONUSES_API_KEY      - If you have paid API access
    CREDITCARDBONUSES_BASE_URL     - Required if using API key
    CREDITCARDBONUSES_EXPORT_URL   - Override default public export URL
    CREDITCARDBONUSES_TIMEOUT_SEC  - Request timeout (default: 15)

If no API key is set, the client falls back to the free public GitHub export.
"""

# -----------------------------------------------------------------------------
# Base client (reusable for other APIs in the future)
# -----------------------------------------------------------------------------
from .client_base import (
    BaseAPIClient,
    APIClientError,
    APIClientHTTPError,
    APIClientTimeout,
)

# -----------------------------------------------------------------------------
# Credit Card Bonuses API client
# -----------------------------------------------------------------------------
from .credit_card_bonuses_api import (
    CreditCardBonusesClient,
    CreditCardBonusesConfigError,
    CreditCardBonusesUpstreamError,
)

# -----------------------------------------------------------------------------
# Data normalization (converts raw API responses to clean schema)
# -----------------------------------------------------------------------------
from .normalizer import normalize_creditcardbonuses_offer

# -----------------------------------------------------------------------------
# Unified schema (same structure used by scrapers and API)
# -----------------------------------------------------------------------------
from .schema import CardOffer

__all__ = [
    # Base client
    "BaseAPIClient",
    "APIClientError",
    "APIClientHTTPError",
    "APIClientTimeout",
    # Credit Card Bonuses client
    "CreditCardBonusesClient",
    "CreditCardBonusesConfigError",
    "CreditCardBonusesUpstreamError",
    # Normalizer
    "normalize_creditcardbonuses_offer",
    # Schema
    "CardOffer",
]
