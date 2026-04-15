from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .normalizer import normalize_creditcardbonuses_offer
from .schema import CardOffer

from .client_base import (
    BaseAPIClient,
    APIClientError,
    APIClientHTTPError,
    APIClientTimeout,
)

logger = logging.getLogger(__name__)


class CreditCardBonusesConfigError(RuntimeError):
    """Raised when required environment configuration is missing or invalid."""


class CreditCardBonusesUpstreamError(RuntimeError):
    """Raised when the upstream source fails (HTTP errors, timeouts, invalid JSON)."""


class CreditCardBonusesClient(BaseAPIClient):
    """
    Dual-mode client:

    - If CREDITCARDBONUSES_API_KEY is set => keyed API mode (future).
    - Else => public export fallback mode (GitHub JSON export).

    Uses BaseAPIClient for all HTTP calls (session, retries, timeout, JSON parsing).
    """

    DEFAULT_EXPORT_URL = "https://raw.githubusercontent.com/andenacitelli/credit-card-bonuses-api/master/exports/data.json"

    def __init__(self) -> None:
        # Env vars
        self.api_key: Optional[str] = os.getenv("CREDITCARDBONUSES_API_KEY") or None
        self.base_url_env: Optional[str] = (
            os.getenv("CREDITCARDBONUSES_BASE_URL") or None
        )
        self.export_url: str = os.getenv(
            "CREDITCARDBONUSES_EXPORT_URL", self.DEFAULT_EXPORT_URL
        )

        timeout_raw = os.getenv("CREDITCARDBONUSES_TIMEOUT_SEC", "15").strip()
        try:
            timeout_sec = float(timeout_raw)
        except ValueError as e:
            raise CreditCardBonusesConfigError(
                f"CREDITCARDBONUSES_TIMEOUT_SEC must be a number, got '{timeout_raw}'."
            ) from e

        # Decide mode
        self.mode: str = "keyed_api" if self.api_key else "public_export"

        # Configure BaseAPIClient depending on mode
        if self.mode == "keyed_api":
            if not self.base_url_env:
                raise CreditCardBonusesConfigError(
                    "CREDITCARDBONUSES_BASE_URL must be set when CREDITCARDBONUSES_API_KEY is provided."
                )

            # Initialize BaseAPIClient with the real API base URL
            super().__init__(
                base_url=self.base_url_env,
                default_headers={
                    # Never log the key; just attach it to headers.
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=timeout_sec,
            )
            logger.info("CreditCardBonusesClient initialized in keyed_api mode.")

        else:
            if not self.export_url:
                raise CreditCardBonusesConfigError(
                    "CREDITCARDBONUSES_EXPORT_URL is missing and no API key was provided. "
                    "Set CREDITCARDBONUSES_EXPORT_URL (fallback) or set the keyed API env vars."
                )

            # Parse export_url into base_url + endpoint path so we can use BaseAPIClient.get_json()
            parsed = urlparse(self.export_url)
            if not parsed.scheme or not parsed.netloc:
                raise CreditCardBonusesConfigError(
                    f"CREDITCARDBONUSES_EXPORT_URL is not a valid URL: '{self.export_url}'"
                )

            self._export_endpoint = parsed.path  # includes leading '/'
            export_base_url = f"{parsed.scheme}://{parsed.netloc}"

            super().__init__(
                base_url=export_base_url,
                default_headers={},  # no auth in public export mode
                timeout=timeout_sec,
            )
            logger.info("CreditCardBonusesClient initialized in public_export mode.")

    # -------------------------------------------------
    # Public method
    # -------------------------------------------------
    def fetch_current_offers(self) -> List[Dict[str, Any]]:
        """
        Fetch raw offers from the configured upstream source.
        Returns a list[dict] (raw records), to be normalized later.
        """
        if self.mode == "keyed_api":
            data = self._fetch_from_keyed_api()
            return self._coerce_offers_list(data, source_hint="keyed_api")

        data = self._fetch_from_public_export()
        return self._coerce_offers_list(data, source_hint="public_export")

    def fetch_normalized_offers(self) -> List["CardOffer"]:
        """
        Fetch all offers and return them as normalized CardOffer objects.

        This is the recommended method for Story 2.4 orchestration because:
        1. Returns validated, clean data ready for merging with scraped data
        2. Filters out invalid/incomplete records automatically
        3. Logs statistics for monitoring data quality

        Returns:
            List of CardOffer objects (Pydantic models)

        Raises:
            CreditCardBonusesUpstreamError: If the API/export fetch fails

        Example:
            >>> client = CreditCardBonusesClient()
            >>> offers = client.fetch_normalized_offers()
            >>> print(f"Got {len(offers)} valid offers")
            >>> print(offers[0].card_name, offers[0].annual_fee)
        """
        # Step 1: Fetch raw data from API or public export
        raw_offers = self.fetch_current_offers()

        # Step 2: Normalize each offer, filtering out invalid ones
        normalized = []
        failed_count = 0

        for raw in raw_offers:
            offer = normalize_creditcardbonuses_offer(raw)
            if offer is not None:
                normalized.append(offer)
            else:
                failed_count += 1

        # Step 3: Log results for monitoring
        logger.info(
            f"CreditCardBonuses fetch complete: "
            f"{len(raw_offers)} raw → {len(normalized)} normalized "
            f"({failed_count} skipped due to missing/invalid data)"
        )

        return normalized

    def fetch_as_dicts(self) -> List[Dict[str, Any]]:
        """
        Fetch normalized offers and convert to plain dictionaries.

        This is useful for Story 2.4 when merging with scraper output,
        since scrapers return dicts, not Pydantic models.

        Returns:
            List of dictionaries with card data

        Example:
            >>> client = CreditCardBonusesClient()
            >>> cards = client.fetch_as_dicts()
            >>> # Now can merge with scraper output (also dicts)
            >>> all_cards = scraped_cards + cards
        """
        offers = self.fetch_normalized_offers()
        return [offer.model_dump() for offer in offers]

    # -------------------------------------------------
    # Mode implementations (now using BaseAPIClient.get_json)
    # -------------------------------------------------
    def _fetch_from_keyed_api(self) -> Any:
        """
        Future real API endpoint.
        Adjust endpoint once you have a real provider spec.
        """
        endpoint = "/offers"
        try:
            return self.get_json(endpoint=endpoint)
        except (APIClientTimeout, APIClientHTTPError, APIClientError) as e:
            raise CreditCardBonusesUpstreamError(f"Keyed API fetch failed: {e}") from e

    def _fetch_from_public_export(self) -> Any:
        """
        Temporary fallback: fetch the GitHub export JSON using BaseAPIClient.
        """
        try:
            return self.get_json(endpoint=self._export_endpoint)
        except (APIClientTimeout, APIClientHTTPError, APIClientError) as e:
            raise CreditCardBonusesUpstreamError(
                f"Public export fetch failed: {e}"
            ) from e

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    @staticmethod
    def _coerce_offers_list(data: Any, source_hint: str) -> List[Dict[str, Any]]:
        """
        Accept common shapes:
        - list[dict]
        - {"offers": list[dict]}
        - {"data": list[dict]}
        - {"results": list[dict]}
        """
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if isinstance(data, dict):
            for key in ("offers", "data", "results"):
                val = data.get(key)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]

        raise CreditCardBonusesUpstreamError(
            f"Unexpected {source_hint} response shape; expected list[object] "
            f"or object with offers/data/results list."
        )
