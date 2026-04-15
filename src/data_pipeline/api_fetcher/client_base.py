from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class APIClientError(RuntimeError):
    """Base error for API client failures."""


class APIClientTimeout(APIClientError):
    """Raised when request times out."""


class APIClientHTTPError(APIClientError):
    """Raised for non-success HTTP responses."""


class BaseAPIClient:
    """
    Reusable base HTTP client for external APIs.

    Features:
    - Persistent session
    - Default headers
    - Retry with exponential backoff
    - Configurable timeout
    - Safe JSON parsing
    """

    DEFAULT_TIMEOUT = 15  # seconds
    DEFAULT_RETRIES = 3
    DEFAULT_BACKOFF_FACTOR = 0.5

    def __init__(
        self,
        base_url: str,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        self.session = requests.Session()

        # Default headers
        headers = {
            "User-Agent": "RewardSense/1.0",
            "Accept": "application/json",
        }

        if default_headers:
            headers.update(default_headers)

        self.session.headers.update(headers)

        # Retry strategy
        retry_strategy = Retry(
            total=retries if retries is not None else self.DEFAULT_RETRIES,
            backoff_factor=(
                backoff_factor
                if backoff_factor is not None
                else self.DEFAULT_BACKOFF_FACTOR
            ),
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ---------------------------------------------------
    # Core request method
    # ---------------------------------------------------
    def get_json(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Send GET request and return parsed JSON.
        Raises clean, structured errors.
        """

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.Timeout as e:
            raise APIClientTimeout(f"Request timed out calling {url}") from e
        except requests.RequestException as e:
            raise APIClientError(f"Request failed calling {url}") from e

        if response.status_code >= 400:
            raise APIClientHTTPError(f"HTTP {response.status_code} returned from {url}")

        try:
            return response.json()
        except ValueError as e:
            raise APIClientError(f"Invalid JSON returned from {url}") from e
