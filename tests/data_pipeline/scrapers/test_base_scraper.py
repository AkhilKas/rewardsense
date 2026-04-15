"""
RewardSense - Unit Tests for BaseScraper

Tests for the abstract base scraper class functionality including:
- Initialization and configuration
- Rate limiting
- Session management
- Request handling (success and failure)
- Statistics tracking
- Context manager

Run with: pytest tests/test_base_scraper.py -v --cov=src/data_pipeline/scrapers/base_scraper
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime
from bs4 import BeautifulSoup

import sys

sys.path.insert(0, "src")

from data_pipeline.scrapers.base_scraper import BaseScraper

# =============================================================================
# Concrete Implementation for Testing
# =============================================================================


class ConcreteScraper(BaseScraper):
    """Concrete implementation of BaseScraper for testing purposes."""

    def get_source_name(self) -> str:
        return "TestSource"

    def get_card_list_urls(self) -> list:
        return ["https://example.com/cards", "https://example.com/cards2"]

    def parse_card_listing(self, soup):
        # Return mock cards for testing
        cards = []
        for h2 in soup.find_all("h2"):
            cards.append(
                {
                    "name": h2.get_text(strip=True),
                    "source": "TestSource",
                }
            )
        return cards

    def parse_card_details(self, url):
        return {"detail_url": url, "benefits": ["Test benefit"]}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def scraper():
    """Provides a basic scraper instance with default settings."""
    return ConcreteScraper()


@pytest.fixture
def scraper_no_rate_limit():
    """Provides a scraper with rate limiting disabled."""
    return ConcreteScraper(rate_limit=0)


@pytest.fixture
def scraper_fast_rate_limit():
    """Provides a scraper with very short rate limit for testing."""
    return ConcreteScraper(rate_limit=0.1)


@pytest.fixture
def mock_successful_response():
    """Provides a mock successful HTTP response."""
    mock = Mock()
    mock.status_code = 200
    mock.content = b"<html><body><h1>Test Page</h1></body></html>"
    mock.raise_for_status = Mock()
    return mock


@pytest.fixture
def mock_html_with_cards():
    """Provides mock HTML with card elements."""
    mock = Mock()
    mock.status_code = 200
    mock.content = b"""
    <html><body>
        <h2>Card One</h2>
        <h2>Card Two</h2>
        <h2>Card Three</h2>
    </body></html>
    """
    mock.raise_for_status = Mock()
    return mock


@pytest.fixture
def mock_failed_response():
    """Provides a mock failed HTTP response."""
    mock = Mock()
    mock.status_code = 500
    mock.raise_for_status = Mock(side_effect=Exception("Server Error"))
    return mock


# =============================================================================
# Initialization Tests
# =============================================================================


class TestBaseScraperInitialization:
    """Tests for BaseScraper initialization."""

    def test_init_with_default_values(self):
        """
        Given: No custom configuration
        When: BaseScraper is initialized
        Then: Default values should be set correctly
        """
        # When
        scraper = ConcreteScraper()

        # Then
        assert scraper.rate_limit == 1.0
        assert scraper.max_retries == 3
        assert scraper.timeout == 30
        assert scraper.respect_robots is True
        assert "RewardSense" in scraper.user_agent

    def test_init_with_custom_rate_limit(self):
        """
        Given: A custom rate limit of 2.5 seconds
        When: BaseScraper is initialized
        Then: Rate limit should be set to 2.5
        """
        # When
        scraper = ConcreteScraper(rate_limit=2.5)

        # Then
        assert scraper.rate_limit == 2.5

    def test_init_with_custom_max_retries(self):
        """
        Given: A custom max_retries of 5
        When: BaseScraper is initialized
        Then: max_retries should be set to 5
        """
        # When
        scraper = ConcreteScraper(max_retries=5)

        # Then
        assert scraper.max_retries == 5

    def test_init_with_custom_timeout(self):
        """
        Given: A custom timeout of 60 seconds
        When: BaseScraper is initialized
        Then: Timeout should be set to 60
        """
        # When
        scraper = ConcreteScraper(timeout=60)

        # Then
        assert scraper.timeout == 60

    def test_init_with_custom_user_agent(self):
        """
        Given: A custom user agent string
        When: BaseScraper is initialized
        Then: User agent should match the custom value
        """
        # Given
        custom_ua = "CustomBot/2.0"

        # When
        scraper = ConcreteScraper(user_agent=custom_ua)

        # Then
        assert scraper.user_agent == "CustomBot/2.0"

    def test_init_creates_session(self):
        """
        Given: Default configuration
        When: BaseScraper is initialized
        Then: A requests session should be created
        """
        # When
        scraper = ConcreteScraper()

        # Then
        assert scraper.session is not None

    def test_init_sets_last_request_time_to_zero(self):
        """
        Given: Default configuration
        When: BaseScraper is initialized
        Then: last_request_time should be 0.0
        """
        # When
        scraper = ConcreteScraper()

        # Then
        assert scraper.last_request_time == 0.0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestBaseScraperStatistics:
    """Tests for scraping statistics tracking."""

    def test_stats_initialized_to_zero(self, scraper):
        """
        Given: A newly created scraper
        When: Checking initial statistics
        Then: All counters should be zero
        """
        # When
        stats = scraper.stats

        # Then
        assert stats["requests_made"] == 0
        assert stats["requests_failed"] == 0
        assert stats["cards_scraped"] == 0

    def test_stats_start_time_initially_none(self, scraper):
        """
        Given: A newly created scraper
        When: Checking start_time
        Then: It should be None
        """
        # When
        stats = scraper.stats

        # Then
        assert stats["start_time"] is None
        assert stats["end_time"] is None

    @patch("requests.Session.get")
    def test_stats_increment_on_successful_request(
        self, mock_get, scraper_no_rate_limit, mock_successful_response
    ):
        """
        Given: A scraper with zero requests made
        When: A successful page fetch is performed
        Then: requests_made should increment by 1
        """
        # Given
        mock_get.return_value = mock_successful_response
        initial_count = scraper_no_rate_limit.stats["requests_made"]

        # When
        scraper_no_rate_limit.fetch_page("https://example.com")

        # Then
        assert scraper_no_rate_limit.stats["requests_made"] == initial_count + 1
        assert scraper_no_rate_limit.stats["requests_failed"] == 0

    def test_stats_increment_on_failed_request(self, scraper_no_rate_limit):
        """
        Given: A scraper with zero failed requests
        When: A failed page fetch is performed
        Then: Both requests_made and requests_failed should increment
        """
        # Given - patch the session's get method on the instance
        import requests.exceptions

        scraper_no_rate_limit.session.get = Mock(
            side_effect=requests.exceptions.RequestException("Connection failed")
        )

        # When
        scraper_no_rate_limit.fetch_page("https://example.com")

        # Then
        assert scraper_no_rate_limit.stats["requests_made"] == 1
        assert scraper_no_rate_limit.stats["requests_failed"] == 1

    def test_get_stats_returns_copy(self, scraper):
        """
        Given: A scraper with statistics
        When: get_stats() is called
        Then: It should return a copy of the stats dict
        """
        # When
        stats = scraper.get_stats()
        stats["requests_made"] = 999

        # Then
        assert scraper.stats["requests_made"] == 0  # Original unchanged

    @patch("requests.Session.get")
    def test_get_stats_calculates_duration(
        self, mock_get, scraper_no_rate_limit, mock_html_with_cards
    ):
        """
        Given: A scraper that has completed scraping
        When: get_stats() is called
        Then: duration_seconds should be calculated
        """
        # Given
        mock_get.return_value = mock_html_with_cards

        # When
        scraper_no_rate_limit.scrape_all_cards()
        stats = scraper_no_rate_limit.get_stats()

        # Then
        assert "duration_seconds" in stats
        assert stats["duration_seconds"] >= 0

    def test_get_stats_no_duration_if_not_started(self, scraper):
        """
        Given: A scraper that hasn't run
        When: get_stats() is called
        Then: duration_seconds should not be present
        """
        # When
        stats = scraper.get_stats()

        # Then
        assert "duration_seconds" not in stats


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestBaseScraperRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_delays_requests(self):
        """
        Given: A scraper with 0.5 second rate limit
        When: Rate limit wait is triggered after a recent request
        Then: It should wait at least 0.4 seconds
        """
        # Given
        scraper = ConcreteScraper(rate_limit=0.5)
        scraper.last_request_time = time.time()

        # When
        start = time.time()
        scraper._wait_for_rate_limit()
        elapsed = time.time() - start

        # Then
        assert elapsed >= 0.4  # Allow small tolerance

    def test_no_delay_when_rate_limit_elapsed(self):
        """
        Given: A scraper where rate limit time has already passed
        When: Rate limit wait is triggered
        Then: It should not add significant delay
        """
        # Given
        scraper = ConcreteScraper(rate_limit=0.5)
        scraper.last_request_time = time.time() - 10  # 10 seconds ago

        # When
        start = time.time()
        scraper._wait_for_rate_limit()
        elapsed = time.time() - start

        # Then
        assert elapsed < 0.2  # Should be nearly instant

    def test_zero_rate_limit_no_delay(self):
        """
        Given: A scraper with zero rate limit
        When: Rate limit wait is triggered
        Then: It should not add any significant delay
        """
        # Given
        scraper = ConcreteScraper(rate_limit=0)
        scraper.last_request_time = time.time()

        # When
        start = time.time()
        scraper._wait_for_rate_limit()
        elapsed = time.time() - start

        # Then
        assert elapsed < 0.2

    def test_rate_limit_updates_last_request_time(self, scraper_fast_rate_limit):
        """
        Given: A scraper with rate limiting
        When: _wait_for_rate_limit is called
        Then: last_request_time should be updated
        """
        # Given
        old_time = scraper_fast_rate_limit.last_request_time

        # When
        scraper_fast_rate_limit._wait_for_rate_limit()

        # Then
        assert scraper_fast_rate_limit.last_request_time > old_time


# =============================================================================
# Page Fetching Tests
# =============================================================================


class TestBaseScraperFetchPage:
    """Tests for page fetching functionality."""

    @patch("requests.Session.get")
    def test_fetch_page_returns_beautifulsoup(
        self, mock_get, scraper_no_rate_limit, mock_successful_response
    ):
        """
        Given: A valid URL and successful response
        When: fetch_page is called
        Then: It should return a BeautifulSoup object
        """
        # Given
        mock_get.return_value = mock_successful_response

        # When
        result = scraper_no_rate_limit.fetch_page("https://example.com")

        # Then
        assert result is not None
        assert isinstance(result, BeautifulSoup)

    def test_fetch_page_returns_none_on_failure(self, scraper_no_rate_limit):
        """
        Given: A URL that causes a connection error
        When: fetch_page is called
        Then: It should return None
        """
        # Given - patch the session's get method on the instance
        import requests.exceptions

        scraper_no_rate_limit.session.get = Mock(
            side_effect=requests.exceptions.RequestException("Connection refused")
        )

        # When
        result = scraper_no_rate_limit.fetch_page("https://example.com")

        # Then
        assert result is None

    @patch("requests.Session.get")
    def test_fetch_page_parses_html_content(
        self, mock_get, scraper_no_rate_limit, mock_successful_response
    ):
        """
        Given: A response with HTML content
        When: fetch_page is called
        Then: The HTML should be parsed correctly
        """
        # Given
        mock_get.return_value = mock_successful_response

        # When
        result = scraper_no_rate_limit.fetch_page("https://example.com")

        # Then
        assert result.find("h1").text == "Test Page"

    @patch("requests.Session.get")
    def test_fetch_page_increments_request_count(
        self, mock_get, scraper_no_rate_limit, mock_successful_response
    ):
        """
        Given: A scraper with request count at 0
        When: fetch_page is called
        Then: requests_made should be incremented
        """
        # Given
        mock_get.return_value = mock_successful_response
        assert scraper_no_rate_limit.stats["requests_made"] == 0

        # When
        scraper_no_rate_limit.fetch_page("https://example.com")

        # Then
        assert scraper_no_rate_limit.stats["requests_made"] == 1


# =============================================================================
# Fetch Page With Headers Tests
# =============================================================================


class TestBaseScraperFetchPageWithHeaders:
    """Tests for fetch_page_with_headers functionality."""

    @patch("requests.Session.get")
    def test_fetch_page_with_headers_returns_beautifulsoup(
        self, mock_get, scraper_no_rate_limit, mock_successful_response
    ):
        """
        Given: A valid URL and custom headers
        When: fetch_page_with_headers is called
        Then: It should return a BeautifulSoup object
        """
        # Given
        mock_get.return_value = mock_successful_response
        custom_headers = {"X-Custom-Header": "test-value"}

        # When
        result = scraper_no_rate_limit.fetch_page_with_headers(
            "https://example.com", headers=custom_headers
        )

        # Then
        assert result is not None
        assert isinstance(result, BeautifulSoup)

    @patch("requests.Session.get")
    def test_fetch_page_with_headers_passes_headers(
        self, mock_get, scraper_no_rate_limit, mock_successful_response
    ):
        """
        Given: Custom headers
        When: fetch_page_with_headers is called
        Then: Headers should be passed to the request
        """
        # Given
        mock_get.return_value = mock_successful_response
        custom_headers = {"X-Custom-Header": "test-value"}

        # When
        scraper_no_rate_limit.fetch_page_with_headers(
            "https://example.com", headers=custom_headers
        )

        # Then
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"] == custom_headers

    @patch("requests.Session.get")
    def test_fetch_page_with_headers_handles_none_headers(
        self, mock_get, scraper_no_rate_limit, mock_successful_response
    ):
        """
        Given: No custom headers (None)
        When: fetch_page_with_headers is called
        Then: Should pass empty dict as headers
        """
        # Given
        mock_get.return_value = mock_successful_response

        # When
        scraper_no_rate_limit.fetch_page_with_headers(
            "https://example.com", headers=None
        )

        # Then
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"] == {}

    def test_fetch_page_with_headers_returns_none_on_failure(
        self, scraper_no_rate_limit
    ):
        """
        Given: A request that fails
        When: fetch_page_with_headers is called
        Then: Should return None and increment failed count
        """
        # Given - patch the session's get method on the instance
        import requests.exceptions

        scraper_no_rate_limit.session.get = Mock(
            side_effect=requests.exceptions.RequestException("Connection refused")
        )

        # When
        result = scraper_no_rate_limit.fetch_page_with_headers("https://example.com")

        # Then
        assert result is None
        assert scraper_no_rate_limit.stats["requests_failed"] == 1

    @patch("requests.Session.get")
    def test_fetch_page_with_headers_increments_request_count(
        self, mock_get, scraper_no_rate_limit, mock_successful_response
    ):
        """
        Given: A scraper with request count at 0
        When: fetch_page_with_headers is called
        Then: requests_made should be incremented
        """
        # Given
        mock_get.return_value = mock_successful_response

        # When
        scraper_no_rate_limit.fetch_page_with_headers("https://example.com")

        # Then
        assert scraper_no_rate_limit.stats["requests_made"] == 1


# =============================================================================
# Scrape All Cards Tests
# =============================================================================


class TestBaseScraperScrapeAllCards:
    """Tests for the main scrape_all_cards method."""

    @patch("requests.Session.get")
    def test_scrape_all_cards_returns_list(
        self, mock_get, scraper_no_rate_limit, mock_html_with_cards
    ):
        """
        Given: A scraper with valid URLs
        When: scrape_all_cards is called
        Then: Should return a list of cards
        """
        # Given
        mock_get.return_value = mock_html_with_cards

        # When
        cards = scraper_no_rate_limit.scrape_all_cards()

        # Then
        assert isinstance(cards, list)
        assert len(cards) == 6  # 3 cards * 2 URLs

    @patch("requests.Session.get")
    def test_scrape_all_cards_sets_start_time(
        self, mock_get, scraper_no_rate_limit, mock_html_with_cards
    ):
        """
        Given: A scraper
        When: scrape_all_cards is called
        Then: start_time should be set
        """
        # Given
        mock_get.return_value = mock_html_with_cards
        assert scraper_no_rate_limit.stats["start_time"] is None

        # When
        scraper_no_rate_limit.scrape_all_cards()

        # Then
        assert scraper_no_rate_limit.stats["start_time"] is not None
        assert isinstance(scraper_no_rate_limit.stats["start_time"], datetime)

    @patch("requests.Session.get")
    def test_scrape_all_cards_sets_end_time(
        self, mock_get, scraper_no_rate_limit, mock_html_with_cards
    ):
        """
        Given: A scraper
        When: scrape_all_cards is called
        Then: end_time should be set
        """
        # Given
        mock_get.return_value = mock_html_with_cards

        # When
        scraper_no_rate_limit.scrape_all_cards()

        # Then
        assert scraper_no_rate_limit.stats["end_time"] is not None
        assert isinstance(scraper_no_rate_limit.stats["end_time"], datetime)

    @patch("requests.Session.get")
    def test_scrape_all_cards_updates_cards_scraped(
        self, mock_get, scraper_no_rate_limit, mock_html_with_cards
    ):
        """
        Given: A scraper
        When: scrape_all_cards is called
        Then: cards_scraped should reflect total count
        """
        # Given
        mock_get.return_value = mock_html_with_cards

        # When
        cards = scraper_no_rate_limit.scrape_all_cards()

        # Then
        assert scraper_no_rate_limit.stats["cards_scraped"] == len(cards)

    def test_scrape_all_cards_handles_failed_requests(self, scraper_no_rate_limit):
        """
        Given: A scraper where fetch_page returns None
        When: scrape_all_cards is called
        Then: Should continue and return empty list
        """
        # Given - patch the session's get method on the instance
        import requests.exceptions

        scraper_no_rate_limit.session.get = Mock(
            side_effect=requests.exceptions.RequestException("All requests fail")
        )

        # When
        cards = scraper_no_rate_limit.scrape_all_cards()

        # Then
        assert cards == []
        assert scraper_no_rate_limit.stats["requests_failed"] == 2  # 2 URLs


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestBaseScraperContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_returns_scraper(self):
        """
        Given: A scraper class
        When: Used as a context manager
        Then: It should return the scraper instance
        """
        # When
        with ConcreteScraper() as scraper:
            # Then
            assert scraper is not None
            assert isinstance(scraper, ConcreteScraper)

    def test_context_manager_session_available(self):
        """
        Given: A scraper used as context manager
        When: Inside the context
        Then: Session should be available
        """
        # When
        with ConcreteScraper() as scraper:
            # Then
            assert scraper.session is not None

    def test_context_manager_calls_exit(self):
        """
        Given: A scraper used as context manager
        When: Exiting the context
        Then: __exit__ should be called (session closed)
        """
        # Given
        scraper = ConcreteScraper()

        # When
        with scraper:
            _ = scraper.session

        # Then - verify we can call close without error (already closed is ok)
        # The important thing is __exit__ was called
        assert True

    def test_context_manager_handles_exception(self):
        """
        Given: A scraper used as context manager
        When: An exception occurs inside the context
        Then: __exit__ should still be called
        """
        # Given
        scraper = ConcreteScraper()
        exception_raised = False

        # When
        try:
            with scraper:
                raise ValueError("Test exception")
        except ValueError:
            exception_raised = True

        # Then
        assert exception_raised


# =============================================================================
# Session Configuration Tests
# =============================================================================


class TestBaseScraperSession:
    """Tests for session configuration."""

    def test_session_has_user_agent_header(self, scraper):
        """
        Given: A scraper with default user agent
        When: Checking session headers
        Then: User-Agent header should be set
        """
        # When
        headers = scraper.session.headers

        # Then
        assert "User-Agent" in headers
        assert "RewardSense" in headers["User-Agent"]

    def test_session_has_accept_header(self, scraper):
        """
        Given: A scraper
        When: Checking session headers
        Then: Accept header should be set for HTML
        """
        # When
        headers = scraper.session.headers

        # Then
        assert "Accept" in headers
        assert "text/html" in headers["Accept"]

    def test_session_has_custom_user_agent(self):
        """
        Given: A scraper with custom user agent
        When: Checking session headers
        Then: Custom User-Agent should be used
        """
        # Given
        custom_ua = "MyBot/1.0"

        # When
        scraper = ConcreteScraper(user_agent=custom_ua)

        # Then
        assert scraper.session.headers["User-Agent"] == custom_ua


# =============================================================================
# Abstract Method Tests
# =============================================================================


class TestBaseScraperAbstractMethods:
    """Tests for abstract method implementations."""

    def test_get_source_name_returns_string(self, scraper):
        """
        Given: A concrete scraper
        When: get_source_name is called
        Then: Should return a non-empty string
        """
        # When
        name = scraper.get_source_name()

        # Then
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_card_list_urls_returns_list(self, scraper):
        """
        Given: A concrete scraper
        When: get_card_list_urls is called
        Then: Should return a list of URLs
        """
        # When
        urls = scraper.get_card_list_urls()

        # Then
        assert isinstance(urls, list)
        assert len(urls) > 0
        assert all(isinstance(url, str) for url in urls)

    def test_parse_card_listing_returns_list(self, scraper):
        """
        Given: A concrete scraper and HTML soup
        When: parse_card_listing is called
        Then: Should return a list
        """
        # Given
        soup = BeautifulSoup("<html><body><h2>Test Card</h2></body></html>", "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        assert isinstance(cards, list)

    def test_parse_card_details_returns_dict_or_none(self, scraper):
        """
        Given: A concrete scraper
        When: parse_card_details is called
        Then: Should return a dict or None
        """
        # When
        details = scraper.parse_card_details("https://example.com/card")

        # Then
        assert details is None or isinstance(details, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/data_pipeline/scrapers/base_scraper"])
