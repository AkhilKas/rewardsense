"""
RewardSense - Unit Tests for Issuer Scrapers

Tests for issuer-specific scrapers:
    ✅ ChaseScraper - Full tests (working)
    ✅ DiscoverScraper - Full tests (working)
    ✅ AmexScraper - Full tests (Selenium, mocked in CI)
    ✅ CitiScraper - Full tests (Selenium, mocked in CI)
    ✅ CapitalOneScraper - Full tests (Selenium, mocked in CI)

Run with: pytest tests/test_issuer_scrapers.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup

try:
    from selenium.common.exceptions import TimeoutException
except ImportError:
    TimeoutException = Exception  # type: ignore[misc,assignment]

import sys

sys.path.insert(0, "src")

from data_pipeline.scrapers.issuer_scrapers import (
    ChaseScraper,
    AmexScraper,
    CitiScraper,
    CapitalOneScraper,
    DiscoverScraper,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def chase_scraper():
    """Provides a ChaseScraper instance."""
    return ChaseScraper()


@pytest.fixture
def discover_scraper():
    """Provides a DiscoverScraper instance."""
    return DiscoverScraper()


@pytest.fixture
def chase_html():
    """Provides sample Chase card listing HTML matching real site structure."""
    return """
    <html>
    <body>
        <div class="cardsummarylist">
            <div class="cmp-cardsummary-container">
                <div class="cmp-cardsummary--list-view selected">
                    <div class="cmp-cardsummary--list-view--personal">
                        <div class="cmp-cardsummary__inner-container">
                            <div class="cmp-cardsummary__inner-container__title">
                                <h2>Chase Freedom Unlimited® Credit Card Links to product page</h2>
                            </div>
                            <div class="cmp-cardsummary__inner-container--annual-fee">
                                <p>$0 Annual Fee</p>
                            </div>
                            <div class="cmp-cardsummary__inner-container--card-member-offer">
                                <p>NEW CARDMEMBER OFFER Earn a $200 bonus after spending $500</p>
                            </div>
                            <a href="/freedom-unlimited">Apply Now</a>
                            <div class="cmp-cardsummary__inner-container__image">
                                <img src="https://chase.com/freedom.png" alt="Card">
                            </div>
                        </div>
                    </div>
                </div>
                <div class="cmp-cardsummary--list-view selected">
                    <div class="cmp-cardsummary--list-view--personal">
                        <div class="cmp-cardsummary__inner-container">
                            <div class="cmp-cardsummary__inner-container__title">
                                <h2>Chase Sapphire Reserve® Credit Card Links to product page</h2>
                            </div>
                            <div class="cmp-cardsummary__inner-container--annual-fee">
                                <p>$550 Annual Fee</p>
                            </div>
                            <div class="cmp-cardsummary__inner-container--card-member-offer">
                                <p>NEW CARDMEMBER OFFER Earn 75,000 bonus points after spending $4,000</p>
                            </div>
                            <a href="/sapphire-reserve">Learn More</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def discover_html():
    """Provides sample Discover card listing HTML."""
    return """
    <html>
    <body>
        <div class="card-container">
            <h2>Discover it® Cash Back</h2>
            <p>No annual fee</p>
            <p>Earn 5% cash back on rotating categories</p>
            <a href="/credit-cards/cash-back">Learn More</a>
        </div>
        <div class="card-container">
            <h2>Discover it® Cash Back</h2>
            <p>No annual fee - duplicate entry</p>
            <a href="/credit-cards/cash-back">Learn More</a>
        </div>
        <div class="card-container">
            <h2>Discover it® Miles</h2>
            <p>No annual fee</p>
            <p>Earn 1.5X miles on every purchase</p>
            <a href="/credit-cards/miles">Apply</a>
        </div>
        <div class="card-container">
            <h3>Not A Card - Just Content</h3>
            <p>Some random content</p>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def empty_js_html():
    """Provides HTML that simulates a JS-rendered page (empty body)."""
    return """
    <html>
    <head><title>Credit Cards</title></head>
    <body>
        <noscript>Please enable JavaScript</noscript>
    </body>
    </html>
    """


@pytest.fixture
def amex_html():
    """Provides sample Amex card listing HTML matching real DOM structure."""
    return """
    <html>
    <body>
        <div class="_cardTileContainer_16cp2_32">
            <h2 class="flex flex-align-items-center _cardTileCardNameTitle_16cp2_128">American Express® Gold Card</h2>
            <div>Annual Fee: $250</div>
            <div>60,000 Membership Rewards points</div>
        </div>
        <div class="_cardTileContainer_16cp2_32">
            <h2 class="flex flex-align-items-center _cardTileCardNameTitle_16cp2_128">Blue Cash Everyday® Card</h2>
            <div>No Annual Fee $0</div>
            <div>$200 cash back</div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def citi_html():
    """Provides sample Citi card listing HTML matching real DOM structure."""
    return """
    <html>
    <body>
        <div class="content-container h-100">
            <h3 class="cds-text-header card-name cds-text-header-3">Citi® Double Cash Card</h3>
            <div>$0 Annual Fee</div>
        </div>
        <div class="content-container h-100">
            <h3 class="cds-text-header card-name cds-text-header-3">Citi Premier® Card</h3>
            <div>$95 Annual Fee</div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def capone_html():
    """Provides sample Capital One card listing HTML matching real DOM structure."""
    return """
    <html>
    <body>
        <div class="card-details-container">
            <button class="heading default-view-heading ng-star-inserted">Venture Rewards</button>
            <div class="attribute-list">$95 annual fee 2X miles on every purchase Earn 75,000 bonus miles</div>
        </div>
        <div class="card-details-container">
            <button class="heading default-view-heading ng-star-inserted">Quicksilver Rewards</button>
            <div class="attribute-list">$0 annual fee 1.5% cash back on every purchase $200 cash bonus</div>
        </div>
    </body>
    </html>
    """


# =============================================================================
# ChaseScraper Tests
# =============================================================================


class TestChaseScraper:
    """Tests for ChaseScraper."""

    def test_get_source_name(self, chase_scraper):
        """
        Given: A ChaseScraper instance
        When: get_source_name is called
        Then: It should return "Chase"
        """
        # Given / When
        name = chase_scraper.get_source_name()

        # Then
        assert name == "Chase"

    def test_get_card_list_urls_returns_chase_urls(self, chase_scraper):
        """
        Given: A ChaseScraper instance
        When: get_card_list_urls is called
        Then: All URLs should contain "creditcards.chase.com"
        """
        # Given / When
        urls = chase_scraper.get_card_list_urls()

        # Then
        assert len(urls) > 0
        assert all("creditcards.chase.com" in url for url in urls)

    def test_parse_card_listing_extracts_cards(self, chase_scraper, chase_html):
        """
        Given: HTML with Chase card elements
        When: parse_card_listing is called
        Then: Cards should be extracted
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)

        # Then
        assert len(cards) == 2

    def test_parse_card_listing_extracts_card_name(self, chase_scraper, chase_html):
        """
        Given: HTML with Chase cards
        When: parse_card_listing is called
        Then: Card names should be cleaned (no "Credit Card Links to..." suffix)
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)
        freedom = next((c for c in cards if "Freedom" in c.get("name", "")), None)

        # Then
        assert freedom is not None
        assert "Links to" not in freedom["name"]
        assert "Credit Card" not in freedom["name"]
        assert "Chase Freedom Unlimited" in freedom["name"]

    def test_parse_card_listing_sets_issuer_to_chase(self, chase_scraper, chase_html):
        """
        Given: HTML with Chase cards
        When: parse_card_listing is called
        Then: All cards should have issuer set to "Chase"
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            assert card.get("issuer") == "Chase"

    def test_parse_card_listing_sets_source_to_chase(self, chase_scraper, chase_html):
        """
        Given: HTML with Chase cards
        When: parse_card_listing is called
        Then: All cards should have source set to "Chase"
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            assert card.get("source") == "Chase"

    def test_parse_card_listing_extracts_zero_annual_fee(
        self, chase_scraper, chase_html
    ):
        """
        Given: HTML with a $0 annual fee card
        When: parse_card_listing is called
        Then: Annual fee should be 0
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)
        freedom = next((c for c in cards if "Freedom" in c.get("name", "")), None)

        # Then
        assert freedom is not None
        assert freedom.get("annual_fee") == 0

    def test_parse_card_listing_extracts_numeric_annual_fee(
        self, chase_scraper, chase_html
    ):
        """
        Given: HTML with a $550 annual fee card
        When: parse_card_listing is called
        Then: Annual fee should be 550
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)
        reserve = next((c for c in cards if "Reserve" in c.get("name", "")), None)

        # Then
        assert reserve is not None
        assert reserve.get("annual_fee") == 550

    def test_parse_card_listing_extracts_cash_bonus(self, chase_scraper, chase_html):
        """
        Given: HTML with "$200 bonus" offer
        When: parse_card_listing is called
        Then: Welcome bonus should contain "$200"
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)
        freedom = next((c for c in cards if "Freedom" in c.get("name", "")), None)

        # Then
        assert freedom is not None
        assert freedom.get("welcome_bonus") is not None
        assert "$200" in freedom["welcome_bonus"]

    def test_parse_card_listing_extracts_points_bonus(self, chase_scraper, chase_html):
        """
        Given: HTML with "75,000 bonus points" offer
        When: parse_card_listing is called
        Then: Welcome bonus should contain points info
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)
        reserve = next((c for c in cards if "Reserve" in c.get("name", "")), None)

        # Then
        assert reserve is not None
        assert reserve.get("welcome_bonus") is not None
        assert "75,000" in reserve["welcome_bonus"]

    def test_parse_card_listing_extracts_detail_url(self, chase_scraper, chase_html):
        """
        Given: HTML with card links
        When: parse_card_listing is called
        Then: Detail URLs should be absolute URLs
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)

        # Then
        cards_with_urls = [c for c in cards if c.get("detail_url")]
        assert len(cards_with_urls) > 0
        for card in cards_with_urls:
            assert card["detail_url"].startswith("http")

    def test_parse_card_listing_sets_scraped_at(self, chase_scraper, chase_html):
        """
        Given: HTML with Chase cards
        When: parse_card_listing is called
        Then: All cards should have scraped_at timestamp
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            assert card.get("scraped_at") is not None

    def test_parse_card_listing_empty_html_returns_empty_list(self, chase_scraper):
        """
        Given: Empty HTML
        When: parse_card_listing is called
        Then: Should return empty list
        """
        # Given
        soup = BeautifulSoup("<html><body></body></html>", "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)

        # Then
        assert cards == []


# =============================================================================
# DiscoverScraper Tests
# =============================================================================


class TestDiscoverScraper:
    """Tests for DiscoverScraper."""

    def test_get_source_name(self, discover_scraper):
        """
        Given: A DiscoverScraper instance
        When: get_source_name is called
        Then: It should return "Discover"
        """
        # Given / When
        name = discover_scraper.get_source_name()

        # Then
        assert name == "Discover"

    def test_get_card_list_urls_returns_discover_urls(self, discover_scraper):
        """
        Given: A DiscoverScraper instance
        When: get_card_list_urls is called
        Then: All URLs should contain "discover.com"
        """
        # Given / When
        urls = discover_scraper.get_card_list_urls()

        # Then
        assert len(urls) > 0
        assert all("discover.com" in url for url in urls)

    def test_parse_card_listing_deduplicates_cards(
        self, discover_scraper, discover_html
    ):
        """
        Given: HTML with duplicate card entries
        When: parse_card_listing is called
        Then: Duplicates should be removed
        """
        # Given
        soup = BeautifulSoup(discover_html, "lxml")

        # When
        cards = discover_scraper.parse_card_listing(soup)
        card_names = [c.get("name") for c in cards]

        # Then
        # Should have 2 unique cards (Cash Back and Miles), not 3
        assert len(cards) == 2
        assert len(card_names) == len(set(card_names))  # No duplicates

    def test_parse_card_listing_filters_non_card_content(
        self, discover_scraper, discover_html
    ):
        """
        Given: HTML with non-card content
        When: parse_card_listing is called
        Then: Non-card elements should be filtered out
        """
        # Given
        soup = BeautifulSoup(discover_html, "lxml")

        # When
        cards = discover_scraper.parse_card_listing(soup)
        card_names = [c.get("name", "").lower() for c in cards]

        # Then
        assert not any("not a card" in name for name in card_names)

    def test_parse_card_listing_sets_issuer_to_discover(
        self, discover_scraper, discover_html
    ):
        """
        Given: HTML with Discover cards
        When: parse_card_listing is called
        Then: All cards should have issuer set to "Discover"
        """
        # Given
        soup = BeautifulSoup(discover_html, "lxml")

        # When
        cards = discover_scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            assert card.get("issuer") == "Discover"

    def test_parse_card_listing_sets_zero_annual_fee(
        self, discover_scraper, discover_html
    ):
        """
        Given: HTML with "no annual fee" text
        When: parse_card_listing is called
        Then: Annual fee should be 0
        """
        # Given
        soup = BeautifulSoup(discover_html, "lxml")

        # When
        cards = discover_scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            assert card.get("annual_fee") == 0

    def test_normalize_name_removes_special_chars(self, discover_scraper):
        """
        Given: A card name with trademark symbols
        When: _normalize_name is called
        Then: Special characters should be removed
        """
        # Given
        name = "Discover it® Cash Back™"

        # When
        normalized = discover_scraper._normalize_name(name)

        # Then
        assert "®" not in normalized
        assert "™" not in normalized
        assert "discover it cash back" == normalized


# =============================================================================
# TODO Scraper Tests (Skipped)
# =============================================================================


class TestAmexScraper:
    """Tests for AmexScraper with mocked Selenium."""

    def test_get_source_name(self):
        """
        Given: An AmexScraper instance
        When: get_source_name is called
        Then: It should return "American Express"
        """
        scraper = AmexScraper()
        assert scraper.get_source_name() == "American Express"

    def test_get_card_list_urls_returns_amex_urls(self):
        """
        Given: An AmexScraper instance
        When: get_card_list_urls is called
        Then: All URLs should contain "americanexpress.com"
        """
        scraper = AmexScraper()
        urls = scraper.get_card_list_urls()
        assert len(urls) > 0
        assert all("americanexpress.com" in url for url in urls)

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    def test_parse_card_listing_with_mocked_selenium(self, mock_chrome, amex_html):
        """
        Given: Mocked Selenium driver returning Amex HTML
        When: parse_card_listing is called
        Then: Cards should be extracted correctly
        """
        # Given
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = amex_html

        scraper = AmexScraper()
        soup = BeautifulSoup(
            "", "lxml"
        )  # Dummy soup, ignored by Selenium implementation

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        assert len(cards) == 2

        gold_card = next((c for c in cards if "Gold Card" in c["name"]), None)
        assert gold_card is not None
        assert gold_card["annual_fee"] == 250
        assert "60,000" in gold_card["welcome_bonus"]

        blue_cash = next((c for c in cards if "Blue Cash" in c["name"]), None)
        assert blue_cash is not None
        assert blue_cash["annual_fee"] == 0
        assert "$200 cash back" in blue_cash["welcome_bonus"]

        # Verify Selenium calls
        mock_driver.get.assert_called()
        mock_driver.quit.assert_called_once()

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    def test_parse_card_listing_handles_selenium_error(self, mock_chrome):
        """
        Given: Selenium driver raises an exception
        When: parse_card_listing is called
        Then: Should handle error gracefully and return empty list
        """
        # Given
        mock_chrome.side_effect = Exception("Selenium failed")

        scraper = AmexScraper()
        soup = BeautifulSoup("", "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        assert cards == []

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    def test_parse_card_listing_handles_navigation_timeout(self, mock_chrome):
        """
        Given: Mocked driver timeouts on navigation
        When: parse_card_listing is called
        Then: Should catch exception and proceed/return result
        """
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        # Make get() raise TimeoutException, or just generic Exception for simplicity as the code catches generic Exception
        mock_driver.get.side_effect = TimeoutException("Timed out")

        scraper = AmexScraper()
        cards = scraper.parse_card_listing(BeautifulSoup("", "lxml"))

        # Should catch and return empty (since page_source wasn't reached/set)
        assert cards == []
        mock_driver.quit.assert_called()


class TestCitiScraper:
    """Tests for CitiScraper with mocked Selenium."""

    def test_get_source_name(self):
        scraper = CitiScraper()
        assert scraper.get_source_name() == "Citi"

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    def test_parse_card_listing_with_mocked_selenium(self, mock_chrome, citi_html):
        """
        Given: Mocked Selenium driver returning Citi HTML
        When: parse_card_listing is called
        Then: Cards should be extracted correctly
        """
        # Given
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = citi_html

        scraper = CitiScraper()

        # When
        cards = scraper.parse_card_listing(BeautifulSoup("", "lxml"))

        # Then
        assert len(cards) == 2

        double_cash = next((c for c in cards if "Double Cash" in c["name"]), None)
        assert double_cash is not None
        assert double_cash["annual_fee"] == 0

        premier = next((c for c in cards if "Premier" in c["name"]), None)
        assert premier is not None
        assert premier["annual_fee"] == 95

        mock_driver.quit.assert_called()


class TestCapitalOneScraper:
    """Tests for CapitalOneScraper with mocked Selenium."""

    def test_get_source_name(self):
        scraper = CapitalOneScraper()
        assert scraper.get_source_name() == "Capital One"

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    def test_parse_card_listing_with_mocked_selenium(self, mock_chrome, capone_html):
        """
        Given: Mocked Selenium driver returning Capital One HTML
        When: parse_card_listing is called
        Then: Cards should be extracted correctly
        """
        # Given
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = capone_html

        scraper = CapitalOneScraper()

        # When
        cards = scraper.parse_card_listing(BeautifulSoup("", "lxml"))

        # Then
        assert len(cards) == 2
        assert any("Venture" in c["name"] for c in cards)
        assert any("Quicksilver" in c["name"] for c in cards)

        venture = next((c for c in cards if "Venture" in c["name"]), None)
        assert venture["annual_fee"] == 95
        assert "75,000 bonus miles" in venture["welcome_bonus"]

        quicksilver = next((c for c in cards if "Quicksilver" in c["name"]), None)
        assert quicksilver["annual_fee"] == 0
        assert "$200 cash bonus" in quicksilver["welcome_bonus"]

        mock_driver.quit.assert_called()


# =============================================================================
# Common Behavior Tests
# =============================================================================


@patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
@patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
class TestAllIssuerScrapersCommonBehavior:
    """Tests for behavior common to all issuer scrapers."""

    def setup_method(self):
        self.scrapers = [
            ChaseScraper(),
            AmexScraper(),
            CitiScraper(),
            CapitalOneScraper(),
            DiscoverScraper(),
        ]

    def test_all_scrapers_return_list_from_parse_card_listing(self, mock_chrome):
        """
        Given: All issuer scrapers (with mocked Selenium)
        When: parse_card_listing is called with empty HTML
        Then: All should return a list (possibly empty)
        """
        # Mock the driver
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = "<html><body></body></html>"

        soup = BeautifulSoup("<html><body></body></html>", "lxml")

        # When / Then
        for scraper in self.scrapers:
            result = scraper.parse_card_listing(soup)
            assert isinstance(result, list), f"{scraper.get_source_name()} failed"

    def test_all_scrapers_have_non_empty_urls(self, mock_chrome):
        """
        Given: All issuer scrapers
        When: get_card_list_urls is called
        Then: All should return at least one URL
        """
        # When / Then
        for scraper in self.scrapers:
            urls = scraper.get_card_list_urls()
            assert len(urls) > 0, f"{scraper.get_source_name()} has no URLs"
            assert isinstance(urls, list)

    def test_all_scrapers_urls_are_https(self, mock_chrome):
        """
        Given: All issuer scrapers
        When: get_card_list_urls is called
        Then: All URLs should use HTTPS
        """
        # When / Then
        for scraper in self.scrapers:
            urls = scraper.get_card_list_urls()
            for url in urls:
                assert url.startswith(
                    "https://"
                ), f"{scraper.get_source_name()} has non-HTTPS URL: {url}"

    def test_all_scrapers_return_none_or_dict_from_parse_card_details(
        self, mock_chrome
    ):
        """
        Given: All issuer scrapers
        When: parse_card_details is called
        Then: Should return None or a dict
        """
        for scraper in self.scrapers:
            result = scraper.parse_card_details("http://example.com")
            assert result is None or isinstance(result, dict)


class TestAmexScraperEdgeCases:
    """Tests for specific edge cases in AmexScraper."""

    def test_parse_amex_card_missing_monitor_elements(self):
        scraper = AmexScraper()
        # Create a container that lacks required h2._cardTileCardNameTitle_ element
        soup = BeautifulSoup('<div class="_cardTileContainer_16cp2_32"></div>', "lxml")
        container = soup.div

        result = scraper._parse_amex_card(container)
        assert result is None

    def test_parse_amex_card_no_annual_fee_text(self):
        scraper = AmexScraper()
        soup = BeautifulSoup(
            """
            <div class="_cardTileContainer_16cp2_32">
                <h2 class="_cardTileCardNameTitle_16cp2_128">Test Card</h2>
                <div>Some other text</div>
            </div>
            """,
            "lxml",
        )
        container = soup.div
        card = scraper._parse_amex_card(container)
        assert card["name"] == "Test Card"
        assert card["annual_fee"] == 0

    def test_parse_amex_card_exception_safety(self):
        scraper = AmexScraper()
        # Pass something that causes an error, e.g. None or object without .find
        result = scraper._parse_amex_card(None)
        assert result is None

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    @patch("data_pipeline.scrapers.issuer_scrapers.WebDriverWait", create=True)
    def test_parse_card_listing_timeout_waits(self, mock_wait, mock_chrome):
        # Setup mock to raise TimeoutException when until() is called
        mock_wait.return_value.until.side_effect = Exception(
            "Timeout"
        )  # Using generic Exception as code catches generic Exception for warning

        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = "<html></html>"

        scraper = AmexScraper()
        soup = BeautifulSoup("", "lxml")

        # Should not raise exception, just log warning
        cards = scraper.parse_card_listing(soup)
        assert isinstance(cards, list)


class TestCitiScraperEdgeCases:
    """Tests for specific edge cases in CitiScraper."""

    def test_parse_citi_card_missing_title(self):
        scraper = CitiScraper()
        # Empty name element
        soup = BeautifulSoup('<h3 class="card-name"></h3>', "lxml")
        name_elem = soup.find("h3")
        result = scraper._parse_citi_card(name_elem, None)
        assert result is None

    def test_parse_citi_card_no_annual_fee_text(self):
        scraper = CitiScraper()
        soup = BeautifulSoup(
            """
            <div class="content-container">
                <h3 class="card-name">Citi Test Card</h3>
                <div>no annual fee</div>
            </div>
            """,
            "lxml",
        )
        name_elem = soup.find("h3", class_="card-name")
        container = soup.find("div", class_="content-container")
        card = scraper._parse_citi_card(name_elem, container)
        assert card["name"] == "Citi Test Card"
        assert card["annual_fee"] == 0

    def test_parse_citi_card_exception_safety(self):
        scraper = CitiScraper()
        result = scraper._parse_citi_card(None, None)
        assert result is None

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    @patch("data_pipeline.scrapers.issuer_scrapers.WebDriverWait", create=True)
    def test_parse_card_listing_timeout_waits(self, mock_wait, mock_chrome):
        mock_wait.return_value.until.side_effect = Exception("Timeout")

        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = "<html></html>"

        scraper = CitiScraper()
        cards = scraper.parse_card_listing(BeautifulSoup("", "lxml"))
        assert isinstance(cards, list)

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    def test_parse_card_listing_handles_selenium_error(self, mock_chrome):
        """
        Given: Selenium driver raises an exception
        When: parse_card_listing is called
        Then: Should handle error gracefully and return empty list
        """
        mock_chrome.side_effect = Exception("Selenium failed")

        scraper = CitiScraper()
        cards = scraper.parse_card_listing(BeautifulSoup("", "lxml"))
        assert cards == []


class TestCapitalOneScraperEdgeCases:
    """Tests for specific edge cases in CapitalOneScraper."""

    def test_parse_capone_card_missing_title(self):
        scraper = CapitalOneScraper()
        soup = BeautifulSoup('<div class="card-details-container"></div>', "lxml")
        container = soup.div
        result = scraper._parse_capone_card(container)
        assert result is None

    def test_parse_capone_card_exception_safety(self):
        scraper = CapitalOneScraper()
        result = scraper._parse_capone_card(None)
        assert result is None

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    @patch("data_pipeline.scrapers.issuer_scrapers.WebDriverWait", create=True)
    def test_parse_card_listing_timeout_waits(self, mock_wait, mock_chrome):
        mock_wait.return_value.until.side_effect = Exception("Timeout")

        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = "<html></html>"

        scraper = CapitalOneScraper()
        cards = scraper.parse_card_listing(BeautifulSoup("", "lxml"))
        assert isinstance(cards, list)

    @patch("data_pipeline.scrapers.issuer_scrapers.SELENIUM_AVAILABLE", True)
    @patch("data_pipeline.scrapers.issuer_scrapers.webdriver.Chrome", create=True)
    def test_parse_card_listing_handles_selenium_error(self, mock_chrome):
        """
        Given: Selenium driver raises an exception
        When: parse_card_listing is called
        Then: Should handle error gracefully and return empty list
        """
        mock_chrome.side_effect = Exception("Selenium failed")

        scraper = CapitalOneScraper()
        cards = scraper.parse_card_listing(BeautifulSoup("", "lxml"))
        assert cards == []


# =============================================================================
# Data Validation Tests
# =============================================================================


class TestIssuerScrapersDataValidation:
    """Tests for data validation across issuer scrapers."""

    def test_chase_cards_have_required_fields(self, chase_scraper, chase_html):
        """
        Given: Parsed Chase cards
        When: Checking card fields
        Then: Required fields should be present
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)

        # Then
        required_fields = ["source", "issuer", "scraped_at", "name"]
        for card in cards:
            for field in required_fields:
                assert field in card, f"Missing field: {field}"

    def test_annual_fee_is_non_negative(self, chase_scraper, chase_html):
        """
        Given: Parsed cards with annual fees
        When: Checking annual fee values
        Then: All fees should be >= 0
        """
        # Given
        soup = BeautifulSoup(chase_html, "lxml")

        # When
        cards = chase_scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            if card.get("annual_fee") is not None:
                assert card["annual_fee"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
