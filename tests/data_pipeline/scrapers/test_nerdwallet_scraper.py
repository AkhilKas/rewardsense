"""
RewardSense - Unit Tests for NerdWalletScraper

Tests for NerdWallet-specific scraping functionality including:
- JSON-LD structured data extraction
- HTML card parsing
- Issuer extraction
- Fee and bonus extraction
- URL generation (updated for new site structure)

Run with: pytest tests/test_nerdwallet_scraper.py -v
"""

import pytest
from bs4 import BeautifulSoup

import sys

sys.path.insert(0, "src")

from data_pipeline.scrapers.nerdwallet_scraper import (
    NerdWalletScraper,
    NerdWalletSeleniumScraper,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def scraper():
    """Provides a NerdWalletScraper instance with default settings."""
    return NerdWalletScraper()


@pytest.fixture
def scraper_filtered():
    """Provides a NerdWalletScraper with filtered categories."""
    return NerdWalletScraper(categories=["cash_back", "travel"])


@pytest.fixture
def html_with_json_ld():
    """Provides HTML containing JSON-LD structured data."""
    return """
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@type": "Product",
            "name": "Chase Sapphire Preferred Card",
            "description": "Excellent travel rewards card",
            "url": "https://www.nerdwallet.com/card/chase-sapphire-preferred",
            "offers": {"price": "95"},
            "aggregateRating": {
                "ratingValue": 4.8,
                "reviewCount": 1250
            }
        }
        </script>
    </head>
    <body></body>
    </html>
    """


@pytest.fixture
def html_with_json_ld_item_list():
    """Provides HTML with JSON-LD itemListElement containing URL strings (not dicts)."""
    return """
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@type": "ItemList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": 1,
                    "item": "https://www.nerdwallet.com/card/some-card"
                },
                {
                    "@type": "ListItem",
                    "position": 2,
                    "item": {
                        "@type": "Product",
                        "name": "Amex Gold Card",
                        "offers": {"price": "250"}
                    }
                }
            ]
        }
        </script>
    </head>
    <body></body>
    </html>
    """


@pytest.fixture
def html_with_card_products():
    """Provides HTML containing card product elements."""
    return """
    <html>
    <body>
        <div class="card-product">
            <h2 class="card-name">Capital One Venture Rewards Credit Card</h2>
            <p>$95 annual fee</p>
            <p>Earn 2X miles on every purchase</p>
            <p>75,000 miles sign-up bonus</p>
            <a href="/cards/capital-one-venture">Learn More</a>
        </div>
        <div class="ProductCard">
            <h3>Citi Double Cash Card</h3>
            <p>No annual fee</p>
            <p>2% cash back on all purchases</p>
            <a href="/cards/citi-double-cash">Details</a>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def html_with_no_fee_card():
    """Provides HTML with a no annual fee card."""
    return """
    <div class="card-product">
        <h2>Discover it Cash Back</h2>
        <p>$0 Annual Fee</p>
        <p>5% cash back in rotating categories</p>
    </div>
    """


@pytest.fixture
def html_empty():
    """Provides empty HTML."""
    return "<html><body></body></html>"


@pytest.fixture
def html_malformed():
    """Provides malformed HTML."""
    return "<html><body><div>Unclosed div<p>Nested badly"


# =============================================================================
# Source Configuration Tests
# =============================================================================


class TestNerdWalletScraperConfiguration:
    """Tests for scraper configuration."""

    def test_get_source_name(self, scraper):
        """
        Given: A NerdWalletScraper instance
        When: get_source_name is called
        Then: It should return "NerdWallet"
        """
        # Given / When
        name = scraper.get_source_name()

        # Then
        assert name == "NerdWallet"

    def test_get_card_list_urls_returns_nerdwallet_urls(self, scraper):
        """
        Given: A NerdWalletScraper with default categories
        When: get_card_list_urls is called
        Then: All URLs should be NerdWallet URLs with new structure
        """
        # Given / When
        urls = scraper.get_card_list_urls()

        # Then
        assert len(urls) > 0
        assert all("nerdwallet.com" in url for url in urls)

    def test_get_card_list_urls_uses_new_url_structure(self, scraper):
        """
        Given: A NerdWalletScraper instance
        When: get_card_list_urls is called
        Then: URLs should use new structure (/credit-cards) not old (/best/credit-cards)
        """
        # Given / When
        urls = scraper.get_card_list_urls()

        # Then
        for url in urls:
            assert "/best/credit-cards" not in url, f"Old URL structure found: {url}"
            assert "/credit-cards" in url

    def test_get_card_list_urls_filtered_categories(self, scraper_filtered):
        """
        Given: A NerdWalletScraper with only cash_back and travel categories
        When: get_card_list_urls is called
        Then: Only 2 URLs should be returned
        """
        # Given / When
        urls = scraper_filtered.get_card_list_urls()

        # Then
        assert len(urls) == 2

    def test_base_url_is_https(self, scraper):
        """
        Given: A NerdWalletScraper instance
        When: Checking the BASE_URL
        Then: It should use HTTPS
        """
        # Given / When
        base_url = scraper.BASE_URL

        # Then
        assert base_url.startswith("https://")

    def test_category_urls_updated(self, scraper):
        """
        Given: A NerdWalletScraper instance
        When: Checking CATEGORY_URLS
        Then: URLs should reflect new site structure
        """
        # Given / When
        category_urls = scraper.CATEGORY_URLS

        # Then
        assert "all_cards" in category_urls
        assert category_urls["all_cards"] == "/credit-cards"
        assert "best_cards" in category_urls
        assert category_urls["best_cards"] == "/credit-cards/best"


# =============================================================================
# JSON-LD Parsing Tests
# =============================================================================


class TestNerdWalletJsonLdParsing:
    """Tests for JSON-LD structured data extraction."""

    def test_parse_json_ld_extracts_card_name(self, scraper, html_with_json_ld):
        """
        Given: HTML with JSON-LD containing a card name
        When: parse_card_listing is called
        Then: The card name should be extracted
        """
        # Given
        soup = BeautifulSoup(html_with_json_ld, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        assert len(cards) >= 1
        assert any("Chase Sapphire" in c.get("name", "") for c in cards)

    def test_parse_json_ld_extracts_annual_fee(self, scraper, html_with_json_ld):
        """
        Given: HTML with JSON-LD containing a price/annual fee
        When: parse_card_listing is called
        Then: The annual fee should be extracted as a number
        """
        # Given
        soup = BeautifulSoup(html_with_json_ld, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)
        chase_card = next(
            (c for c in cards if "Chase Sapphire" in c.get("name", "")), None
        )

        # Then
        assert chase_card is not None
        assert chase_card.get("annual_fee") == 95.0

    def test_parse_json_ld_extracts_rating(self, scraper, html_with_json_ld):
        """
        Given: HTML with JSON-LD containing aggregate rating
        When: parse_card_listing is called
        Then: Rating and review count should be extracted
        """
        # Given
        soup = BeautifulSoup(html_with_json_ld, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)
        chase_card = next(
            (c for c in cards if "Chase Sapphire" in c.get("name", "")), None
        )

        # Then
        assert chase_card is not None
        assert chase_card.get("rating") == 4.8
        assert chase_card.get("review_count") == 1250

    def test_parse_json_ld_sets_source(self, scraper, html_with_json_ld):
        """
        Given: HTML with JSON-LD card data
        When: parse_card_listing is called
        Then: Source should be set to "NerdWallet"
        """
        # Given
        soup = BeautifulSoup(html_with_json_ld, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            assert card.get("source") == "NerdWallet"

    def test_parse_json_ld_handles_string_items(
        self, scraper, html_with_json_ld_item_list
    ):
        """
        Given: HTML with JSON-LD itemListElement containing URL strings
        When: parse_card_listing is called
        Then: Should handle gracefully without errors (skip string items)
        """
        # Given
        soup = BeautifulSoup(html_with_json_ld_item_list, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        # Should only get the dict item (Amex Gold), not the string URL
        assert len(cards) == 1
        assert "Amex Gold" in cards[0].get("name", "")


# =============================================================================
# HTML Parsing Tests
# =============================================================================


class TestNerdWalletHtmlParsing:
    """Tests for HTML card element parsing."""

    def test_parse_html_extracts_card_names(self, scraper, html_with_card_products):
        """
        Given: HTML with card-product divs
        When: parse_card_listing is called
        Then: Card names should be extracted from headings
        """
        # Given
        soup = BeautifulSoup(html_with_card_products, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)
        card_names = [c.get("name", "") for c in cards]

        # Then
        assert any("Capital One Venture" in name for name in card_names)
        assert any("Citi Double Cash" in name for name in card_names)

    def test_parse_html_extracts_annual_fee_numeric(
        self, scraper, html_with_card_products
    ):
        """
        Given: HTML with "$95 annual fee" text
        When: parse_card_listing is called
        Then: Annual fee should be extracted as integer 95
        """
        # Given
        soup = BeautifulSoup(html_with_card_products, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)
        venture_card = next((c for c in cards if "Venture" in c.get("name", "")), None)

        # Then
        assert venture_card is not None
        assert venture_card.get("annual_fee") == 95

    def test_parse_html_extracts_no_annual_fee(self, scraper, html_with_no_fee_card):
        """
        Given: HTML with "$0 Annual Fee" text
        When: parse_card_listing is called
        Then: Annual fee should be 0
        """
        # Given
        soup = BeautifulSoup(html_with_no_fee_card, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        if cards:
            assert cards[0].get("annual_fee") == 0

    def test_parse_html_extracts_detail_url(self, scraper, html_with_card_products):
        """
        Given: HTML with card links
        When: parse_card_listing is called
        Then: Detail URLs should be extracted and made absolute
        """
        # Given
        soup = BeautifulSoup(html_with_card_products, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        cards_with_urls = [c for c in cards if c.get("detail_url")]
        assert len(cards_with_urls) > 0
        for card in cards_with_urls:
            assert card["detail_url"].startswith("http")


# =============================================================================
# Issuer Extraction Tests
# =============================================================================


class TestNerdWalletIssuerExtraction:
    """Tests for issuer extraction from card names."""

    def test_extract_issuer_chase(self, scraper):
        """
        Given: A card name containing "Chase"
        When: _extract_issuer is called
        Then: It should return "Chase"
        """
        # Given
        card_name = "Chase Sapphire Preferred Card"

        # When
        issuer = scraper._extract_issuer(card_name)

        # Then
        assert issuer == "Chase"

    def test_extract_issuer_amex(self, scraper):
        """
        Given: A card name containing "Amex"
        When: _extract_issuer is called
        Then: It should return "American Express"
        """
        # Given
        card_name = "Amex Gold Card"

        # When
        issuer = scraper._extract_issuer(card_name)

        # Then
        assert issuer == "American Express"

    def test_extract_issuer_american_express(self, scraper):
        """
        Given: A card name containing "American Express"
        When: _extract_issuer is called
        Then: It should return "American Express"
        """
        # Given
        card_name = "The Platinum Card from American Express"

        # When
        issuer = scraper._extract_issuer(card_name)

        # Then
        assert issuer == "American Express"

    def test_extract_issuer_citi(self, scraper):
        """
        Given: A card name containing "Citi"
        When: _extract_issuer is called
        Then: It should return "Citi"
        """
        # Given
        card_name = "Citi Double Cash Card"

        # When
        issuer = scraper._extract_issuer(card_name)

        # Then
        assert issuer == "Citi"

    def test_extract_issuer_capital_one(self, scraper):
        """
        Given: A card name containing "Capital One"
        When: _extract_issuer is called
        Then: It should return "Capital One"
        """
        # Given
        card_name = "Capital One Venture Rewards"

        # When
        issuer = scraper._extract_issuer(card_name)

        # Then
        assert issuer == "Capital One"

    def test_extract_issuer_discover(self, scraper):
        """
        Given: A card name containing "Discover"
        When: _extract_issuer is called
        Then: It should return "Discover"
        """
        # Given
        card_name = "Discover it Cash Back"

        # When
        issuer = scraper._extract_issuer(card_name)

        # Then
        assert issuer == "Discover"

    def test_extract_issuer_unknown(self, scraper):
        """
        Given: A card name with no known issuer
        When: _extract_issuer is called
        Then: It should return None
        """
        # Given
        card_name = "Mystery Rewards Card"

        # When
        issuer = scraper._extract_issuer(card_name)

        # Then
        assert issuer is None

    def test_extract_issuer_case_insensitive(self, scraper):
        """
        Given: A card name with issuer in different case
        When: _extract_issuer is called
        Then: It should still extract the issuer
        """
        # Given
        card_name = "CHASE freedom unlimited"

        # When
        issuer = scraper._extract_issuer(card_name)

        # Then
        assert issuer == "Chase"


# =============================================================================
# Price Parsing Tests
# =============================================================================


class TestNerdWalletPriceParsing:
    """Tests for price/fee parsing."""

    def test_parse_price_integer_string(self, scraper):
        """
        Given: A price string "95"
        When: _parse_price is called
        Then: It should return 95.0
        """
        # Given
        price_str = "95"

        # When
        result = scraper._parse_price(price_str)

        # Then
        assert result == 95.0

    def test_parse_price_with_dollar_sign(self, scraper):
        """
        Given: A price string "$95"
        When: _parse_price is called
        Then: It should return 95.0
        """
        # Given
        price_str = "$95"

        # When
        result = scraper._parse_price(price_str)

        # Then
        assert result == 95.0

    def test_parse_price_with_decimals(self, scraper):
        """
        Given: A price string "$550.00"
        When: _parse_price is called
        Then: It should return 550.0
        """
        # Given
        price_str = "$550.00"

        # When
        result = scraper._parse_price(price_str)

        # Then
        assert result == 550.0

    def test_parse_price_integer_input(self, scraper):
        """
        Given: An integer 95
        When: _parse_price is called
        Then: It should return 95.0
        """
        # Given
        price = 95

        # When
        result = scraper._parse_price(price)

        # Then
        assert result == 95.0

    def test_parse_price_none_input(self, scraper):
        """
        Given: None as input
        When: _parse_price is called
        Then: It should return None
        """
        # Given
        price = None

        # When
        result = scraper._parse_price(price)

        # Then
        assert result is None

    def test_parse_price_with_comma(self, scraper):
        """
        Given: A price string "$1,000"
        When: _parse_price is called
        Then: It should return 1000.0
        """
        # Given
        price_str = "$1,000"

        # When
        result = scraper._parse_price(price_str)

        # Then
        assert result == 1000.0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestNerdWalletEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_empty_html(self, scraper, html_empty):
        """
        Given: Empty HTML
        When: parse_card_listing is called
        Then: It should return an empty list
        """
        # Given
        soup = BeautifulSoup(html_empty, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        assert cards == []

    def test_parse_malformed_html(self, scraper, html_malformed):
        """
        Given: Malformed HTML
        When: parse_card_listing is called
        Then: It should not raise an exception
        """
        # Given
        soup = BeautifulSoup(html_malformed, "lxml")

        # When / Then
        cards = scraper.parse_card_listing(soup)
        assert isinstance(cards, list)

    def test_parse_card_with_special_characters(self, scraper):
        """
        Given: HTML with card name containing special characters
        When: parse_card_listing is called
        Then: Special characters should be preserved
        """
        # Given
        html = """
        <div class="card-product">
            <h2>Chase Sapphire Preferred® Card™</h2>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        if cards:
            assert "Chase Sapphire" in cards[0].get("name", "")

    def test_deduplication_between_json_ld_and_html(self, scraper):
        """
        Given: HTML with same card in both JSON-LD and HTML
        When: parse_card_listing is called
        Then: Card should not be duplicated
        """
        # Given
        html = """
        <html>
        <head>
            <script type="application/ld+json">
            {"@type": "Product", "name": "Test Card"}
            </script>
        </head>
        <body>
            <div class="card-product">
                <h2>Test Card</h2>
            </div>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        test_cards = [c for c in cards if c.get("name") == "Test Card"]
        assert len(test_cards) == 1


# =============================================================================
# Selenium Scraper Tests
# =============================================================================


class TestNerdWalletSeleniumScraper:
    """Tests for Selenium-based scraper."""

    def test_get_source_name(self):
        """
        Given: A NerdWalletSeleniumScraper instance
        When: get_source_name is called
        Then: It should return "NerdWallet (Selenium)"
        """
        # Given
        scraper = NerdWalletSeleniumScraper()

        # When
        name = scraper.get_source_name()

        # Then
        assert name == "NerdWallet (Selenium)"

    def test_headless_default_true(self):
        """
        Given: Default initialization
        When: NerdWalletSeleniumScraper is created
        Then: headless should be True
        """
        # Given / When
        scraper = NerdWalletSeleniumScraper()

        # Then
        assert scraper.headless is True

    def test_headless_can_be_disabled(self):
        """
        Given: headless=False
        When: NerdWalletSeleniumScraper is created
        Then: headless should be False
        """
        # Given / When
        scraper = NerdWalletSeleniumScraper(headless=False)

        # Then
        assert scraper.headless is False

    def test_uses_same_category_urls(self):
        """
        Given: A NerdWalletSeleniumScraper instance
        When: Checking CATEGORY_URLS
        Then: Should use same URLs as regular scraper
        """
        # Given
        selenium_scraper = NerdWalletSeleniumScraper()
        regular_scraper = NerdWalletScraper()

        # When / Then
        assert selenium_scraper.CATEGORY_URLS == regular_scraper.CATEGORY_URLS


# =============================================================================
# Data Validation Tests
# =============================================================================


class TestNerdWalletDataValidation:
    """Tests for data quality and schema validation."""

    def test_scraped_card_has_required_fields(self, scraper, html_with_json_ld):
        """
        Given: Scraped cards
        When: Checking fields
        Then: Required fields should be present
        """
        # Given
        soup = BeautifulSoup(html_with_json_ld, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        required_fields = ["source", "name"]
        for card in cards:
            for field in required_fields:
                assert field in card, f"Missing required field: {field}"

    def test_annual_fee_is_numeric_when_present(self, scraper, html_with_json_ld):
        """
        Given: Scraped cards with annual fees
        When: Checking annual fee values
        Then: Should be numeric (int or float)
        """
        # Given
        soup = BeautifulSoup(html_with_json_ld, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            if card.get("annual_fee") is not None:
                assert isinstance(card["annual_fee"], (int, float))
                assert card["annual_fee"] >= 0

    def test_urls_are_valid_when_present(self, scraper, html_with_card_products):
        """
        Given: Scraped cards with URLs
        When: Checking URL values
        Then: Should be properly formatted URLs
        """
        # Given
        soup = BeautifulSoup(html_with_card_products, "lxml")

        # When
        cards = scraper.parse_card_listing(soup)

        # Then
        for card in cards:
            if card.get("detail_url"):
                assert card["detail_url"].startswith("http")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
