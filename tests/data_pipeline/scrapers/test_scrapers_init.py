"""
RewardSense - Unit Tests for Scrapers Module Init

Tests for:
- Factory function (get_scraper)
- scrape_all_sources function
- Module exports

Run with: pytest tests/test_scrapers_init.py -v
"""

import pytest
from unittest.mock import patch

import sys

sys.path.insert(0, "src")

from data_pipeline.scrapers import (
    get_scraper,
    BaseScraper,
    NerdWalletScraper,
    NerdWalletSeleniumScraper,
    ChaseScraper,
    AmexScraper,
    CitiScraper,
    CapitalOneScraper,
    DiscoverScraper,
)

# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetScraperFactory:
    """Tests for the get_scraper factory function."""

    def test_get_scraper_nerdwallet(self):
        """
        Given: Source name "nerdwallet"
        When: get_scraper is called
        Then: It should return a NerdWalletScraper instance
        """
        # Given
        source = "nerdwallet"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, NerdWalletScraper)

    def test_get_scraper_nerdwallet_selenium(self):
        """
        Given: Source name "nerdwallet_selenium"
        When: get_scraper is called
        Then: It should return a NerdWalletSeleniumScraper instance
        """
        # Given
        source = "nerdwallet_selenium"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, NerdWalletSeleniumScraper)

    def test_get_scraper_chase(self):
        """
        Given: Source name "chase"
        When: get_scraper is called
        Then: It should return a ChaseScraper instance
        """
        # Given
        source = "chase"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, ChaseScraper)

    def test_get_scraper_amex(self):
        """
        Given: Source name "amex"
        When: get_scraper is called
        Then: It should return an AmexScraper instance
        """
        # Given
        source = "amex"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, AmexScraper)

    def test_get_scraper_american_express(self):
        """
        Given: Source name "american_express"
        When: get_scraper is called
        Then: It should return an AmexScraper instance
        """
        # Given
        source = "american_express"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, AmexScraper)

    def test_get_scraper_citi(self):
        """
        Given: Source name "citi"
        When: get_scraper is called
        Then: It should return a CitiScraper instance
        """
        # Given
        source = "citi"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, CitiScraper)

    def test_get_scraper_capital_one(self):
        """
        Given: Source name "capital_one"
        When: get_scraper is called
        Then: It should return a CapitalOneScraper instance
        """
        # Given
        source = "capital_one"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, CapitalOneScraper)

    def test_get_scraper_capitalone_no_underscore(self):
        """
        Given: Source name "capitalone" (no underscore)
        When: get_scraper is called
        Then: It should return a CapitalOneScraper instance
        """
        # Given
        source = "capitalone"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, CapitalOneScraper)

    def test_get_scraper_discover(self):
        """
        Given: Source name "discover"
        When: get_scraper is called
        Then: It should return a DiscoverScraper instance
        """
        # Given
        source = "discover"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, DiscoverScraper)

    def test_get_scraper_invalid_source_raises_error(self):
        """
        Given: An invalid source name
        When: get_scraper is called
        Then: It should raise ValueError
        """
        # Given
        source = "invalid_source"

        # When / Then
        with pytest.raises(ValueError) as exc_info:
            get_scraper(source)

        assert "Unknown source" in str(exc_info.value)

    def test_get_scraper_error_lists_available_sources(self):
        """
        Given: An invalid source name
        When: get_scraper is called
        Then: Error message should list available sources
        """
        # Given
        source = "invalid"

        # When / Then
        with pytest.raises(ValueError) as exc_info:
            get_scraper(source)

        error_msg = str(exc_info.value)
        assert "nerdwallet" in error_msg or "Available sources" in error_msg

    def test_get_scraper_case_insensitive(self):
        """
        Given: Source name in uppercase "CHASE"
        When: get_scraper is called
        Then: It should still return a ChaseScraper
        """
        # Given
        source = "CHASE"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, ChaseScraper)

    def test_get_scraper_with_spaces(self):
        """
        Given: Source name "Capital One" with space
        When: get_scraper is called
        Then: It should return a CapitalOneScraper
        """
        # Given
        source = "Capital One"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, CapitalOneScraper)

    def test_get_scraper_with_hyphen(self):
        """
        Given: Source name "capital-one" with hyphen
        When: get_scraper is called
        Then: It should return a CapitalOneScraper
        """
        # Given
        source = "capital-one"

        # When
        scraper = get_scraper(source)

        # Then
        assert isinstance(scraper, CapitalOneScraper)

    def test_get_scraper_passes_kwargs(self):
        """
        Given: Source name and custom rate_limit
        When: get_scraper is called with rate_limit=5.0
        Then: The scraper should have rate_limit=5.0
        """
        # Given
        source = "nerdwallet"

        # When
        scraper = get_scraper(source, rate_limit=5.0)

        # Then
        assert scraper.rate_limit == 5.0


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_basescraper_is_exported(self):
        """
        Given: The scrapers module
        When: Importing BaseScraper
        Then: It should be available
        """
        # Given / When / Then
        assert BaseScraper is not None

    def test_nerdwalletscraper_is_exported(self):
        """
        Given: The scrapers module
        When: Importing NerdWalletScraper
        Then: It should be available
        """
        # Given / When / Then
        assert NerdWalletScraper is not None

    def test_all_issuer_scrapers_are_exported(self):
        """
        Given: The scrapers module
        When: Checking exports
        Then: All issuer scrapers should be available
        """
        # Given / When / Then
        assert ChaseScraper is not None
        assert AmexScraper is not None
        assert CitiScraper is not None
        assert CapitalOneScraper is not None
        assert DiscoverScraper is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestScraperIntegration:
    """Integration tests for scraper module."""

    def test_all_scrapers_inherit_from_base(self):
        """
        Given: All scraper classes
        When: Checking inheritance
        Then: All should inherit from BaseScraper
        """
        # Given
        scrapers = [
            NerdWalletScraper,
            NerdWalletSeleniumScraper,
            ChaseScraper,
            AmexScraper,
            CitiScraper,
            CapitalOneScraper,
            DiscoverScraper,
        ]

        # When / Then
        for scraper_class in scrapers:
            assert issubclass(
                scraper_class, BaseScraper
            ), f"{scraper_class.__name__} does not inherit from BaseScraper"

    def test_all_scrapers_implement_required_methods(self):
        """
        Given: All scraper instances
        When: Checking for required methods
        Then: All should have the required abstract methods
        """
        # Given
        scrapers = [
            NerdWalletScraper(),
            ChaseScraper(),
            AmexScraper(),
            CitiScraper(),
            CapitalOneScraper(),
            DiscoverScraper(),
        ]
        required_methods = [
            "get_source_name",
            "get_card_list_urls",
            "parse_card_listing",
            "parse_card_details",
        ]

        # When / Then
        for scraper in scrapers:
            for method in required_methods:
                assert hasattr(
                    scraper, method
                ), f"{scraper.get_source_name()} missing method: {method}"
                assert callable(
                    getattr(scraper, method)
                ), f"{scraper.get_source_name()}.{method} is not callable"

    def test_all_scrapers_return_string_source_name(self):
        """
        Given: All scraper instances
        When: Calling get_source_name
        Then: All should return a non-empty string
        """
        # Given
        scrapers = [
            NerdWalletScraper(),
            ChaseScraper(),
            AmexScraper(),
            CitiScraper(),
            CapitalOneScraper(),
            DiscoverScraper(),
        ]

        # When / Then
        for scraper in scrapers:
            name = scraper.get_source_name()
            assert isinstance(
                name, str
            ), f"{scraper.__class__.__name__} source name is not a string"
            assert len(name) > 0, f"{scraper.__class__.__name__} source name is empty"

    def test_all_scrapers_can_be_used_as_context_manager(self):
        """
        Given: All scraper classes
        When: Using as context manager
        Then: All should work without error
        """
        # Given
        scraper_classes = [
            NerdWalletScraper,
            ChaseScraper,
            AmexScraper,
            CitiScraper,
            CapitalOneScraper,
            DiscoverScraper,
        ]

        # When / Then
        for scraper_class in scraper_classes:
            with scraper_class() as scraper:
                assert scraper is not None


class TestScrapeAllSources:
    """Tests for the scrape_all_sources function."""

    @patch("data_pipeline.scrapers.NerdWalletScraper")
    def test_scrape_all_sources_returns_dict(self, mock_scraper_class):
        """
        Given: A valid config file
        When: scrape_all_sources is called
        Then: It should return a dictionary
        """
        # This test would require a proper config file
        # Skipping detailed implementation for now
        pass

    def test_scrape_all_sources_handles_disabled_sources(self):
        """
        Given: A config with disabled sources
        When: scrape_all_sources is called
        Then: Disabled sources should not be scraped
        """
        # This test would require a proper config file
        # Skipping detailed implementation for now
        pass


class TestScrapeAllSourcesDetailed:
    """Detailed tests for scrape_all_sources function."""

    def test_scrape_all_sources_with_mock_config(self, tmp_path):
        """
        Given: A valid config file
        When: scrape_all_sources is called
        Then: It should return a dictionary
        """
        # Given
        config_content = """
            global:
                rate_limit: 0.1
                max_retries: 1
                timeout: 5

            sources:
                nerdwallet:
                    enabled: false
                chase:
                    enabled: false
            """
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        # When
        from data_pipeline.scrapers import scrape_all_sources

        result = scrape_all_sources(str(config_file))

        # Then
        assert isinstance(result, dict)

    def test_scrape_all_sources_handles_scraper_error(self, tmp_path):
        """
        Given: A config that causes scraper to fail
        When: scrape_all_sources is called
        Then: It should handle the error gracefully
        """
        # Given
        config_content = """
        global:
            rate_limit: 0.1
            max_retries: 1
            timeout: 1

        sources:
            nerdwallet:
                enabled: true
                use_selenium: false
        categories:
            - invalid_category
        """
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        # When
        from data_pipeline.scrapers import scrape_all_sources

        result = scrape_all_sources(str(config_file))

        # Then
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
