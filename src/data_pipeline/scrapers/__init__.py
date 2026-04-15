"""
RewardSense - Credit Card Scrapers Module

This module provides scrapers for collecting credit card data from
various sources including aggregators (NerdWallet) and direct issuers
(Chase, Amex, Citi, etc.).

Usage:
    from data_pipeline.scrapers import NerdWalletScraper, ChaseScraper

    # Scrape NerdWallet
    with NerdWalletScraper(rate_limit=2.0) as scraper:
        cards = scraper.scrape_all_cards()

    # Scrape Chase
    with ChaseScraper() as scraper:
        chase_cards = scraper.scrape_all_cards()
"""

from typing import Optional
from .base_scraper import BaseScraper
from .nerdwallet_scraper import NerdWalletScraper, NerdWalletSeleniumScraper
from .issuer_scrapers import (
    ChaseScraper,
    AmexScraper,
    CitiScraper,
    CapitalOneScraper,
    DiscoverScraper,
)

__all__ = [
    "BaseScraper",
    "NerdWalletScraper",
    "NerdWalletSeleniumScraper",
    "ChaseScraper",
    "AmexScraper",
    "CitiScraper",
    "CapitalOneScraper",
    "DiscoverScraper",
]


def get_scraper(source_name: str, **kwargs):
    """
    Factory function to get a scraper by source name.

    Args:
        source_name: Name of the source (e.g., 'nerdwallet', 'chase')
        **kwargs: Arguments to pass to the scraper

    Returns:
        Scraper instance

    Raises:
        ValueError: If source name is not recognized
    """
    scrapers = {
        "nerdwallet": NerdWalletScraper,
        "nerdwallet_selenium": NerdWalletSeleniumScraper,
        "chase": ChaseScraper,
        "amex": AmexScraper,
        "american_express": AmexScraper,
        "citi": CitiScraper,
        "capital_one": CapitalOneScraper,
        "capitalone": CapitalOneScraper,
        "discover": DiscoverScraper,
    }

    source_lower = source_name.lower().replace(" ", "_").replace("-", "_")

    if source_lower not in scrapers:
        raise ValueError(
            f"Unknown source: {source_name}. "
            f"Available sources: {list(scrapers.keys())}"
        )

    return scrapers[source_lower](**kwargs)


def scrape_all_sources(config_path: Optional[str] = None) -> dict:
    """
    Scrape all enabled sources based on configuration.

    Args:
        config_path: Path to scraper_config.yaml (optional)

    Returns:
        Dictionary mapping source names to lists of card data
    """
    import yaml
    from pathlib import Path

    # Load configuration
    resolved_path: Path
    if config_path is None:
        resolved_path = Path(__file__).parent / "scraper_config.yaml"
    else:
        resolved_path = Path(config_path)

    with open(resolved_path) as f:
        config = yaml.safe_load(f)

    global_settings = config.get("global", {})
    sources = config.get("sources", {})

    all_cards = {}

    for source_name, source_config in sources.items():
        if not source_config.get("enabled", False):
            continue

        try:
            # Get appropriate scraper
            if source_name == "nerdwallet" and source_config.get("use_selenium"):
                scraper_class = NerdWalletSeleniumScraper
            else:
                scraper_class = get_scraper(source_name).__class__

            # Initialize with global settings
            with scraper_class(
                rate_limit=global_settings.get("rate_limit", 1.0),
                max_retries=global_settings.get("max_retries", 3),
                timeout=global_settings.get("timeout", 30),
                user_agent=global_settings.get("user_agent"),
            ) as scraper:
                cards = scraper.scrape_all_cards()
                all_cards[source_name] = cards

        except Exception as e:
            print(f"Error scraping {source_name}: {e}")
            all_cards[source_name] = []

    return all_cards
