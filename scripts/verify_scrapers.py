#!/usr/bin/env python3
"""
Verification script for all issuer scrapers.
Fetches card listings and prints the first card found for each scraper.
"""
import sys
import logging
from bs4 import BeautifulSoup

# Add src to path if needed (assuming run from project root)
sys.path.append("src")

from data_pipeline.scrapers.issuer_scrapers import (
    ChaseScraper,
    DiscoverScraper,
    AmexScraper,
    CitiScraper,
    CapitalOneScraper,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verifier")

def verify_scraper(scraper_cls):
    """Run verification for a single scraper class."""
    scraper_name = scraper_cls.__name__
    print(f"\n{'='*60}")
    print(f"Verifying {scraper_name}...")
    print(f"{'='*60}")
    
    try:
        scraper = scraper_cls()
        source_name = scraper.get_source_name()
        print(f"Source: {source_name}")
        
        # Test 1: Get URLs
        urls = scraper.get_card_list_urls()
        print(f"Found {len(urls)} listing URLs")
        if not urls:
            print(f"❌ No URLs found for {scraper_name}")
            return False

        # Test 2: Fetch and parse first URL (using real methods)
        url = urls[0]
        print(f"Fetching: {url}")
        
        # For non-Selenium scrapers, fetch manually if needed, but our implementation
        # of parse_card_listing expects 'soup', except Selenium ones handle their own fetching internally
        # wait, looking at implementation:
        # Chase/Discover: expects soup object passed in.
        # Amex/Citi/CapitalOne: fetches URLs internally inside parse_card_listing, ignores passed soup.
        
        cards = []
        if scraper_name in ["AmexScraper", "CitiScraper", "CapitalOneScraper"]:
            # Selenium scrapers handle their own fetching
            # We pass a dummy soup
            print("Running Selenium-based scraping (this may take a moment)...")
            cards = scraper.parse_card_listing(BeautifulSoup("", "lxml"))
            
        else:
            # Requests-based scrapers need fetched soup
            # BaseScraper has fetch_page method
            print("Fetching page via requests...")
            soup = scraper.fetch_page(url)
            if not soup:
                 print(f"❌ Failed to fetch page for {scraper_name}")
                 return False
            cards = scraper.parse_card_listing(soup)
            
        print(f"Found {len(cards)} cards total.")
        
        if len(cards) > 0:
            print("\n✅ SUCCESS: Retrieved at least one card.")
            first_card = cards[0]
            print(f"First Card Sample:\n{first_card}")
            return True
        else:
            print(f"⚠️  WARNING: No cards found for {scraper_name}. This could be due to layout changes or bot protection.")
            return False

    except Exception as e:
        print(f"❌ ERROR running {scraper_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    scrapers = [
        # Start with simple request-based scrapers
        ChaseScraper,
        DiscoverScraper,
        # Then complex selenium-based ones
        AmexScraper,
        CitiScraper,
        CapitalOneScraper,
    ]
    
    results = {}
    print("Starting verification of all issuer scrapers.\n")
    
    for scraper_cls in scrapers:
        success = verify_scraper(scraper_cls)
        results[scraper_cls.__name__] = "✅ PASS" if success else "❌ FAIL/WARN"
        
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{name:<25}: {status}")

if __name__ == "__main__":
    main()
