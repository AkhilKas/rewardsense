"""
RewardSense - Credit Card Issuer Scrapers

Direct scrapers for major credit card issuers.

Status:
    ✅ Chase - Working (BeautifulSoup)
    ✅ Discover - Working (BeautifulSoup)
    ✅ Amex - Working (Selenium)
    ✅ Citi - Working (Selenium)
    ✅ Capital One - Working (Selenium)
"""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime


from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class ChaseScraper(BaseScraper):
    """
    Scraper for Chase credit cards.

    Updated 2026-02 with correct selectors for Chase's current HTML structure.
    Uses 'cmp-cardsummary__inner-container' as the main card container.
    """

    BASE_URL = "https://creditcards.chase.com"

    CARD_URLS = {
        "all_cards": "/all-credit-cards",
    }

    def get_source_name(self) -> str:
        return "Chase"

    def get_card_list_urls(self) -> List[str]:
        return [f"{self.BASE_URL}{path}" for path in self.CARD_URLS.values()]

    def parse_card_listing(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse Chase credit card listing page."""
        cards = []

        # Find all card title divs, then get their parent containers
        title_divs = soup.find_all(
            "div", class_="cmp-cardsummary__inner-container__title"
        )
        logger.info(f"Found {len(title_divs)} card containers on Chase")

        for title_div in title_divs:
            card = self._parse_chase_card(title_div)
            if card and card.get("name"):
                cards.append(card)

        return cards

    def _parse_chase_card(self, title_div) -> Optional[Dict[str, Any]]:
        """Parse a single Chase card from its title div."""

        # Get the parent container with all card details
        container = title_div.find_parent(
            "div", class_="cmp-cardsummary__inner-container"
        )
        if not container:
            return None

        card: Dict[str, Any] = {
            "source": "Chase",
            "issuer": "Chase",
            "scraped_at": datetime.now().isoformat(),
        }

        # Extract card name from h2
        h2 = title_div.find("h2")
        if not h2:
            return None

        name = h2.get_text(strip=True)
        # Clean up the name - remove common suffixes
        name = re.sub(r"Credit Card.*", "", name, flags=re.IGNORECASE)
        name = re.sub(r"Card\s*Links to.*", "", name, flags=re.IGNORECASE)
        name = re.sub(r"Links to.*", "", name, flags=re.IGNORECASE)
        # Remove trademark symbols but keep the name clean
        name = name.replace("®", "").replace("℠", "").replace("™", "")
        name = re.sub(r"\s+", " ", name).strip()
        card["name"] = name

        # Annual fee
        fee_div = container.find(
            "div", class_="cmp-cardsummary__inner-container--annual-fee"
        )
        if fee_div:
            fee_text = fee_div.get_text()
            fee_match = re.search(r"\$(\d+)", fee_text)
            card["annual_fee"] = int(fee_match.group(1)) if fee_match else 0
        else:
            card["annual_fee"] = 0

        # Welcome bonus / card member offer
        offer_div = container.find(
            "div", class_="cmp-cardsummary__inner-container--card-member-offer"
        )
        if offer_div:
            offer_text = offer_div.get_text()

            # Try to find points/miles bonus
            points_match = re.search(
                r"[Ee]arn\s+(\d{1,3},?\d{3})\s*(bonus\s+)?(points?|miles?)", offer_text
            )
            if points_match:
                points_str = points_match.group(1).replace(",", "")
                card["welcome_bonus"] = (
                    f"{points_match.group(1)} {points_match.group(3)}"
                )
                card["bonus_value_usd"] = self._estimate_points_value(
                    int(points_str), "Chase"
                )
            else:
                # Try cash bonus
                cash_match = re.search(r"\$(\d+)\s*bonus", offer_text, re.I)
                if cash_match:
                    card["welcome_bonus"] = f"${cash_match.group(1)} bonus"
                    card["bonus_value_usd"] = int(cash_match.group(1))

        # Try to extract reward rates from the full container text
        full_text = container.get_text()
        card["reward_rates"] = self._extract_reward_rates(full_text)

        # Get the detail URL
        link = container.find("a", href=True)
        if link:
            href = link.get("href", "")
            if href.startswith("/"):
                card["detail_url"] = f"{self.BASE_URL}{href}"
            elif href.startswith("http"):
                card["detail_url"] = href

        # Card image
        img_div = container.find(
            "div", class_="cmp-cardsummary__inner-container__image"
        )
        if img_div:
            img = img_div.find("img")
            if img and img.get("src"):
                card["image_url"] = img["src"]

        return card

    def _extract_reward_rates(self, text: str) -> Dict[str, Any]:
        """Extract reward rates from card description text."""
        rates = {}

        # Common patterns for Chase cards
        patterns = [
            # "5% cash back on travel"
            (
                r"(\d+(?:\.\d+)?)\s*%\s*(?:cash\s*back|back)\s+(?:on\s+)?([a-zA-Z\s,&]+?)(?:\.|,|and\s+\d|\s+\d)",
                "cashback",
            ),
            # "3X points on dining"
            (
                r"(\d+)[xX]\s*(?:points?\s+)?(?:on\s+)?([a-zA-Z\s,&]+?)(?:\.|,|and\s+\d|\s+\d)",
                "points",
            ),
            # "Earn 3X on dining"
            (
                r"[Ee]arn\s+(\d+)[xX]\s+(?:on\s+)?([a-zA-Z\s,&]+?)(?:\.|,|and\s+\d|\s+\d)",
                "points",
            ),
        ]

        for pattern, rate_type in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                rate, category = match
                category = category.strip().lower()
                # Skip if category is too long (probably not a real category)
                if len(category) > 30:
                    continue
                # Clean up category
                category = re.sub(r"\s+", " ", category).strip()
                if category:
                    rates[category] = {
                        "rate": float(rate) if "." in rate else int(rate),
                        "type": rate_type,
                    }

        return rates

    def _estimate_points_value(self, points: int, issuer: str) -> float:
        """Estimate USD value of points/miles."""
        # Chase Ultimate Rewards typically valued at ~1.5-2 cents per point
        # Using conservative 1.5 cents
        cpp = 0.015
        return round(points * cpp, 2)

    def parse_card_details(self, card_url: str) -> Optional[Dict[str, Any]]:
        """Parse detailed Chase card page."""
        soup = self.fetch_page(card_url)
        if not soup:
            return None

        details = {
            "detail_url": card_url,
            "reward_categories": [],
            "benefits": [],
        }

        return details


class DiscoverScraper(BaseScraper):
    """
    Scraper for Discover credit cards.

    Discover has a simpler card lineup and their site works with BeautifulSoup.
    Includes deduplication logic since same cards may appear multiple times.
    """

    BASE_URL = "https://www.discover.com"

    CARD_URLS = {
        "all_cards": "/credit-cards/",
    }

    def get_source_name(self) -> str:
        return "Discover"

    def get_card_list_urls(self) -> List[str]:
        return [f"{self.BASE_URL}{path}" for path in self.CARD_URLS.values()]

    def parse_card_listing(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse Discover card listing with deduplication."""
        cards = []
        seen_names = set()  # For deduplication

        # Try to find card containers
        card_elements = soup.find_all("div", class_=re.compile(r"card", re.I))

        for element in card_elements:
            card = self._parse_discover_card(element)
            if card and card.get("name"):
                # Deduplicate by normalized name
                normalized_name = self._normalize_name(card["name"])
                if normalized_name not in seen_names:
                    seen_names.add(normalized_name)
                    cards.append(card)
                else:
                    logger.debug(f"Skipping duplicate: {card['name']}")

        logger.info(f"Found {len(cards)} unique Discover cards (after dedup)")
        return cards

    def _parse_discover_card(self, element) -> Optional[Dict[str, Any]]:
        """Parse a single Discover card element."""
        card: Dict[str, Any] = {
            "source": "Discover",
            "issuer": "Discover",
            "scraped_at": datetime.now().isoformat(),
        }

        # Find card name
        name_elem = element.find(["h2", "h3", "h4"])
        if not name_elem:
            return None

        name = name_elem.get_text(strip=True)

        # Only include if it looks like a Discover credit card
        # Must contain "discover" or common Discover card keywords
        name_lower = name.lower()
        is_discover_card = "discover" in name_lower or (
            "it®" in name_lower
            and any(kw in name_lower for kw in ["cash", "miles", "chrome"])
        )
        if not is_discover_card:
            return None

        # Clean up name
        name = name.replace("®", "").replace("™", "")
        name = re.sub(r"\s+", " ", name).strip()
        card["name"] = name

        # Try to extract annual fee
        text = element.get_text()
        if "no annual fee" in text.lower() or "$0 annual fee" in text.lower():
            card["annual_fee"] = 0
        else:
            fee_match = re.search(r"\$(\d+)\s*annual", text, re.I)
            if fee_match:
                card["annual_fee"] = int(fee_match.group(1))
            else:
                # Discover cards typically have no annual fee
                card["annual_fee"] = 0

        # Try to extract rewards info
        rewards_match = re.search(r"(\d+)%\s*cash\s*back", text, re.I)
        if rewards_match:
            card["reward_rates"] = {"cash back": f"{rewards_match.group(1)}%"}

        # Detail URL
        link = element.find("a", href=True)
        if link:
            href = link.get("href", "")
            if href.startswith("/"):
                card["detail_url"] = f"{self.BASE_URL}{href}"
            elif href.startswith("http"):
                card["detail_url"] = href

        return card

    def _normalize_name(self, name: str) -> str:
        """Normalize card name for deduplication comparison."""
        # Lowercase, remove special chars, collapse whitespace
        normalized = name.lower()
        normalized = re.sub(r"[®™©]", "", normalized)
        normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def parse_card_details(self, card_url: str) -> Optional[Dict[str, Any]]:
        return None


# =============================================================================
# TODO: Selenium-based scrapers for JS-rendered sites
# =============================================================================


class AmexScraper(BaseScraper):
    """
    Scraper for American Express credit cards.

    TODO: Implement with Selenium - Amex uses heavy JavaScript rendering.
          The page body is empty when fetched with requests/BeautifulSoup.

    Implementation notes:
        - Requires Selenium WebDriver (Chrome/Firefox)
        - Need to wait for JS to render card tiles
        - Card containers likely in data-testid or specific class patterns
        - Consider using explicit waits for card elements to load
    """

    BASE_URL = "https://www.americanexpress.com"

    CARD_URLS = {
        "all_cards": "/us/credit-cards/",
    }

    def get_source_name(self) -> str:
        return "American Express"

    def get_card_list_urls(self) -> List[str]:
        return [f"{self.BASE_URL}{path}" for path in self.CARD_URLS.values()]

    def parse_card_listing(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Parse Amex credit card listing page using Selenium.

        Amex uses a React/JS-rendered page. Card tiles use CSS module classes
        like '_cardTileContainer_*' with card names in h2._cardTileCardNameTitle_*.
        """
        if not SELENIUM_AVAILABLE:
            logger.warning(
                "Selenium is not installed. Amex scraper requires Selenium. "
                "Install with: pip install selenium"
            )
            return []

        cards = []

        # Configure headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )

        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)

            for url in self.get_card_list_urls():
                logger.info(f"Navigating to {url}")
                driver.get(url)

                # Wait for card tile containers to render (CSS module class)
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "h2[class*='cardTileCardNameTitle']")
                        )
                    )
                except Exception as e:
                    logger.warning(f"Timeout waiting for cards on {url}: {e}")

                # Get the rendered HTML
                html = driver.page_source
                rendered_soup = BeautifulSoup(html, "lxml")

                # Find card tile containers using CSS module class pattern
                card_containers = rendered_soup.find_all(
                    "div", class_=re.compile(r"_cardTileContainer_")
                )
                logger.info(f"Found {len(card_containers)} Amex card tile containers")

                for container in card_containers:
                    card = self._parse_amex_card(container)
                    if card:
                        cards.append(card)

        except Exception as e:
            logger.error(f"Selenium error in AmexScraper: {e}")
        finally:
            if driver:
                driver.quit()

        return cards

    def _parse_amex_card(self, container) -> Optional[Dict[str, Any]]:
        """Parse a single Amex card container from rendered HTML."""
        try:
            card: Dict[str, Any] = {
                "source": "American Express",
                "issuer": "American Express",
                "scraped_at": datetime.now().isoformat(),
            }

            # Extract name from h2 with CSS module class _cardTileCardNameTitle_
            name_elem = container.find(
                "h2", class_=re.compile(r"_cardTileCardNameTitle_")
            )
            if not name_elem:
                return None

            name = name_elem.get_text(strip=True)
            # Cleanup name
            name = name.replace("®", "").replace("℠", "").replace("™", "")
            card["name"] = name.strip()

            # Extract annual fee from text like "Annual Fee: $895"
            full_text = container.get_text()
            fee_match = re.search(r"Annual\s*Fee:\s*\$([0-9,]+)", full_text, re.I)
            if fee_match:
                card["annual_fee"] = int(fee_match.group(1).replace(",", ""))
            elif "no annual fee" in full_text.lower() or "$0" in full_text:
                card["annual_fee"] = 0
            else:
                card["annual_fee"] = 0

            # Extract welcome bonus - look for points amounts before "Membership Rewards"
            bonus_match = re.search(r"([\d,]+)\s*Membership Rewards", full_text, re.I)
            if bonus_match:
                points_str = bonus_match.group(1)
                card["welcome_bonus"] = f"{points_str} Membership Rewards points"
            else:
                # Try cash back pattern
                cash_match = re.search(
                    r"\$([\d,]+)\s*(?:cash\s*back|bonus)", full_text, re.I
                )
                if cash_match:
                    card["welcome_bonus"] = f"${cash_match.group(1)} cash back"

            return card

        except Exception as e:
            logger.warning(f"Error parsing Amex card: {e}")
            return None

    def parse_card_details(self, card_url: str) -> Optional[Dict[str, Any]]:
        return None


class CitiScraper(BaseScraper):
    """
    Scraper for Citi credit cards.

    Uses Selenium to scrape the "View All Cards" page, which renders
    card tiles with h3.card-name elements inside content-container divs.
    """

    BASE_URL = "https://www.citi.com"

    CARD_URLS = {
        "all_cards": "/credit-cards/view-all-credit-cards",
    }

    def get_source_name(self) -> str:
        return "Citi"

    def get_card_list_urls(self) -> List[str]:
        return [f"{self.BASE_URL}{path}" for path in self.CARD_URLS.values()]

    def parse_card_listing(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Parse Citi credit card listing page using Selenium.

        Citi renders cards with h3.card-name elements inside content-container
        divs. The "View All Cards" page lists all available credit cards.
        """
        if not SELENIUM_AVAILABLE:
            logger.warning(
                "Selenium is not installed. Citi scraper requires Selenium. "
                "Install with: pip install selenium"
            )
            return []

        cards = []

        # Configure headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )

        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)

            for url in self.get_card_list_urls():
                logger.info(f"Navigating to {url}")
                driver.get(url)

                # Wait for card name elements to load
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "h3.card-name")
                        )
                    )
                except Exception:
                    logger.warning(f"Timeout waiting for cards on {url}")

                # Get the rendered HTML
                html = driver.page_source
                rendered_soup = BeautifulSoup(html, "lxml")

                # Find all card name h3 elements
                card_name_elements = rendered_soup.find_all("h3", class_="card-name")
                logger.info(f"Found {len(card_name_elements)} Citi card-name elements")

                for name_elem in card_name_elements:
                    # Walk up to find container with fee info
                    container = name_elem.find_parent("div", class_="content-container")
                    if not container:
                        container = name_elem.parent
                    card = self._parse_citi_card(name_elem, container)
                    if card:
                        cards.append(card)

        except Exception as e:
            logger.error(f"Selenium error in CitiScraper: {e}")
        finally:
            if driver:
                driver.quit()

        return cards

    def _parse_citi_card(self, name_elem, container) -> Optional[Dict[str, Any]]:
        """Parse a single Citi card from its name element and container."""
        try:
            card: Dict[str, Any] = {
                "source": "Citi",
                "issuer": "Citi",
                "scraped_at": datetime.now().isoformat(),
            }

            # Extract name from h3.card-name
            name = name_elem.get_text(strip=True)
            if not name:
                return None
            name = name.replace("®", "").replace("℠", "").replace("™", "")
            card["name"] = name.strip()

            # Extract annual fee from container text
            if container:
                text = container.get_text()
                fee_match = re.search(r"\$([\d,]+)\s*[Aa]nnual\s*[Ff]ee", text)
                if fee_match:
                    card["annual_fee"] = int(fee_match.group(1).replace(",", ""))
                elif "no annual fee" in text.lower() or "$0 annual fee" in text.lower():
                    card["annual_fee"] = 0
                else:
                    card["annual_fee"] = 0

                # Extract welcome bonus
                bonus_match = re.search(
                    r"(\d[\d,]*)\s*(?:bonus\s*)?(?:miles|points|ThankYou)",
                    text,
                    re.I,
                )
                if bonus_match:
                    card["welcome_bonus"] = bonus_match.group(0).strip()
                else:
                    cash_match = re.search(r"\$(\d[\d,]*)\s*(?:cash|bonus)", text, re.I)
                    if cash_match:
                        card["welcome_bonus"] = f"${cash_match.group(1)} bonus"

            return card
        except Exception:
            return None

    def parse_card_details(self, card_url: str) -> Optional[Dict[str, Any]]:
        return None


class CapitalOneScraper(BaseScraper):
    """
    Scraper for Capital One credit cards.

    Uses Selenium to scrape the card listing page. Capital One uses an Angular
    app with card-details-container divs containing button.heading elements
    for card names and attribute-list divs for fee/reward details.
    """

    BASE_URL = "https://www.capitalone.com"

    CARD_URLS = {
        "all_cards": "/credit-cards/",
    }

    def get_source_name(self) -> str:
        return "Capital One"

    def get_card_list_urls(self) -> List[str]:
        return [f"{self.BASE_URL}{path}" for path in self.CARD_URLS.values()]

    def parse_card_listing(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Parse Capital One card listing page using Selenium.

        Capital One uses Angular with card-details-container divs.
        Card names are in button.heading elements, fees and rewards
        are in attribute-list divs.
        """
        if not SELENIUM_AVAILABLE:
            logger.warning(
                "Selenium is not installed. Capital One scraper requires Selenium. "
                "Install with: pip install selenium"
            )
            return []

        cards = []

        # Configure headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )

        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)

            for url in self.get_card_list_urls():
                logger.info(f"Navigating to {url}")
                driver.get(url)

                # Wait for card-details-container elements to render
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.card-details-container")
                        )
                    )
                except Exception:
                    logger.warning(f"Timeout waiting for cards on {url}")

                # Get rendered HTML
                html = driver.page_source
                rendered_soup = BeautifulSoup(html, "lxml")

                # Find card containers
                card_containers = rendered_soup.find_all(
                    "div", class_="card-details-container"
                )
                logger.info(f"Found {len(card_containers)} Capital One card containers")

                seen_names = set()
                for container in card_containers:
                    card = self._parse_capone_card(container)
                    if card and card["name"] not in seen_names:
                        seen_names.add(card["name"])
                        cards.append(card)

        except Exception as e:
            logger.error(f"Selenium error in CapitalOneScraper: {e}")
        finally:
            if driver:
                driver.quit()

        return cards

    def _parse_capone_card(self, container) -> Optional[Dict[str, Any]]:
        """Parse a single Capital One card container."""
        try:
            card: Dict[str, Any] = {
                "source": "Capital One",
                "issuer": "Capital One",
                "scraped_at": datetime.now().isoformat(),
            }

            # Extract card name from button.heading element
            name_elem = container.find("button", class_=re.compile(r"heading"))
            if not name_elem:
                # Fallback to any heading element
                name_elem = container.find(["h2", "h3", "h4"])
            if not name_elem:
                return None

            name = name_elem.get_text(strip=True)
            if not name:
                return None
            name = name.replace("®", "").replace("℠", "").replace("™", "")
            card["name"] = name.strip()

            # Extract annual fee and rewards from attribute-list text
            full_text = container.get_text(separator=" ", strip=True)

            # Annual fee like "$95 annual fee" or "$0 annual fee"
            fee_match = re.search(r"\$([\d,]+)\s*annual\s*fee", full_text, re.I)
            if fee_match:
                card["annual_fee"] = int(fee_match.group(1).replace(",", ""))
            else:
                card["annual_fee"] = 0

            # Welcome bonus - look for "bonus miles" or "cash bonus"
            bonus_match = re.search(r"([\d,]+)\s*bonus\s*miles", full_text, re.I)
            if bonus_match:
                card["welcome_bonus"] = f"{bonus_match.group(1)} bonus miles"
            else:
                cash_match = re.search(r"\$([\d,]+)\s*cash\s*bonus", full_text, re.I)
                if cash_match:
                    card["welcome_bonus"] = f"${cash_match.group(1)} cash bonus"

            # Reward rate - look for patterns like "2X miles" or "1.5% cash back"
            rate_match = re.search(
                r"([\d.]+)[Xx%]\s*(miles|cash\s*back|points)", full_text, re.I
            )
            if rate_match:
                card["reward_rates"] = {
                    rate_match.group(2).lower(): rate_match.group(0).strip()
                }

            return card
        except Exception:
            return None

    def parse_card_details(self, card_url: str) -> Optional[Dict[str, Any]]:
        return None
