"""Tests for Story 2.4: Card images and display contracts.

Every test follows a Given / When / Then structure and exercises:
  - image_url populated in catalog responses
  - CardDisplayInfo present on ScoredCard in recommendation responses
  - CardDisplayInfo present on CardSavingsDetail in calculator responses
  - Consistent card_display shape across all endpoints
  - Normalized fields (issuer, reward_highlights, image_url) match catalog
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("passlib")
pytest.importorskip("jose")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COUNTER = 0


def _unique_user() -> dict:
    global _COUNTER
    _COUNTER += 1
    return {
        "email": f"display{_COUNTER}@example.com",
        "password": "password123",
        "display_name": f"Display{_COUNTER}",
    }


def _signup_and_token(client, user: dict | None = None) -> str:
    user = user or _unique_user()
    res = client.post("/auth/signup", json=user)
    assert res.status_code == 201
    return res.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _save_cards(client, token: str, card_ids: list[str]) -> None:
    res = client.put("/me/cards", json={"card_ids": card_ids}, headers=_auth(token))
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Catalog image URLs
# ---------------------------------------------------------------------------


class TestCatalogImageUrls:
    """Verify /cards/catalog returns image_url for every card."""

    def test_given_catalog_when_fetched_then_all_cards_have_image_url(
        self, test_client
    ):
        """Given the public card catalog
        When GET /cards/catalog
        Then every card has a non-null image_url ending in .svg.
        """
        res = test_client.get("/cards/catalog")
        assert res.status_code == 200
        for card in res.json():
            assert card["image_url"] is not None
            assert card["image_url"].endswith(".svg")

    def test_given_catalog_when_fetched_then_image_url_contains_card_id(
        self, test_client
    ):
        """Given the public card catalog
        When GET /cards/catalog
        Then each image_url contains the card's own card_id.
        """
        res = test_client.get("/cards/catalog")
        for card in res.json():
            assert card["card_id"] in card["image_url"]


# ---------------------------------------------------------------------------
# CardDisplayInfo on portfolio recommendations
# ---------------------------------------------------------------------------


class TestPortfolioCardDisplay:
    """Verify card_display is present on ScoredCard in portfolio recommendations."""

    def test_given_saved_cards_when_portfolio_recommend_then_card_display_present(
        self, test_client
    ):
        """Given a user with saved cards
        When POST /recommendations/portfolio
        Then every ranked card has a card_display with all required fields.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 500}, "monthly_spend": 500},
            headers=_auth(token),
        )
        assert res.status_code == 200
        for card in res.json()["ranked"]:
            cd = card["card_display"]
            assert cd is not None
            assert "card_id" in cd
            assert "card_name" in cd
            assert "issuer" in cd
            assert "annual_fee" in cd
            assert "reward_highlights" in cd
            assert "image_url" in cd

    def test_given_saved_cards_when_portfolio_then_display_matches_catalog(
        self, test_client
    ):
        """Given a user with amex_gold saved
        When POST /recommendations/portfolio
        Then card_display for amex_gold matches the catalog entry.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold"])
        catalog_res = test_client.get("/cards/catalog")
        catalog_amex = next(
            c for c in catalog_res.json() if c["card_id"] == "amex_gold"
        )

        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 200}, "monthly_spend": 200},
            headers=_auth(token),
        )
        cd = res.json()["ranked"][0]["card_display"]
        assert cd["card_id"] == catalog_amex["card_id"]
        assert cd["card_name"] == catalog_amex["card_name"]
        assert cd["issuer"] == catalog_amex["issuer"]
        assert cd["annual_fee"] == catalog_amex["annual_fee"]
        assert cd["reward_highlights"] == catalog_amex["reward_highlights"]
        assert cd["image_url"] == catalog_amex["image_url"]

    def test_given_saved_cards_when_portfolio_then_top_card_has_display(
        self, test_client
    ):
        """Given a user with saved cards
        When POST /recommendations/portfolio
        Then top_card also has card_display.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 100}, "monthly_spend": 100},
            headers=_auth(token),
        )
        top = res.json()["top_card"]
        assert top["card_display"] is not None
        assert top["card_display"]["card_id"] == "citi_double_cash"


# ---------------------------------------------------------------------------
# CardDisplayInfo on quick-transaction recommendations
# ---------------------------------------------------------------------------


class TestQuickTransactionCardDisplay:
    """Verify card_display on quick-transaction responses."""

    def test_given_saved_cards_when_quick_then_card_display_present(self, test_client):
        """Given a user with saved cards
        When POST /recommendations/quick-transaction
        Then top_card and alternatives each have card_display.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = test_client.post(
            "/recommendations/quick-transaction",
            json={"merchant": "Chipotle", "amount": 15.0},
            headers=_auth(token),
        )
        data = res.json()
        assert data["top_card"]["card_display"] is not None
        for alt in data["alternatives"]:
            assert alt["card_display"] is not None

    def test_given_no_saved_cards_when_quick_then_top_card_is_null(self, test_client):
        """Given a user with no saved cards
        When POST /recommendations/quick-transaction
        Then top_card is null (so card_display is not applicable).
        """
        token = _signup_and_token(test_client)
        res = test_client.post(
            "/recommendations/quick-transaction",
            json={"merchant": "Starbucks", "amount": 5.0},
            headers=_auth(token),
        )
        assert res.json()["top_card"] is None


# ---------------------------------------------------------------------------
# CardDisplayInfo on savings calculator
# ---------------------------------------------------------------------------


class TestSavingsCalculatorCardDisplay:
    """Verify card_display on savings calculator card entries."""

    def test_given_saved_cards_when_calc_then_card_display_present(self, test_client):
        """Given a user with saved cards
        When POST /recommendations/savings-calculator
        Then every card entry has a card_display.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = test_client.post(
            "/recommendations/savings-calculator",
            json={"spending_by_category": {"dining": 500}},
            headers=_auth(token),
        )
        assert res.status_code == 200
        for card in res.json()["cards"]:
            assert card["card_display"] is not None
            assert card["card_display"]["image_url"] is not None

    def test_given_saved_cards_when_calc_then_display_matches_card_id(
        self, test_client
    ):
        """Given a user with saved cards
        When POST /recommendations/savings-calculator
        Then card_display.card_id matches the parent card_id.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["capital_one_venture", "discover_it"])
        res = test_client.post(
            "/recommendations/savings-calculator",
            json={"spending_by_category": {"travel": 300}},
            headers=_auth(token),
        )
        for card in res.json()["cards"]:
            assert card["card_display"]["card_id"] == card["card_id"]


# ---------------------------------------------------------------------------
# Cross-endpoint consistency
# ---------------------------------------------------------------------------


class TestCrossEndpointConsistency:
    """card_display shape is identical across catalog, portfolio, and calculator."""

    def test_given_amex_gold_when_all_endpoints_then_display_fields_match(
        self, test_client
    ):
        """Given amex_gold in wallet
        When fetching catalog, portfolio recommendation, and savings calculator
        Then card_display fields are identical across all three.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold"])

        # Catalog
        catalog_res = test_client.get("/cards/catalog")
        catalog_amex = next(
            c for c in catalog_res.json() if c["card_id"] == "amex_gold"
        )

        # Portfolio recommendation
        portfolio_res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 100}, "monthly_spend": 100},
            headers=_auth(token),
        )
        portfolio_display = portfolio_res.json()["ranked"][0]["card_display"]

        # Savings calculator
        calc_res = test_client.post(
            "/recommendations/savings-calculator",
            json={"spending_by_category": {"dining": 100}},
            headers=_auth(token),
        )
        calc_card = next(
            c for c in calc_res.json()["cards"] if c["card_id"] == "amex_gold"
        )
        calc_display = calc_card["card_display"]

        # All three should match
        for display in [portfolio_display, calc_display]:
            assert display["card_id"] == catalog_amex["card_id"]
            assert display["card_name"] == catalog_amex["card_name"]
            assert display["issuer"] == catalog_amex["issuer"]
            assert display["annual_fee"] == catalog_amex["annual_fee"]
            assert display["reward_highlights"] == catalog_amex["reward_highlights"]
            assert display["image_url"] == catalog_amex["image_url"]


# ---------------------------------------------------------------------------
# Transaction endpoint also has card_display
# ---------------------------------------------------------------------------


class TestTransactionCardDisplay:
    """Verify /recommendations/transaction also returns card_display."""

    def test_given_saved_cards_when_transaction_then_card_display_present(
        self, test_client
    ):
        """Given a user with saved cards
        When POST /recommendations/transaction
        Then every ranked card has card_display with image_url.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["chase_sapphire_preferred"])
        res = test_client.post(
            "/recommendations/transaction",
            json={"merchant": "Starbucks", "amount": 6.50},
            headers=_auth(token),
        )
        assert res.status_code == 200
        for card in res.json()["ranked"]:
            assert card["card_display"] is not None
            assert card["card_display"]["image_url"] is not None
