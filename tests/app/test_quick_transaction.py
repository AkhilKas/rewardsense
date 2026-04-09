"""Tests for Story 2.2: Single-transaction recommendation endpoint.

Every test follows a Given / When / Then structure and exercises:
  - merchant/category heuristic resolution (heuristic-first, user fallback)
  - saved-card-only ranking (no catalog fallback)
  - empty-wallet graceful response
  - enriched response fields (estimated_reward, money_saved, score_breakdown, etc.)
  - optional date passed through to scorer
  - persona-aware ranking for single purchases
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
        "email": f"quick{_COUNTER}@example.com",
        "password": "password123",
        "display_name": f"Quick{_COUNTER}",
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


def _set_personas(client, token: str, personas: list[str]) -> None:
    res = client.patch("/me/profile", json={"personas": personas}, headers=_auth(token))
    assert res.status_code == 200


def _quick(client, token: str, payload: dict) -> dict:
    res = client.post(
        "/recommendations/quick-transaction",
        json=payload,
        headers=_auth(token),
    )
    return res


# ---------------------------------------------------------------------------
# Category resolution — heuristic first, then user fallback
# ---------------------------------------------------------------------------


class TestCategoryResolution:
    """Verify heuristic-first category resolution with user-supplied fallback."""

    def test_given_known_merchant_when_no_category_then_heuristic_resolves(
        self, test_client
    ):
        """Given merchant 'McDonald's' (known dining keyword)
        When no category is supplied
        Then category_used is 'dining'.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "McDonald's", "amount": 15.0})
        assert res.status_code == 200
        assert res.json()["category_used"] == "dining"

    def test_given_known_merchant_when_category_supplied_then_heuristic_wins(
        self, test_client
    ):
        """Given merchant 'Starbucks' (known dining keyword) and category='groceries'
        When POST /recommendations/quick-transaction
        Then heuristic takes priority — category_used is 'dining', not 'groceries'.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(
            test_client,
            token,
            {"merchant": "Starbucks", "amount": 5.0, "category": "groceries"},
        )
        assert res.status_code == 200
        assert res.json()["category_used"] == "dining"

    def test_given_unknown_merchant_when_category_supplied_then_fallback_used(
        self, test_client
    ):
        """Given an unknown merchant and user-supplied category='travel'
        When POST /recommendations/quick-transaction
        Then the user's category is used as fallback.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(
            test_client,
            token,
            {"merchant": "SomeLocalShop", "amount": 42.0, "category": "travel"},
        )
        assert res.status_code == 200
        assert res.json()["category_used"] == "travel"

    def test_given_unknown_merchant_when_no_category_then_defaults_to_other(
        self, test_client
    ):
        """Given an unknown merchant and no category supplied
        When POST /recommendations/quick-transaction
        Then category_used defaults to 'other'.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(
            test_client,
            token,
            {"merchant": "Random Place", "amount": 20.0},
        )
        assert res.status_code == 200
        assert res.json()["category_used"] == "other"


# ---------------------------------------------------------------------------
# Saved-card-only ranking (no catalog fallback)
# ---------------------------------------------------------------------------


class TestSavedCardOnly:
    """Only saved cards are ranked — no catalog fallback."""

    def test_given_saved_cards_when_quick_then_only_saved_cards_returned(
        self, test_client
    ):
        """Given a user with 2 saved cards
        When POST /recommendations/quick-transaction
        Then ranked list contains exactly those 2 cards.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Chipotle", "amount": 12.0})
        data = res.json()
        assert data["has_saved_cards"] is True
        card_ids = {data["top_card"]["card_id"]}
        card_ids.update(c["card_id"] for c in data["alternatives"])
        assert card_ids == {"amex_gold", "citi_double_cash"}

    def test_given_one_saved_card_when_quick_then_alternatives_empty(self, test_client):
        """Given a user with exactly one saved card
        When POST /recommendations/quick-transaction
        Then top_card is that card and alternatives is empty.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Subway", "amount": 8.0})
        data = res.json()
        assert data["top_card"]["card_id"] == "citi_double_cash"
        assert data["alternatives"] == []


# ---------------------------------------------------------------------------
# Empty wallet — graceful response
# ---------------------------------------------------------------------------


class TestEmptyWallet:
    """No saved cards → informative empty response, no 500."""

    def test_given_no_saved_cards_when_quick_then_has_saved_cards_false(
        self, test_client
    ):
        """Given a user with no saved cards
        When POST /recommendations/quick-transaction
        Then has_saved_cards is False, top_card is null, estimated_reward is 0.
        """
        token = _signup_and_token(test_client)
        res = _quick(test_client, token, {"merchant": "Amazon", "amount": 50.0})
        assert res.status_code == 200
        data = res.json()
        assert data["has_saved_cards"] is False
        assert data["top_card"] is None
        assert data["alternatives"] == []
        assert data["estimated_reward"] == 0.0
        assert data["money_saved"] == 0.0

    def test_given_no_saved_cards_when_quick_then_persona_context_prompts_add_cards(
        self, test_client
    ):
        """Given a user with no saved cards
        When POST /recommendations/quick-transaction
        Then persona_context tells the user to add cards.
        """
        token = _signup_and_token(test_client)
        res = _quick(test_client, token, {"merchant": "Target", "amount": 30.0})
        data = res.json()
        assert "add cards" in data["persona_context"].lower()


# ---------------------------------------------------------------------------
# Estimated reward and money_saved
# ---------------------------------------------------------------------------


class TestEstimatedRewardAndMoneySaved:
    """Verify estimated_reward and money_saved are present and correct."""

    def test_given_dining_purchase_with_amex_gold_when_quick_then_reward_positive(
        self, test_client
    ):
        """Given amex_gold (4x dining) as sole saved card and a $100 dining purchase
        When POST /recommendations/quick-transaction
        Then estimated_reward = $4.00 and money_saved = $4.00.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold"])
        res = _quick(test_client, token, {"merchant": "Chipotle", "amount": 100.0})
        data = res.json()
        # amex_gold dining rate = 4.0%, $100 → $4.00 raw reward
        assert data["estimated_reward"] == pytest.approx(4.0, abs=0.01)
        assert data["money_saved"] == pytest.approx(4.0, abs=0.01)

    def test_given_citi_double_cash_when_quick_then_reward_is_2_pct(self, test_client):
        """Given citi_double_cash (2% universal) and a $50 purchase
        When POST /recommendations/quick-transaction
        Then estimated_reward = $1.00.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Random Store", "amount": 50.0})
        data = res.json()
        assert data["estimated_reward"] == pytest.approx(1.0, abs=0.01)
        assert data["money_saved"] == data["estimated_reward"]

    def test_given_multiple_cards_when_quick_then_reward_reflects_top_card(
        self, test_client
    ):
        """Given amex_gold + citi_double_cash and a dining purchase
        When POST /recommendations/quick-transaction
        Then estimated_reward matches the top card's raw reward.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Pizza Hut", "amount": 40.0})
        data = res.json()
        top_raw = data["top_card"]["score_breakdown"]["raw_reward_amount"]
        assert data["estimated_reward"] == pytest.approx(top_raw, abs=0.01)


# ---------------------------------------------------------------------------
# Score breakdown and enriched fields
# ---------------------------------------------------------------------------


class TestEnrichedFields:
    """Verify all Story 2.1 enrichment fields carry through."""

    def test_given_saved_cards_when_quick_then_score_breakdown_present(
        self, test_client
    ):
        """Given a user with saved cards
        When POST /recommendations/quick-transaction
        Then every card has a complete score_breakdown.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Starbucks", "amount": 6.0})
        data = res.json()
        all_cards = [data["top_card"]] + data["alternatives"]
        for card in all_cards:
            bd = card["score_breakdown"]
            assert bd is not None
            assert "raw_reward_rate" in bd
            assert "raw_reward_amount" in bd
            assert "personalization_multiplier" in bd
            assert "persona_category_boost" in bd
            assert "persona_fee_penalty" in bd

    def test_given_saved_cards_when_quick_then_persona_match_reason_present(
        self, test_client
    ):
        """Given a user with saved cards
        When POST /recommendations/quick-transaction
        Then every card has a persona_match_reason string.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Shell", "amount": 45.0})
        data = res.json()
        assert data["top_card"]["persona_match_reason"] is not None

    def test_given_saved_cards_when_quick_then_projected_savings_present(
        self, test_client
    ):
        """Given a user with saved cards
        When POST /recommendations/quick-transaction
        Then every card has projected_savings > 0.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold"])
        res = _quick(test_client, token, {"merchant": "Whole Foods", "amount": 120.0})
        data = res.json()
        assert data["top_card"]["projected_savings"] > 0


# ---------------------------------------------------------------------------
# Optional date
# ---------------------------------------------------------------------------


class TestOptionalDate:
    """Verify optional date field is accepted and passed through."""

    def test_given_valid_date_when_quick_then_200(self, test_client):
        """Given a valid ISO-8601 date
        When POST /recommendations/quick-transaction with date
        Then endpoint returns 200.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(
            test_client,
            token,
            {"merchant": "Uber", "amount": 25.0, "date": "2026-04-09"},
        )
        assert res.status_code == 200
        assert res.json()["top_card"] is not None

    def test_given_no_date_when_quick_then_200(self, test_client):
        """Given no date field
        When POST /recommendations/quick-transaction
        Then endpoint still returns 200.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Uber", "amount": 25.0})
        assert res.status_code == 200

    def test_given_invalid_date_when_quick_then_422(self, test_client):
        """Given a malformed date string
        When POST /recommendations/quick-transaction
        Then endpoint returns 422.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(
            test_client,
            token,
            {"merchant": "Uber", "amount": 25.0, "date": "not-a-date"},
        )
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Persona-aware ranking for single purchases
# ---------------------------------------------------------------------------


class TestPersonaRanking:
    """Persona adjustments affect quick-transaction ranking."""

    def test_given_student_persona_when_dining_then_no_fee_card_preferred(
        self, test_client
    ):
        """Given student persona (2x fee penalty) with amex_gold + citi_double_cash
        When quick-transaction for a small dining purchase
        Then citi_double_cash (no fee) ranks #1.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        _set_personas(test_client, token, ["student"])
        res = _quick(test_client, token, {"merchant": "Taco Bell", "amount": 10.0})
        data = res.json()
        # Student fee penalty on Amex Gold ($250 fee) dominates the small reward
        assert data["top_card"]["card_id"] == "citi_double_cash"

    def test_given_traveler_persona_when_travel_then_travel_card_preferred(
        self, test_client
    ):
        """Given traveler persona with capital_one_venture + citi_double_cash
        When quick-transaction for a travel purchase
        Then capital_one_venture ranks #1.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["capital_one_venture", "citi_double_cash"])
        _set_personas(test_client, token, ["traveler"])
        res = _quick(
            test_client, token, {"merchant": "Delta Airlines", "amount": 350.0}
        )
        data = res.json()
        assert data["top_card"]["card_id"] == "capital_one_venture"

    def test_given_persona_when_quick_then_persona_context_populated(self, test_client):
        """Given a user with traveler persona
        When POST /recommendations/quick-transaction
        Then persona_context mentions 'traveler'.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        _set_personas(test_client, token, ["traveler"])
        res = _quick(test_client, token, {"merchant": "Hilton", "amount": 200.0})
        assert "traveler" in res.json()["persona_context"].lower()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Request validation edge cases."""

    def test_given_zero_amount_when_quick_then_422(self, test_client):
        """Given amount=0
        When POST /recommendations/quick-transaction
        Then 422 because amount must be > 0.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Store", "amount": 0})
        assert res.status_code == 422

    def test_given_negative_amount_when_quick_then_422(self, test_client):
        """Given a negative amount
        When POST /recommendations/quick-transaction
        Then 422.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "Store", "amount": -10.0})
        assert res.status_code == 422

    def test_given_empty_merchant_when_quick_then_422(self, test_client):
        """Given an empty merchant string
        When POST /recommendations/quick-transaction
        Then 422.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _quick(test_client, token, {"merchant": "", "amount": 10.0})
        assert res.status_code == 422

    def test_given_no_auth_when_quick_then_401(self, test_client):
        """Given no authorization header
        When POST /recommendations/quick-transaction
        Then 401.
        """
        res = test_client.post(
            "/recommendations/quick-transaction",
            json={"merchant": "Starbucks", "amount": 5.0},
        )
        assert res.status_code == 401
