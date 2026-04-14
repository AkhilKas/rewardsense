"""Tests for Story 2.1: Portfolio-based recommendations.

Every test follows a Given / When / Then structure and exercises:
  - saved-card wallet vs. curated-catalog fallback
  - enriched response fields (top_card, alternatives, score_breakdown,
    persona_match_reason, projected_savings)
  - persona-aware ranking and reasons
  - optional spending_categories in PortfolioRecommendRequest
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("passlib")
pytest.importorskip("jose")

from src.app.personas.modifier import PersonaModifier  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER = {
    "email": "story21@example.com",
    "password": "password123",
    "display_name": "Story21",
}


def _signup_and_token(client, user: dict | None = None) -> str:
    user = user or _USER
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


# ---------------------------------------------------------------------------
# PersonaModifier.card_persona_reason — unit tests (no HTTP layer)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def modifier():
    return PersonaModifier()


class TestCardPersonaReasonUnit:
    """Unit tests for PersonaModifier.card_persona_reason()."""

    def test_given_no_personas_when_reason_requested_then_default_message(
        self, modifier
    ):
        """Given no active personas
        When card_persona_reason is called
        Then it returns the default 'no active persona' message.
        """
        card = {"card_id": "citi_double_cash", "card_name": "Citi", "annual_fee": 0}
        reason = modifier.card_persona_reason(card, [], "dining")
        assert "no active persona" in reason.lower()

    def test_given_student_and_no_fee_card_when_reason_then_mentions_ideal_fee(
        self, modifier
    ):
        """Given student persona and a $0-fee card with fee multiplier > 1
        When card_persona_reason is called
        Then the reason highlights suitability for fee-sensitive personas.
        """
        card = {
            "card_id": "citi_double_cash",
            "card_name": "Citi Double Cash",
            "annual_fee": 0.0,
            "persona_adjustments": {
                "category_boost_applied": 1.0,
                "fee_multiplier_applied": 2.0,
                "extra_fee_penalty": 0.0,
            },
        }
        reason = modifier.card_persona_reason(card, ["student"], "other")
        assert "no annual fee" in reason.lower()
        assert "fee-sensitive" in reason.lower()

    def test_given_traveler_and_travel_boost_when_reason_then_mentions_boost(
        self, modifier
    ):
        """Given traveler persona and travel category
        When card_persona_reason is called
        Then the reason mentions the category boost.
        """
        card = {
            "card_id": "chase_sapphire_preferred",
            "card_name": "Chase Sapphire Preferred",
            "annual_fee": 95.0,
            "persona_adjustments": {
                "category_boost_applied": 1.5,
                "fee_multiplier_applied": 0.5,
                "extra_fee_penalty": -3.96,
            },
        }
        reason = modifier.card_persona_reason(card, ["traveler"], "travel")
        assert "boosted" in reason.lower()
        assert "1.50" in reason

    def test_given_traveler_and_high_fee_card_when_reason_then_mentions_discount(
        self, modifier
    ):
        """Given traveler persona (fee_mult < 1) and a card with annual fee
        When card_persona_reason is called
        Then the reason mentions that the fee is discounted.
        """
        card = {
            "card_id": "amex_gold",
            "card_name": "Amex Gold",
            "annual_fee": 250.0,
            "persona_adjustments": {
                "category_boost_applied": 1.0,
                "fee_multiplier_applied": 0.5,
                "extra_fee_penalty": -10.42,
            },
        }
        reason = modifier.card_persona_reason(card, ["traveler"], "other")
        assert "discounted" in reason.lower()

    def test_given_student_and_high_fee_card_when_reason_then_mentions_penalty(
        self, modifier
    ):
        """Given student persona (fee_mult > 1) and a high-fee card
        When card_persona_reason is called
        Then the reason mentions the fee penalty.
        """
        card = {
            "card_id": "amex_gold",
            "card_name": "Amex Gold",
            "annual_fee": 250.0,
            "persona_adjustments": {
                "category_boost_applied": 1.0,
                "fee_multiplier_applied": 2.0,
                "extra_fee_penalty": 20.83,
            },
        }
        reason = modifier.card_persona_reason(card, ["student"], "other")
        assert "penalised" in reason.lower()


# ---------------------------------------------------------------------------
# HTTP integration: enriched portfolio response
# ---------------------------------------------------------------------------


class TestPortfolioResponseStructure:
    """Verify that the enriched response shape is correct."""

    def test_given_saved_cards_when_portfolio_recommend_then_response_has_top_card(
        self, test_client
    ):
        """Given a user with saved cards
        When POST /recommendations/portfolio
        Then response includes top_card as a full ScoredCard.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 500.0}, "monthly_spend": 500.0},
            headers=_auth(token),
        )
        assert res.status_code == 200
        data = res.json()
        assert data["top_card"] is not None
        assert data["top_card"]["rank"] == 1
        assert data["top_card"]["card_id"] == data["best_card_id"]

    def test_given_multiple_cards_when_recommend_then_alternatives_excludes_top(
        self, test_client
    ):
        """Given a user with >=2 saved cards
        When POST /recommendations/portfolio
        Then alternatives contains all cards except top_card.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "alt@example.com",
                "password": "password123",
                "display_name": "Alt",
            },
        )
        _save_cards(
            test_client,
            token,
            ["amex_gold", "citi_double_cash", "chase_sapphire_preferred"],
        )
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 400.0}, "monthly_spend": 400.0},
            headers=_auth(token),
        )
        data = res.json()
        assert len(data["alternatives"]) == len(data["ranked"]) - 1
        alt_ids = {c["card_id"] for c in data["alternatives"]}
        assert data["top_card"]["card_id"] not in alt_ids

    def test_given_single_card_when_recommend_then_alternatives_empty(
        self, test_client
    ):
        """Given a user with exactly one saved card
        When POST /recommendations/portfolio
        Then alternatives is an empty list.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "solo@example.com",
                "password": "password123",
                "display_name": "Solo",
            },
        )
        _save_cards(test_client, token, ["citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 100.0}, "monthly_spend": 100.0},
            headers=_auth(token),
        )
        data = res.json()
        assert data["alternatives"] == []
        assert data["top_card"]["card_id"] == "citi_double_cash"


# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------


class TestScoreBreakdown:
    """Verify score_breakdown is populated with expected fields."""

    def test_given_saved_cards_when_recommend_then_breakdown_present(self, test_client):
        """Given a user with saved cards
        When POST /recommendations/portfolio
        Then every ranked card includes a score_breakdown.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "bd@example.com",
                "password": "password123",
                "display_name": "BD",
            },
        )
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 200.0}, "monthly_spend": 200.0},
            headers=_auth(token),
        )
        data = res.json()
        for card in data["ranked"]:
            bd = card["score_breakdown"]
            assert bd is not None
            assert "raw_reward_rate" in bd
            assert "raw_reward_amount" in bd
            assert "personalization_multiplier" in bd
            assert "persona_category_boost" in bd
            assert "persona_fee_penalty" in bd

    def test_given_no_persona_when_recommend_then_breakdown_shows_neutral_boost(
        self, test_client
    ):
        """Given a user with no active persona
        When POST /recommendations/portfolio
        Then persona_category_boost == 1.0 and persona_fee_penalty == 0.0.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "neutral@example.com",
                "password": "password123",
                "display_name": "Neutral",
            },
        )
        _save_cards(test_client, token, ["citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 100.0}, "monthly_spend": 100.0},
            headers=_auth(token),
        )
        data = res.json()
        bd = data["ranked"][0]["score_breakdown"]
        assert bd["persona_category_boost"] == pytest.approx(1.0)
        assert bd["persona_fee_penalty"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Persona match reason
# ---------------------------------------------------------------------------


class TestPersonaMatchReason:
    """Verify persona_match_reason is populated and contextual."""

    def test_given_no_persona_when_recommend_then_reason_is_default(self, test_client):
        """Given a user with no active persona
        When POST /recommendations/portfolio
        Then persona_match_reason indicates no persona is active.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "noreason@example.com",
                "password": "password123",
                "display_name": "NoReason",
            },
        )
        _save_cards(test_client, token, ["citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 100.0}, "monthly_spend": 100.0},
            headers=_auth(token),
        )
        reason = res.json()["ranked"][0]["persona_match_reason"]
        assert "no active persona" in reason.lower()

    def test_given_student_persona_when_recommend_then_reason_mentions_student(
        self, test_client
    ):
        """Given a user with student persona
        When POST /recommendations/portfolio
        Then persona_match_reason references the student persona.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "studentreason@example.com",
                "password": "password123",
                "display_name": "StudentR",
            },
        )
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        _set_personas(test_client, token, ["student"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 300.0}, "monthly_spend": 300.0},
            headers=_auth(token),
        )
        for card in res.json()["ranked"]:
            assert "student" in card["persona_match_reason"].lower()


# ---------------------------------------------------------------------------
# Projected savings
# ---------------------------------------------------------------------------


class TestProjectedSavings:
    """Verify projected_savings is present and reasonable."""

    def test_given_saved_cards_when_recommend_then_projected_savings_positive(
        self, test_client
    ):
        """Given a user with saved cards and non-zero spend
        When POST /recommendations/portfolio
        Then projected_savings > 0 for all cards.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "savings@example.com",
                "password": "password123",
                "display_name": "Savings",
            },
        )
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 500.0}, "monthly_spend": 500.0},
            headers=_auth(token),
        )
        for card in res.json()["ranked"]:
            assert card["projected_savings"] is not None
            assert card["projected_savings"] > 0

    def test_given_monthly_spend_when_recommend_then_savings_annualized(
        self, test_client
    ):
        """Given a monthly_spend of $1000 on dining
        When POST /recommendations/portfolio with a 2% card
        Then projected_savings ≈ raw_reward * 12.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "annual@example.com",
                "password": "password123",
                "display_name": "Annual",
            },
        )
        _save_cards(test_client, token, ["citi_double_cash"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 1000.0}, "monthly_spend": 1000.0},
            headers=_auth(token),
        )
        card = res.json()["ranked"][0]
        raw = card["score_breakdown"]["raw_reward_amount"]
        # projected_savings = raw_reward_amount * 12 (amount == monthly_spend)
        assert card["projected_savings"] == pytest.approx(raw * 12, rel=0.01)


# ---------------------------------------------------------------------------
# Portfolio: empty wallet / wallet-only / card-finder (use_full_catalog)
# ---------------------------------------------------------------------------


class TestWalletFallback:
    """Portfolio behavior depends on saved wallet and use_full_catalog flag."""

    def test_given_no_saved_cards_when_recommend_then_is_generic_true(
        self, test_client
    ):
        """Given a user with no saved cards
        When POST /recommendations/portfolio
        Then is_generic is True and results use the full catalog.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "generic@example.com",
                "password": "password123",
                "display_name": "Generic",
            },
        )
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 100.0}, "monthly_spend": 100.0},
            headers=_auth(token),
        )
        data = res.json()
        assert data["is_generic"] is True
        # All 5 catalog cards should be present
        assert len(data["ranked"]) == 5

    def test_given_saved_cards_when_recommend_then_wallet_only_ranked(
        self, test_client
    ):
        """Given a user with saved cards and default body (no use_full_catalog)
        When POST /recommendations/portfolio
        Then is_generic is False and only saved cards are ranked.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "wallet@example.com",
                "password": "password123",
                "display_name": "Wallet",
            },
        )
        _save_cards(test_client, token, ["amex_gold"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"dining": 200.0}, "monthly_spend": 200.0},
            headers=_auth(token),
        )
        data = res.json()
        assert data["is_generic"] is False
        assert len(data["ranked"]) == 1
        assert data["ranked"][0]["card_id"] == "amex_gold"

    def test_given_saved_cards_when_use_full_catalog_then_excludes_wallet(
        self, test_client
    ):
        token = _signup_and_token(
            test_client,
            {
                "email": "finder@example.com",
                "password": "password123",
                "display_name": "Finder",
            },
        )
        _save_cards(test_client, token, ["amex_gold"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={
                "spending_categories": {"dining": 200.0},
                "monthly_spend": 200.0,
                "use_full_catalog": True,
            },
            headers=_auth(token),
        )
        data = res.json()
        assert data["is_generic"] is False
        ranked_ids = {c["card_id"] for c in data["ranked"]}
        assert len(ranked_ids) == 4
        assert "amex_gold" not in ranked_ids
        assert ranked_ids == {
            "chase_sapphire_preferred",
            "citi_double_cash",
            "capital_one_venture",
            "discover_it",
        }


# ---------------------------------------------------------------------------
# Optional spending_categories
# ---------------------------------------------------------------------------


class TestOptionalSpendingCategories:
    """spending_categories is optional — monthly_spend alone should work."""

    def test_given_no_spending_categories_when_recommend_then_200(self, test_client):
        """Given a request with no spending_categories
        When POST /recommendations/portfolio
        Then the endpoint returns 200 with valid results.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "nocat@example.com",
                "password": "password123",
                "display_name": "NoCat",
            },
        )
        res = test_client.post(
            "/recommendations/portfolio",
            json={"monthly_spend": 500.0},
            headers=_auth(token),
        )
        assert res.status_code == 200
        data = res.json()
        assert data["top_card"] is not None
        assert len(data["ranked"]) > 0

    def test_given_empty_spending_categories_when_recommend_then_falls_back(
        self, test_client
    ):
        """Given spending_categories={}
        When POST /recommendations/portfolio
        Then defaults to 'other' category with monthly_spend as amount.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "empty@example.com",
                "password": "password123",
                "display_name": "Empty",
            },
        )
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {}, "monthly_spend": 300.0},
            headers=_auth(token),
        )
        assert res.status_code == 200
        assert len(res.json()["ranked"]) > 0

    def test_given_no_spend_at_all_when_recommend_then_uses_default_amount(
        self, test_client
    ):
        """Given neither spending_categories nor monthly_spend
        When POST /recommendations/portfolio
        Then endpoint still returns 200 using a $100 default.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "default@example.com",
                "password": "password123",
                "display_name": "Default",
            },
        )
        res = test_client.post(
            "/recommendations/portfolio",
            json={},
            headers=_auth(token),
        )
        assert res.status_code == 200
        assert res.json()["top_card"] is not None


# ---------------------------------------------------------------------------
# Persona-driven ranking (integration)
# ---------------------------------------------------------------------------


class TestPersonaDrivenRanking:
    """End-to-end: persona choice changes which card is recommended first."""

    def test_given_student_persona_when_recommend_then_no_fee_card_ranks_first(
        self, test_client
    ):
        """Given student persona (2× fee penalty)
        When recommending between Amex Gold ($250 fee) and Citi ($0 fee)
        Then Citi Double Cash ranks #1.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "studentrank@example.com",
                "password": "password123",
                "display_name": "StudentRank",
            },
        )
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        _set_personas(test_client, token, ["student"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"other": 100.0}, "monthly_spend": 100.0},
            headers=_auth(token),
        )
        data = res.json()
        assert data["top_card"]["card_id"] == "citi_double_cash"
        assert data["alternatives"][0]["card_id"] == "amex_gold"

    def test_given_traveler_persona_when_travel_spend_then_travel_card_ranks_first(
        self, test_client
    ):
        """Given traveler persona (1.5× travel boost, 0.5× fee mult)
        When spending in travel category with Capital One Venture vs Citi
        Then Capital One Venture ranks #1.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "travelrank@example.com",
                "password": "password123",
                "display_name": "TravelRank",
            },
        )
        _save_cards(test_client, token, ["capital_one_venture", "citi_double_cash"])
        _set_personas(test_client, token, ["traveler"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"travel": 500.0}, "monthly_spend": 500.0},
            headers=_auth(token),
        )
        data = res.json()
        assert data["top_card"]["card_id"] == "capital_one_venture"

    def test_given_multiple_personas_when_recommend_then_blended_adjustments(
        self, test_client
    ):
        """Given both student and traveler personas (opposing fee sensitivities)
        When recommending for travel spending
        Then category boost and fee multiplier are averaged across both personas.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "blended@example.com",
                "password": "password123",
                "display_name": "Blended",
            },
        )
        _save_cards(test_client, token, ["capital_one_venture", "citi_double_cash"])
        _set_personas(test_client, token, ["student", "traveler"])
        res = test_client.post(
            "/recommendations/portfolio",
            json={"spending_categories": {"travel": 500.0}, "monthly_spend": 500.0},
            headers=_auth(token),
        )
        data = res.json()
        assert set(data["active_personas"]) == {"student", "traveler"}
        # Blended travel boost: avg(student=1.0, traveler=1.5) = 1.25
        for card in data["ranked"]:
            assert card["score_breakdown"]["persona_category_boost"] == pytest.approx(
                1.25
            )


# ---------------------------------------------------------------------------
# Transaction endpoint still works with enriched response
# ---------------------------------------------------------------------------


class TestTransactionEndpointEnriched:
    """Verify /recommendations/transaction also returns enriched fields."""

    def test_given_transaction_when_recommend_then_enriched_fields_present(
        self, test_client
    ):
        """Given a transaction recommendation request
        When POST /recommendations/transaction
        Then response includes top_card, alternatives, score_breakdown,
             persona_match_reason, and projected_savings.
        """
        token = _signup_and_token(
            test_client,
            {
                "email": "txn@example.com",
                "password": "password123",
                "display_name": "Txn",
            },
        )
        res = test_client.post(
            "/recommendations/transaction",
            json={"merchant": "Starbucks", "amount": 6.50},
            headers=_auth(token),
        )
        assert res.status_code == 200
        data = res.json()
        assert data["top_card"] is not None
        assert "alternatives" in data
        card = data["top_card"]
        assert card["score_breakdown"] is not None
        assert card["persona_match_reason"] is not None
        assert card["projected_savings"] is not None
