"""Tests for Story 1.3: PersonaModifier and recommendation endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("passlib")
pytest.importorskip("jose")

# ---------------------------------------------------------------------------
# PersonaModifier unit tests (no HTTP layer)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def modifier():
    from src.app.personas.modifier import PersonaModifier

    return PersonaModifier()


def _make_cards():
    """Two scored cards: one high-fee, one no-fee."""
    return [
        {
            "card_id": "amex_gold",
            "card_name": "Amex Gold",
            "reward_amount": 5.0,
            "annual_fee": 250.0,
            "rank": 1,
        },
        {
            "card_id": "citi_double_cash",
            "card_name": "Citi Double Cash",
            "reward_amount": 4.0,
            "annual_fee": 0.0,
            "rank": 2,
        },
    ]


def test_no_active_personas_returns_unchanged(modifier):
    cards = _make_cards()
    result = modifier.apply(cards, [], "dining")
    assert result["ranked"] == cards
    assert result["persona_context"] == ""


def test_student_persona_penalises_high_fee_card(modifier):
    """Student doubles the fee penalty — Amex Gold ($250 fee) should fall below Citi."""
    cards = _make_cards()
    result = modifier.apply(cards, ["student"], "other")
    ranked = result["ranked"]
    # Amex extra monthly penalty = (2.0-1.0) * 250/12 ≈ 20.83
    # Amex adjusted ≈ 5.0 - 20.83 = -15.83  <  Citi adjusted = 4.0
    assert ranked[0]["card_id"] == "citi_double_cash"
    assert ranked[1]["card_id"] == "amex_gold"


def test_traveler_persona_boosts_travel_category(modifier):
    """Traveler gives 1.5× boost to travel — a card with equal rewards should rise."""
    cards = [
        {
            "card_id": "chase_sapphire_preferred",
            "card_name": "Chase Sapphire Preferred",
            "reward_amount": 3.0,
            "annual_fee": 95.0,
            "rank": 1,
        },
        {
            "card_id": "citi_double_cash",
            "card_name": "Citi Double Cash",
            "reward_amount": 3.0,
            "annual_fee": 0.0,
            "rank": 2,
        },
    ]
    result = modifier.apply(cards, ["traveler"], "travel")
    ranked = result["ranked"]
    # Both start at 3.0 reward, but Chase gets 3.0*1.5=4.5 before fee adjustment
    # Traveler fee_multiplier=0.5 → extra_penalty = (0.5-1.0)*95/12 ≈ -3.96 (credit!)
    # Chase adjusted ≈ 4.5 + 3.96 = 8.46  >  Citi adjusted = 4.5
    assert ranked[0]["card_id"] == "chase_sapphire_preferred"


def test_persona_adjustments_metadata_present(modifier):
    cards = _make_cards()
    result = modifier.apply(cards, ["student"], "dining")
    for card in result["ranked"]:
        assert "persona_adjustments" in card
        adj = card["persona_adjustments"]
        assert "category_boost_applied" in adj
        assert "fee_multiplier_applied" in adj
        assert "extra_fee_penalty" in adj


def test_multiple_personas_averages_multipliers(modifier):
    """Student (fee_mult=2.0) + traveler (fee_mult=0.5) → average = 1.25."""
    cards = _make_cards()
    result = modifier.apply(cards, ["student", "traveler"], "other")
    for card in result["ranked"]:
        assert card["persona_adjustments"]["fee_multiplier_applied"] == pytest.approx(
            1.25, abs=0.01
        )


def test_persona_context_populated_when_active(modifier):
    cards = _make_cards()
    result = modifier.apply(cards, ["student"], "other")
    assert "student" in result["persona_context"].lower()


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

_USER = {
    "email": "carol@example.com",
    "password": "password123",
    "display_name": "Carol",
}


def _signup_and_token(client) -> str:
    res = client.post("/auth/signup", json=_USER)
    assert res.status_code == 201
    return res.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_portfolio_recommendation_is_generic_when_no_saved_cards(test_client):
    token = _signup_and_token(test_client)
    res = test_client.post(
        "/recommendations/portfolio",
        json={"spending_categories": {"dining": 500.0}, "monthly_spend": 500.0},
        headers=_auth(token),
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_generic"] is True
    assert len(data["ranked"]) > 0


def test_portfolio_recommendation_card_finder_excludes_wallet_cards(test_client):
    token = _signup_and_token(test_client)
    test_client.put(
        "/me/cards",
        json={"card_ids": ["citi_double_cash"]},
        headers=_auth(token),
    )
    res = test_client.post(
        "/recommendations/portfolio",
        json={
            "spending_categories": {"dining": 300.0},
            "monthly_spend": 300.0,
            "use_full_catalog": True,
        },
        headers=_auth(token),
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_generic"] is False
    card_ids = [c["card_id"] for c in data["ranked"]]
    assert len(card_ids) == 4
    assert "citi_double_cash" not in card_ids
    assert set(card_ids) == {
        "chase_sapphire_preferred",
        "amex_gold",
        "capital_one_venture",
        "discover_it",
    }


def test_transaction_recommendation_resolves_merchant_category(test_client):
    token = _signup_and_token(test_client)
    res = test_client.post(
        "/recommendations/transaction",
        json={"merchant": "Starbucks", "amount": 6.50},
        headers=_auth(token),
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["ranked"]) > 0
    assert "best_card_id" in data


def test_transaction_recommendation_respects_explicit_category(test_client):
    token = _signup_and_token(test_client)
    res = test_client.post(
        "/recommendations/transaction",
        json={"merchant": "Some Store", "amount": 50.0, "category": "groceries"},
        headers=_auth(token),
    )
    assert res.status_code == 200


def test_persona_context_in_response_when_persona_set(test_client):
    token = _signup_and_token(test_client)
    # Set student persona
    test_client.patch(
        "/me/profile",
        json={"personas": ["student"]},
        headers=_auth(token),
    )
    res = test_client.post(
        "/recommendations/transaction",
        json={"merchant": "Amazon", "amount": 100.0},
        headers=_auth(token),
    )
    assert res.status_code == 200
    data = res.json()
    assert data["active_personas"] == ["student"]
    assert "student" in data["persona_context"].lower()
