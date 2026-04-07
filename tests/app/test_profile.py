"""Tests for Story 1.2: /me, /me/profile, /me/cards, /cards/catalog."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("passlib")
pytest.importorskip("jose")

_USER = {
    "email": "bob@example.com",
    "password": "password123",
    "display_name": "Bob",
}


def _signup_and_token(client) -> str:
    res = client.post("/auth/signup", json=_USER)
    assert res.status_code == 201
    return res.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------


def test_get_me_returns_profile_after_signup(test_client):
    token = _signup_and_token(test_client)
    res = test_client.get("/me", headers=_auth(token))
    assert res.status_code == 200
    data = res.json()
    assert data["display_name"] == "Bob"
    assert data["email"] == "bob@example.com"
    assert data["personas"] == []
    assert data["reward_preference"] == "cashback"
    assert data["transaction_logging_enabled"] is False
    assert data["dark_mode"] is False
    assert data["saved_card_ids"] == []


def test_get_me_requires_auth(test_client):
    res = test_client.get("/me")
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# PATCH /me/profile — display name
# ---------------------------------------------------------------------------


def test_patch_display_name(test_client):
    token = _signup_and_token(test_client)
    res = test_client.patch(
        "/me/profile",
        json={"display_name": "Robert"},
        headers=_auth(token),
    )
    assert res.status_code == 200
    assert res.json()["display_name"] == "Robert"


# ---------------------------------------------------------------------------
# PATCH /me/profile — personas
# ---------------------------------------------------------------------------


def test_patch_personas_persists(test_client):
    token = _signup_and_token(test_client)
    res = test_client.patch(
        "/me/profile",
        json={"personas": ["student", "traveler"]},
        headers=_auth(token),
    )
    assert res.status_code == 200
    assert set(res.json()["personas"]) == {"student", "traveler"}


def test_patch_personas_replaces_not_appends(test_client):
    token = _signup_and_token(test_client)
    test_client.patch(
        "/me/profile",
        json={"personas": ["student", "traveler"]},
        headers=_auth(token),
    )
    res = test_client.patch(
        "/me/profile",
        json={"personas": ["family"]},
        headers=_auth(token),
    )
    assert res.status_code == 200
    assert res.json()["personas"] == ["family"]


def test_patch_invalid_persona_returns_422(test_client):
    token = _signup_and_token(test_client)
    res = test_client.patch(
        "/me/profile",
        json={"personas": ["high-roller"]},
        headers=_auth(token),
    )
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# PATCH /me/profile — settings fields
# ---------------------------------------------------------------------------


def test_patch_settings_fields(test_client):
    token = _signup_and_token(test_client)
    res = test_client.patch(
        "/me/profile",
        json={
            "reward_preference": "miles",
            "transaction_logging_enabled": True,
            "dark_mode": True,
        },
        headers=_auth(token),
    )
    assert res.status_code == 200
    data = res.json()
    assert data["reward_preference"] == "miles"
    assert data["transaction_logging_enabled"] is True
    assert data["dark_mode"] is True


def test_patch_invalid_reward_preference_returns_422(test_client):
    token = _signup_and_token(test_client)
    res = test_client.patch(
        "/me/profile",
        json={"reward_preference": "crypto"},
        headers=_auth(token),
    )
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# PUT /me/cards
# ---------------------------------------------------------------------------


def test_put_cards_stores_card_ids(test_client):
    token = _signup_and_token(test_client)
    res = test_client.put(
        "/me/cards",
        json={"card_ids": ["chase_sapphire_preferred", "citi_double_cash"]},
        headers=_auth(token),
    )
    assert res.status_code == 200
    assert set(res.json()["saved_card_ids"]) == {
        "chase_sapphire_preferred",
        "citi_double_cash",
    }


def test_put_cards_replaces_on_second_call(test_client):
    token = _signup_and_token(test_client)
    test_client.put(
        "/me/cards",
        json={"card_ids": ["chase_sapphire_preferred", "citi_double_cash"]},
        headers=_auth(token),
    )
    res = test_client.put(
        "/me/cards",
        json={"card_ids": ["amex_gold"]},
        headers=_auth(token),
    )
    assert res.status_code == 200
    assert res.json()["saved_card_ids"] == ["amex_gold"]


# ---------------------------------------------------------------------------
# GET /cards/catalog
# ---------------------------------------------------------------------------


def test_get_catalog_returns_cards_without_auth(test_client):
    res = test_client.get("/cards/catalog")
    assert res.status_code == 200
    cards = res.json()
    assert len(cards) > 0
    ids = [c["card_id"] for c in cards]
    assert "chase_sapphire_preferred" in ids
    assert "citi_double_cash" in ids


def test_catalog_items_have_required_fields(test_client):
    res = test_client.get("/cards/catalog")
    for card in res.json():
        assert "card_id" in card
        assert "card_name" in card
        assert "issuer" in card
        assert "annual_fee" in card
        assert "reward_highlights" in card
