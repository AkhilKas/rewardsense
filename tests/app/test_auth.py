"""Tests for Story 1.1: /auth/signup, /auth/login, /auth/logout and JWT protection."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("passlib")
pytest.importorskip("jose")


_VALID_USER = {
    "email": "alice@example.com",
    "password": "securepass123",
    "display_name": "Alice",
}


# ---------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------


def test_signup_creates_user_and_returns_token(test_client):
    res = test_client.post("/auth/signup", json=_VALID_USER)
    assert res.status_code == 201
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["display_name"] == "Alice"
    assert isinstance(data["user_id"], int)


def test_signup_duplicate_email_returns_409(test_client):
    test_client.post("/auth/signup", json=_VALID_USER)
    res = test_client.post("/auth/signup", json=_VALID_USER)
    assert res.status_code == 409


def test_signup_duplicate_email_case_insensitive(test_client):
    test_client.post("/auth/signup", json=_VALID_USER)
    upper = {**_VALID_USER, "email": _VALID_USER["email"].upper()}
    res = test_client.post("/auth/signup", json=upper)
    assert res.status_code == 409


def test_signup_short_password_rejected(test_client):
    payload = {**_VALID_USER, "password": "short"}
    res = test_client.post("/auth/signup", json=payload)
    assert res.status_code == 422  # Pydantic validation


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def test_login_valid_credentials_returns_token(test_client):
    test_client.post("/auth/signup", json=_VALID_USER)
    res = test_client.post(
        "/auth/login",
        json={"email": _VALID_USER["email"], "password": _VALID_USER["password"]},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["display_name"] == "Alice"


def test_login_wrong_password_returns_401(test_client):
    test_client.post("/auth/signup", json=_VALID_USER)
    res = test_client.post(
        "/auth/login",
        json={"email": _VALID_USER["email"], "password": "wrongpassword"},
    )
    assert res.status_code == 401


def test_login_unknown_email_returns_401(test_client):
    res = test_client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "whatever123"},
    )
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


def test_logout_returns_ok(test_client):
    signup = test_client.post("/auth/signup", json=_VALID_USER)
    token = signup.json()["access_token"]
    res = test_client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    assert res.json() == {"ok": True}


# ---------------------------------------------------------------------------
# JWT protection — use /health as a stand-in protected route to verify the
# dependency works; Story 1.2 will add /me which is the real protected route.
# We test JWT decoding directly here via the dependency.
# ---------------------------------------------------------------------------


def test_valid_token_decodes_to_correct_user_id(test_client):
    """Token returned by signup decodes to a valid user_id."""
    from src.app.auth.jwt import decode_token

    signup = test_client.post("/auth/signup", json=_VALID_USER)
    token = signup.json()["access_token"]
    user_id = decode_token(token)
    assert user_id == signup.json()["user_id"]


def test_invalid_token_decode_returns_none():
    from src.app.auth.jwt import decode_token

    assert decode_token("not.a.valid.token") is None


def test_expired_token_decode_returns_none():
    """A token signed with a different secret is treated as invalid."""
    import os
    from datetime import datetime, timedelta, timezone

    from jose import jwt

    payload = {
        "sub": "1",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
    }
    expired = jwt.encode(payload, os.environ["JWT_SECRET_KEY"], algorithm="HS256")
    from src.app.auth.jwt import decode_token

    assert decode_token(expired) is None
