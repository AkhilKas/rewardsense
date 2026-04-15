"""
Tests for Transaction Ledger, Summary, and Export.

Coverage:
- Opt-in required before persistence, manual log entry
- Schema fields stored correctly, card_was_saved detection
- Summary aggregates categories and cards correctly
- CSV/XLSX export schema matches stored fields

Run:
    PYTHONPATH=. pytest tests/app/test_transactions.py -v -o "addopts="
"""

from __future__ import annotations

import csv
import io

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SIGNUP_PAYLOAD = {
    "email": "txn@test.com",
    "password": "StrongPass123!",
    "display_name": "Txn User",
}


def _signup_and_login(client) -> str:
    """Create a user and return the auth token."""
    client.post("/auth/signup", json=SIGNUP_PAYLOAD)
    resp = client.post(
        "/auth/login",
        json={"email": SIGNUP_PAYLOAD["email"], "password": SIGNUP_PAYLOAD["password"]},
    )
    return resp.json()["access_token"]


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _enable_logging(client, token: str) -> None:
    """Enable transaction logging for the user."""
    client.patch(
        "/me/profile",
        json={"transaction_logging_enabled": True},
        headers=_auth_header(token),
    )


def _save_cards(client, token: str, card_ids: list) -> None:
    """Set the user's saved card wallet."""
    client.put(
        "/me/cards",
        json={"card_ids": card_ids},
        headers=_auth_header(token),
    )


def _create_txn(client, token: str, **overrides) -> dict:
    """Create a transaction and return the response dict."""
    payload = {
        "merchant": "Test Merchant",
        "amount": 50.0,
        "category": "dining",
        "chosen_card_id": "amex_gold",
        "chosen_card_name": "Amex Gold Card",
        "reward_earned": 2.0,
        "estimated_savings": 1.5,
        "source_flow": "manual",
    }
    payload.update(overrides)
    resp = client.post(
        "/transactions",
        json=payload,
        headers=_auth_header(token),
    )
    return resp.json(), resp.status_code


# =====================================================================
# Story 3.1: Opt-in required
# =====================================================================


class TestTransactionOptIn:
    def test_create_rejected_when_logging_disabled(self, test_client):
        token = _signup_and_login(test_client)
        # Do NOT enable logging
        body, status = _create_txn(test_client, token)
        assert status == 403
        assert "disabled" in body["detail"].lower()

    def test_create_succeeds_when_logging_enabled(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)
        body, status = _create_txn(test_client, token)
        assert status == 201
        assert body["merchant"] == "Test Merchant"

    def test_list_works_without_logging_enabled(self, test_client):
        """Listing should work even if logging is off (just returns empty)."""
        token = _signup_and_login(test_client)
        resp = test_client.get("/transactions", headers=_auth_header(token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["total_rewards"] == 0.0
        assert body["total_savings"] == 0.0

    def test_unauthenticated_rejected(self, test_client):
        resp = test_client.post(
            "/transactions", json={"merchant": "x", "amount": 10, "category": "dining"}
        )
        assert resp.status_code == 401


# =====================================================================
# Story 3.2: Schema fields
# =====================================================================


class TestTransactionSchema:
    def test_all_fields_stored(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)

        body, status = _create_txn(
            test_client,
            token,
            merchant="Whole Foods",
            amount=85.50,
            category="Groceries",
            chosen_card_id="blue_cash_preferred",
            chosen_card_name="Blue Cash Preferred",
            reward_earned=5.13,
            estimated_savings=3.42,
            source_flow="portfolio",
        )
        assert status == 201
        assert body["merchant"] == "Whole Foods"
        assert body["amount"] == 85.50
        assert body["category"] == "groceries"  # lowered
        assert body["chosen_card_id"] == "blue_cash_preferred"
        assert body["reward_earned"] == 5.13
        assert body["estimated_savings"] == 3.42
        assert body["source_flow"] == "portfolio"
        assert body["timestamp"]  # non-empty

    def test_card_was_saved_true_when_card_in_wallet(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)
        _save_cards(test_client, token, ["amex_gold"])

        body, status = _create_txn(test_client, token, chosen_card_id="amex_gold")
        assert status == 201
        assert body["card_was_saved"] is True

    def test_card_was_saved_false_when_card_not_in_wallet(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)
        # No cards saved

        body, status = _create_txn(test_client, token, chosen_card_id="amex_gold")
        assert status == 201
        assert body["card_was_saved"] is False

    def test_invalid_source_flow_rejected(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)

        body, status = _create_txn(test_client, token, source_flow="invalid_flow")
        assert status == 422

    def test_custom_timestamp(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)

        body, status = _create_txn(
            test_client, token, timestamp="2026-03-15T12:30:00+00:00"
        )
        assert status == 201
        assert "2026-03-15" in body["timestamp"]

    def test_invalid_timestamp_rejected(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)

        body, status = _create_txn(test_client, token, timestamp="not-a-date")
        assert status == 422


# =====================================================================
# Story 3.1: Pagination
# =====================================================================


class TestTransactionPagination:
    def test_pagination(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)

        # Create 5 transactions
        for i in range(5):
            _create_txn(test_client, token, merchant=f"Store {i}", amount=10.0 + i)

        # Page 1 of 2
        resp = test_client.get(
            "/transactions?page=1&page_size=3",
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["transactions"]) == 3
        assert data["has_next"] is True
        assert data["page"] == 1

        # Page 2
        resp2 = test_client.get(
            "/transactions?page=2&page_size=3",
            headers=_auth_header(token),
        )
        data2 = resp2.json()
        assert len(data2["transactions"]) == 2
        assert data2["has_next"] is False

    def test_empty_history(self, test_client):
        token = _signup_and_login(test_client)
        resp = test_client.get("/transactions", headers=_auth_header(token))
        data = resp.json()
        assert data["total"] == 0
        assert data["transactions"] == []
        assert data["total_rewards"] == 0.0
        assert data["total_savings"] == 0.0


# =====================================================================
# Story 3.3: Summary
# =====================================================================


class TestTransactionSummary:
    def test_empty_summary(self, test_client):
        token = _signup_and_login(test_client)
        resp = test_client.get("/summary", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["transaction_count"] == 0
        assert data["total_spend"] == 0.0
        assert len(data["top_insights"]) >= 1
        assert "no data" in data["top_insights"][0]["label"].lower()

    def test_summary_aggregates_correctly(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)

        # Dining transactions
        _create_txn(
            test_client,
            token,
            merchant="Restaurant A",
            amount=100.0,
            category="dining",
            chosen_card_id="amex_gold",
            chosen_card_name="Amex Gold",
            reward_earned=4.0,
            estimated_savings=3.0,
        )
        _create_txn(
            test_client,
            token,
            merchant="Restaurant B",
            amount=50.0,
            category="dining",
            chosen_card_id="amex_gold",
            chosen_card_name="Amex Gold",
            reward_earned=2.0,
            estimated_savings=1.5,
        )
        # Groceries transaction
        _create_txn(
            test_client,
            token,
            merchant="Whole Foods",
            amount=80.0,
            category="groceries",
            chosen_card_id="blue_cash_preferred",
            chosen_card_name="Blue Cash",
            reward_earned=4.8,
            estimated_savings=4.0,
        )

        resp = test_client.get("/summary", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()

        assert data["transaction_count"] == 3
        assert data["total_spend"] == 230.0
        assert data["total_rewards"] == 10.8

        # Category breakdown
        cats = {c["category"]: c for c in data["spend_by_category"]}
        assert "dining" in cats
        assert cats["dining"]["total_spend"] == 150.0
        assert cats["dining"]["transaction_count"] == 2
        assert "groceries" in cats
        assert cats["groceries"]["total_spend"] == 80.0

        # Card breakdown
        cards = {c["card_id"]: c for c in data["savings_by_card"]}
        assert "amex_gold" in cards
        assert cards["amex_gold"]["transaction_count"] == 2

    def test_summary_unauthenticated(self, test_client):
        resp = test_client.get("/summary")
        assert resp.status_code == 401


# =====================================================================
# Story 3.4: Export
# =====================================================================


class TestTransactionExport:
    def test_csv_export_empty(self, test_client):
        token = _signup_and_login(test_client)
        resp = test_client.get(
            "/transactions/export?format=csv",
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_csv_export_with_data(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)
        _create_txn(test_client, token, merchant="Export Test", amount=42.0)
        _create_txn(test_client, token, merchant="Export Test 2", amount=18.0)

        resp = test_client.get(
            "/transactions/export?format=csv",
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows
        assert rows[0] == [
            "id",
            "merchant",
            "amount",
            "category",
            "chosen_card_id",
            "chosen_card_name",
            "reward_earned",
            "estimated_savings",
            "source_flow",
            "card_was_saved",
            "timestamp",
        ]

    def test_xlsx_export(self, test_client):
        token = _signup_and_login(test_client)
        _enable_logging(test_client, token)
        _create_txn(test_client, token, merchant="XLSX Test", amount=33.0)

        resp = test_client.get(
            "/transactions/export?format=xlsx",
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        # Should be either XLSX or CSV fallback
        assert len(resp.content) > 0

    def test_export_only_own_rows(self, test_client):
        """Each user should only export their own transactions."""
        # User 1
        test_client.post(
            "/auth/signup",
            json={
                "email": "user1@test.com",
                "password": "Pass123!",
                "display_name": "U1",
            },
        )
        resp1 = test_client.post(
            "/auth/login", json={"email": "user1@test.com", "password": "Pass123!"}
        )
        token1 = resp1.json()["access_token"]
        _enable_logging(test_client, token1)
        _create_txn(test_client, token1, merchant="User1 Store", amount=10.0)

        # User 2
        test_client.post(
            "/auth/signup",
            json={
                "email": "user2@test.com",
                "password": "Pass456!",
                "display_name": "U2",
            },
        )
        resp2 = test_client.post(
            "/auth/login", json={"email": "user2@test.com", "password": "Pass456!"}
        )
        token2 = resp2.json()["access_token"]
        _enable_logging(test_client, token2)
        _create_txn(test_client, token2, merchant="User2 Store", amount=20.0)

        # User 1 export should only contain their row
        resp = test_client.get(
            "/transactions/export?format=csv", headers=_auth_header(token1)
        )
        assert "User1 Store" in resp.text
        assert "User2 Store" not in resp.text

    def test_export_unauthenticated(self, test_client):
        resp = test_client.get("/transactions/export?format=csv")
        assert resp.status_code == 401

    def test_invalid_format_rejected(self, test_client):
        token = _signup_and_login(test_client)
        resp = test_client.get(
            "/transactions/export?format=pdf",
            headers=_auth_header(token),
        )
        assert resp.status_code == 422
