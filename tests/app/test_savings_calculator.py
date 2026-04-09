"""Tests for Story 2.3: Savings calculator and card impact views.

Every test follows a Given / When / Then structure and exercises:
  - baseline selection (first saved catch-all card vs generic 1%)
  - category-by-category reward and uplift computation
  - monthly / annual totals and net benefit (accounting for fee difference)
  - default spending profile when none supplied
  - empty wallet behaviour
  - sorting by net_annual_benefit
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
        "email": f"savings_calc{_COUNTER}@example.com",
        "password": "password123",
        "display_name": f"Calc{_COUNTER}",
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


def _calc(client, token: str, payload: dict) -> dict:
    return client.post(
        "/recommendations/savings-calculator",
        json=payload,
        headers=_auth(token),
    )


# ---------------------------------------------------------------------------
# Unit: baseline detection helpers
# ---------------------------------------------------------------------------


class TestBaselineDetection:
    """Verify _find_baseline picks the right comparison card."""

    def test_given_catch_all_card_saved_when_calc_then_baseline_is_that_card(
        self, test_client
    ):
        """Given citi_double_cash (no category bonuses) is in the wallet
        When POST /recommendations/savings-calculator
        Then baseline_card_id is 'citi_double_cash'.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 500}, "monthly_spend": 500},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["baseline_card_id"] == "citi_double_cash"
        assert "Citi" in data["baseline_card_name"]

    def test_given_only_category_cards_when_calc_then_baseline_is_generic(
        self, test_client
    ):
        """Given only cards with category bonuses are saved
        When POST /recommendations/savings-calculator
        Then baseline_card_id is null (generic 1%).
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "chase_sapphire_preferred"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 500}, "monthly_spend": 500},
        )
        data = res.json()
        assert data["baseline_card_id"] is None
        assert "1%" in data["baseline_card_name"].lower()
        assert data["baseline_annual_fee"] == 0.0

    def test_given_multiple_catch_all_cards_when_calc_then_first_saved_wins(
        self, test_client
    ):
        """Given discover_it and citi_double_cash both saved (catch-all)
        When POST /recommendations/savings-calculator
        Then baseline is discover_it because it was listed first.
        """
        token = _signup_and_token(test_client)
        # discover_it listed first in the PUT payload
        _save_cards(test_client, token, ["discover_it", "citi_double_cash"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"other": 100}, "monthly_spend": 100},
        )
        data = res.json()
        assert data["baseline_card_id"] == "discover_it"


# ---------------------------------------------------------------------------
# Category-by-category breakdown
# ---------------------------------------------------------------------------


class TestCategoryBreakdown:
    """Verify per-category reward, baseline, and uplift values."""

    def test_given_amex_gold_and_dining_when_calc_then_dining_reward_is_4pct(
        self, test_client
    ):
        """Given amex_gold (4x dining) and $500 dining spend
        When POST /recommendations/savings-calculator
        Then amex_gold dining reward_amount == $20.00.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 500.0}},
        )
        data = res.json()
        amex = next(c for c in data["cards"] if c["card_id"] == "amex_gold")
        dining = next(r for r in amex["categories"] if r["category"] == "dining")
        assert dining["reward_amount"] == pytest.approx(20.0, abs=0.01)
        assert dining["monthly_spend"] == 500.0

    def test_given_citi_baseline_when_calc_then_baseline_reward_is_2pct(
        self, test_client
    ):
        """Given citi_double_cash (2% universal) as baseline and $500 dining
        When POST /recommendations/savings-calculator
        Then amex_gold's dining baseline_reward == $10.00 (2% of $500).
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 500.0}},
        )
        data = res.json()
        amex = next(c for c in data["cards"] if c["card_id"] == "amex_gold")
        dining = next(r for r in amex["categories"] if r["category"] == "dining")
        assert dining["baseline_reward"] == pytest.approx(10.0, abs=0.01)

    def test_given_known_rewards_when_calc_then_uplift_is_reward_minus_baseline(
        self, test_client
    ):
        """Given amex_gold 4% dining ($20) vs citi 2% baseline ($10)
        When POST /recommendations/savings-calculator
        Then uplift == $10.00.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 500.0}},
        )
        data = res.json()
        amex = next(c for c in data["cards"] if c["card_id"] == "amex_gold")
        dining = next(r for r in amex["categories"] if r["category"] == "dining")
        assert dining["uplift"] == pytest.approx(10.0, abs=0.01)

    def test_given_multi_category_spend_when_calc_then_all_categories_present(
        self, test_client
    ):
        """Given spending in dining, groceries, and travel
        When POST /recommendations/savings-calculator
        Then each card has rows for all three categories.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        cats = {"dining": 300.0, "groceries": 400.0, "travel": 200.0}
        res = _calc(test_client, token, {"spending_by_category": cats})
        data = res.json()
        for card in data["cards"]:
            returned_cats = {r["category"] for r in card["categories"]}
            assert returned_cats == {"dining", "groceries", "travel"}


# ---------------------------------------------------------------------------
# Monthly / annual totals and net benefit
# ---------------------------------------------------------------------------


class TestTotals:
    """Verify monthly, annual, and net benefit aggregation."""

    def test_given_single_category_when_calc_then_annual_is_monthly_times_12(
        self, test_client
    ):
        """Given a single dining category
        When POST /recommendations/savings-calculator
        Then annual_reward_total == monthly_reward_total * 12.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 1000.0}},
        )
        data = res.json()
        for card in data["cards"]:
            assert card["annual_reward_total"] == pytest.approx(
                card["monthly_reward_total"] * 12, rel=0.01
            )

    def test_given_fee_card_when_calc_then_net_benefit_subtracts_fee_difference(
        self, test_client
    ):
        """Given amex_gold ($250 fee) vs citi baseline ($0 fee)
        When POST /recommendations/savings-calculator
        Then net_annual_benefit == annual_uplift - 250.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 500.0}},
        )
        data = res.json()
        amex = next(c for c in data["cards"] if c["card_id"] == "amex_gold")
        expected_net = amex["annual_uplift_vs_baseline"] - (250 - 0)
        assert amex["net_annual_benefit"] == pytest.approx(expected_net, abs=0.01)

    def test_given_baseline_card_when_calc_then_its_uplift_is_zero(self, test_client):
        """Given citi_double_cash is the baseline
        When POST /recommendations/savings-calculator
        Then citi's monthly_uplift_vs_baseline == 0.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "citi_double_cash"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 500.0}},
        )
        data = res.json()
        citi = next(c for c in data["cards"] if c["card_id"] == "citi_double_cash")
        assert citi["monthly_uplift_vs_baseline"] == pytest.approx(0.0, abs=0.01)
        assert citi["annual_uplift_vs_baseline"] == pytest.approx(0.0, abs=0.01)
        assert citi["net_annual_benefit"] == pytest.approx(0.0, abs=0.01)

    def test_given_generic_baseline_when_calc_then_uplift_vs_1pct(self, test_client):
        """Given only category cards (no catch-all), generic 1% baseline
        When POST /recommendations/savings-calculator with $1000 dining
        Then amex_gold uplift = ($40 - $10) = $30/month.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["amex_gold", "chase_sapphire_preferred"])
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 1000.0}},
        )
        data = res.json()
        amex = next(c for c in data["cards"] if c["card_id"] == "amex_gold")
        assert amex["monthly_uplift_vs_baseline"] == pytest.approx(30.0, abs=0.01)


# ---------------------------------------------------------------------------
# Default spending profile
# ---------------------------------------------------------------------------


class TestDefaultSpending:
    """When no spending_by_category is provided, a default profile is used."""

    def test_given_no_spending_when_calc_then_default_categories_used(
        self, test_client
    ):
        """Given no spending_by_category and monthly_spend=1000
        When POST /recommendations/savings-calculator
        Then spending_profile uses default categories summing to ~1000.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _calc(test_client, token, {"monthly_spend": 1000.0})
        data = res.json()
        assert data["total_monthly_spend"] == pytest.approx(1000.0, abs=1.0)
        assert len(data["spending_profile"]) > 1

    def test_given_empty_payload_when_calc_then_200_with_defaults(self, test_client):
        """Given completely empty payload
        When POST /recommendations/savings-calculator
        Then 200 with a default $1000 spending profile.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        res = _calc(test_client, token, {})
        assert res.status_code == 200
        data = res.json()
        assert data["total_monthly_spend"] == pytest.approx(1000.0, abs=1.0)
        assert len(data["cards"]) > 0


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSorting:
    """Cards are sorted by net_annual_benefit descending."""

    def test_given_multiple_cards_when_calc_then_sorted_by_net_benefit(
        self, test_client
    ):
        """Given amex_gold, chase_sapphire, citi_double_cash
        When POST /recommendations/savings-calculator
        Then cards list is sorted by net_annual_benefit descending.
        """
        token = _signup_and_token(test_client)
        _save_cards(
            test_client,
            token,
            ["amex_gold", "chase_sapphire_preferred", "citi_double_cash"],
        )
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 500.0, "groceries": 300.0}},
        )
        data = res.json()
        benefits = [c["net_annual_benefit"] for c in data["cards"]]
        assert benefits == sorted(benefits, reverse=True)


# ---------------------------------------------------------------------------
# Empty wallet
# ---------------------------------------------------------------------------


class TestEmptyWallet:
    """When the user has no saved cards, fall back to full catalog."""

    def test_given_no_saved_cards_when_calc_then_all_catalog_cards_shown(
        self, test_client
    ):
        """Given a user with no saved cards
        When POST /recommendations/savings-calculator
        Then all 5 catalog cards are evaluated.
        """
        token = _signup_and_token(test_client)
        res = _calc(
            test_client,
            token,
            {"spending_by_category": {"dining": 200.0}, "monthly_spend": 200.0},
        )
        assert res.status_code == 200
        data = res.json()
        assert len(data["cards"]) == 5


# ---------------------------------------------------------------------------
# Auth required
# ---------------------------------------------------------------------------


class TestAuth:
    """Endpoint requires authentication."""

    def test_given_no_auth_when_calc_then_401(self, test_client):
        """Given no authorization header
        When POST /recommendations/savings-calculator
        Then 401.
        """
        res = test_client.post(
            "/recommendations/savings-calculator",
            json={"spending_by_category": {"dining": 100}},
        )
        assert res.status_code == 401


# ---------------------------------------------------------------------------
# Spending profile echo
# ---------------------------------------------------------------------------


class TestSpendingProfileEcho:
    """Response echoes back the spending profile used."""

    def test_given_explicit_spending_when_calc_then_profile_echoed(self, test_client):
        """Given explicit spending_by_category
        When POST /recommendations/savings-calculator
        Then spending_profile in response matches input.
        """
        token = _signup_and_token(test_client)
        _save_cards(test_client, token, ["citi_double_cash"])
        cats = {"dining": 300.0, "travel": 200.0}
        res = _calc(test_client, token, {"spending_by_category": cats})
        data = res.json()
        assert data["spending_profile"] == cats
        assert data["total_monthly_spend"] == pytest.approx(500.0)
