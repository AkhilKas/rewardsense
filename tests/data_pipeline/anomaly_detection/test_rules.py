"""
Unit tests for domain-specific anomaly rules.

Tests cover credit card, transaction, and user profile
business logic validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.anomaly_detection.detectors import AnomalySeverity
from src.data_pipeline.anomaly_detection.rules import DomainRuleEngine


@pytest.fixture
def engine():
    return DomainRuleEngine()


# =====================================================================
# Credit card rules
# =====================================================================


class TestCreditCardRules:
    def test_clean_cards_no_anomalies(self, engine):
        df = pd.DataFrame(
            {
                "card_name": ["Card A", "Card B"],
                "annual_fee": [95, 250],
                "base_reward_rate": [1.5, 2.0],
                "welcome_bonus_roi": [0.15, 0.10],
                "discontinued": [False, False],
            }
        )
        anomalies = engine.check_credit_card_rules(df)
        assert len(anomalies) == 0

    def test_reward_rate_too_high(self, engine):
        df = pd.DataFrame(
            {
                "card_name": ["Bugged Card"],
                "base_reward_rate": [50.0],  # clearly a parsing error
            }
        )
        anomalies = engine.check_credit_card_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_reward_rate_too_high" in names
        assert anomalies[0].severity == AnomalySeverity.CRITICAL

    def test_extreme_annual_fee(self, engine):
        df = pd.DataFrame(
            {
                "card_name": ["Expensive Card"],
                "annual_fee": [10000],
            }
        )
        anomalies = engine.check_credit_card_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_annual_fee_extreme" in names

    def test_extreme_welcome_bonus_roi(self, engine):
        df = pd.DataFrame(
            {
                "card_name": ["Too Good Card"],
                "welcome_bonus_roi": [5.0],  # 500% ROI
            }
        )
        anomalies = engine.check_credit_card_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_welcome_bonus_roi_extreme" in names

    def test_high_discontinued_ratio(self, engine):
        df = pd.DataFrame(
            {
                "card_name": [f"Card {i}" for i in range(10)],
                "discontinued": [True] * 5 + [False] * 5,  # 50% discontinued
            }
        )
        anomalies = engine.check_credit_card_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_high_discontinued_ratio" in names

    def test_duplicate_card_names(self, engine):
        df = pd.DataFrame(
            {
                "card_name": ["Card A", "Card A", "Card B"],
            }
        )
        anomalies = engine.check_credit_card_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_duplicate_card_names" in names

    def test_empty_dataframe(self, engine):
        df = pd.DataFrame(columns=["card_name", "annual_fee"])
        assert len(engine.check_credit_card_rules(df)) == 0


# =====================================================================
# Transaction rules
# =====================================================================


class TestTransactionRules:
    def test_clean_transactions_no_anomalies(self, engine):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "transaction_id": [f"txn_{i}" for i in range(n)],
                "user_id": [f"u{i % 20}" for i in range(n)],
                "date": pd.date_range("2025-01-01", periods=n, freq="D"),
                "amount": np.random.normal(100, 30, n).clip(1),
                "category": np.random.choice(
                    ["dining", "groceries", "gas", "travel"], n
                ),
            }
        )
        anomalies = engine.check_transaction_rules(df)
        critical = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]
        assert len(critical) == 0

    def test_high_value_concentration(self, engine):
        df = pd.DataFrame(
            {
                "transaction_id": [f"txn_{i}" for i in range(100)],
                "user_id": ["u1"] * 100,
                "amount": [50.0] * 95 + [50000.0] * 5,  # 5% > $10K
                "date": pd.date_range("2025-01-01", periods=100, freq="D"),
            }
        )
        anomalies = engine.check_transaction_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_high_value_concentration" in names

    def test_category_imbalance(self, engine):
        df = pd.DataFrame(
            {
                "transaction_id": [f"txn_{i}" for i in range(100)],
                "user_id": ["u1"] * 100,
                "category": ["dining"] * 80 + ["groceries"] * 20,  # 80% dining
                "date": pd.date_range("2025-01-01", periods=100, freq="D"),
            }
        )
        anomalies = engine.check_transaction_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_category_imbalance" in names

    def test_future_dates_flagged(self, engine):
        df = pd.DataFrame(
            {
                "transaction_id": ["txn_1"],
                "user_id": ["u1"],
                "date": [pd.Timestamp("2099-01-01")],
                "amount": [10.0],
            }
        )
        anomalies = engine.check_transaction_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_future_dates_present" in names

    def test_duplicate_transaction_ids(self, engine):
        df = pd.DataFrame(
            {
                "transaction_id": ["txn_1", "txn_1", "txn_2"],
                "user_id": ["u1", "u1", "u2"],
                "date": pd.date_range("2025-01-01", periods=3, freq="D"),
                "amount": [10.0, 20.0, 30.0],
            }
        )
        anomalies = engine.check_transaction_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_duplicate_transaction_ids" in names

    def test_empty_dataframe(self, engine):
        df = pd.DataFrame(columns=["transaction_id", "user_id", "amount"])
        assert len(engine.check_transaction_rules(df)) == 0


# =====================================================================
# User profile rules
# =====================================================================


class TestUserRules:
    def test_clean_users_no_anomalies(self, engine):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "archetype": [
                    "young_professional",
                    "suburban_family",
                    "budget_conscious",
                ],
                "monthly_budget": [3500, 7000, 2000],
            }
        )
        anomalies = engine.check_user_rules(df)
        assert len(anomalies) == 0

    def test_budget_archetype_mismatch(self, engine):
        df = pd.DataFrame(
            {
                "user_id": ["u1"],
                "archetype": ["minimal_user"],
                "monthly_budget": [50000.0],  # way too high for minimal_user
            }
        )
        anomalies = engine.check_user_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_budget_archetype_mismatch" in names

    def test_duplicate_user_ids(self, engine):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "archetype": [
                    "young_professional",
                    "young_professional",
                    "budget_conscious",
                ],
                "monthly_budget": [3500, 3500, 2000],
            }
        )
        anomalies = engine.check_user_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_duplicate_user_ids" in names
        dup = [a for a in anomalies if a.check_name == "domain_duplicate_user_ids"]
        assert dup[0].severity == AnomalySeverity.CRITICAL

    def test_invalid_archetype(self, engine):
        df = pd.DataFrame(
            {
                "user_id": ["u1"],
                "archetype": ["totally_made_up"],
                "monthly_budget": [3000],
            }
        )
        anomalies = engine.check_user_rules(df)
        names = [a.check_name for a in anomalies]
        assert "domain_invalid_archetype" in names

    def test_empty_dataframe(self, engine):
        df = pd.DataFrame(columns=["user_id", "archetype", "monthly_budget"])
        assert len(engine.check_user_rules(df)) == 0
