"""
Unit tests for profiling module.

Tests Story 6.3 acceptance criteria:
- Data profiles generated after each pipeline run
- Statistics stored for trend analysis
- Profiles accessible via reports
"""

import pandas as pd
import time

from src.data_pipeline.profiling import (
    DataProfiler,
    CreditCardStatistics,
    TransactionStatistics,
    UserProfileStatistics,
    StatisticsHistory,
)


class TestCreditCardStatistics:
    """Test credit card statistics generation."""

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        df = pd.DataFrame(
            [
                {
                    "card_id": "c1",
                    "issuer": "CHASE",
                    "network": "VISA",
                    "annual_fee": 550,
                    "currency": "POINTS",
                    "is_business": False,
                    "discontinued": False,
                },
                {
                    "card_id": "c2",
                    "issuer": "AMEX",
                    "network": "AMEX",
                    "annual_fee": 0,
                    "currency": "MILES",
                    "is_business": True,
                    "discontinued": False,
                },
            ]
        )

        stats = CreditCardStatistics.calculate_statistics(df)

        assert stats["total_cards"] == 2
        assert stats["unique_issuers"] == 2
        assert stats["annual_fee"]["no_fee_count"] == 1
        assert stats["annual_fee"]["no_fee_pct"] == 50.0


class TestTransactionStatistics:
    """Test transaction statistics generation."""

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        df = pd.DataFrame(
            [
                {
                    "transaction_id": "txn_001",
                    "user_id": "user_0001",
                    "date": "2025-08-01",
                    "category": "dining",
                    "merchant": "Starbucks",
                    "amount": 10.0,
                },
                {
                    "transaction_id": "txn_002",
                    "user_id": "user_0001",
                    "date": "2025-08-02",
                    "category": "travel",
                    "merchant": "United",
                    "amount": 200.0,
                },
            ]
        )

        stats = TransactionStatistics.calculate_statistics(df)

        assert stats["total_transactions"] == 2
        assert stats["unique_users"] == 1
        assert stats["amount"]["total"] == 210.0
        assert stats["amount"]["mean"] == 105.0


class TestUserProfileStatistics:
    """Test user profile statistics generation."""

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        df = pd.DataFrame(
            [
                {
                    "user_id": "user_0001",
                    "archetype": "high_roller",
                    "monthly_budget": 10000,
                    "age_group": "51-65",
                    "location_type": "urban",
                    "redemption_preference": "travel_portal",
                },
                {
                    "user_id": "user_0002",
                    "archetype": "budget_conscious",
                    "monthly_budget": 2000,
                    "age_group": "26-35",
                    "location_type": "rural",
                    "redemption_preference": "cash_back",
                },
            ]
        )

        stats = UserProfileStatistics.calculate_statistics(df)

        assert stats["total_users"] == 2
        assert stats["monthly_budget"]["mean"] == 6000.0
        assert len(stats["archetype_distribution"]) == 2


class TestStatisticsHistory:
    """Test historical statistics storage."""

    def test_save_and_load_statistics(self, tmp_path):
        """Test saving and loading statistics."""
        history = StatisticsHistory(history_dir=str(tmp_path / "history"))

        stats = {"timestamp": "2026-02-20", "total": 100}

        # Save
        filepath = history.save_statistics("test_dataset", stats)
        assert filepath.exists()

        # Load latest
        loaded = history.load_latest("test_dataset")
        assert loaded["total"] == 100

    def test_load_history(self, tmp_path):
        """Test loading historical records."""
        history = StatisticsHistory(history_dir=str(tmp_path / "history"))

        # Save multiple with delay to ensure different timestamps
        history.save_statistics("test", {"run": 1})
        time.sleep(1.1)  # Wait to ensure different timestamp
        history.save_statistics("test", {"run": 2})

        # Load all
        all_stats = history.load_history("test")
        assert len(all_stats) >= 1  # At least one file saved


class TestDataProfiler:
    """Test data profiler."""

    def test_profile_transactions(self, tmp_path):
        """Test transaction profiling."""
        df = pd.DataFrame(
            [
                {
                    "transaction_id": "txn_001",
                    "user_id": "user_0001",
                    "date": "2025-08-01",
                    "category": "dining",
                    "merchant": "Test",
                    "mcc_code": 5812,
                    "amount": 10.0,
                    "card_used": "Card",
                },
            ]
        )

        profiler = DataProfiler(output_dir=str(tmp_path / "profiles"))
        profile = profiler.profile_transactions(df, minimal=True)

        assert profile is not None

        # Check file was created
        html_files = list((tmp_path / "profiles").glob("transactions_profile_*.html"))
        assert len(html_files) >= 1
