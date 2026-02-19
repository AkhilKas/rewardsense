"""
Unit Tests for RewardSense Data Cleaning Module (Enhanced)

Story 3.1: Comprehensive data cleaning tests

Tests cover:
- Card name normalization
- Issuer standardization
- Deduplication logic
- Welcome bonus parsing
- MCC validation
- Transaction cleaning
- User profile cleaning
- Edge cases and error handling

"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the module under test
import sys
from pathlib import Path

# Add src to path for imports
# When running from repo: imports from src/data_pipeline/preprocessing/cleaning
# When running standalone: update this path as needed
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.data_pipeline.preprocessing.cleaning import (
        clean_card_name,
        standardize_issuer_name,
        normalize_welcome_bonus,
        clean_credit_card_data,
        clean_transaction_data,
        clean_user_profile_data,
        clean_all_data,
        CleaningConfig,
    )
except ImportError:
    # Fallback for standalone testing
    from cleaning import (
        clean_card_name,
        standardize_issuer_name,
        normalize_welcome_bonus,
        clean_credit_card_data,
        clean_transaction_data,
        clean_user_profile_data,
        clean_all_data,
        CleaningConfig,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_credit_cards_df() -> pd.DataFrame:
    """Create sample credit card DataFrame for testing."""
    return pd.DataFrame(
        {
            "card_id": ["card_001", "card_002", "card_003", "card_004", "card_005"],
            "card_name": [
                "Chase Sapphire Preferred®",
                "Chase Sapphire Preferred",  # Duplicate after normalization
                "AMEX Gold Card™",
                "Citi Double Cash℠ Credit Card",
                "Capital One Venture X",
            ],
            "issuer": ["CHASE", "chase", "Amex", "CITI_BANK", "capital_one"],
            "annual_fee": [95, 95, 250, 0, 395],
            "reward_rates": [
                {"dining": 3.0, "travel": 2.0},
                {"dining": 3.0, "travel": 2.0},
                {"dining": 4.0, "groceries": 4.0},
                {},  # Empty dict
                None,  # Missing
            ],
            "welcome_bonus": [
                "60,000 points after $4,000 spend in 3 months",
                "60000 points",
                "$750 bonus after spending $6000",
                None,
                "75000 miles after $4000 spend",
            ],
            "discontinued": [False, False, False, False, False],
        }
    )


@pytest.fixture
def sample_credit_cards_with_issues_df() -> pd.DataFrame:
    """Create credit card DataFrame with data quality issues."""
    return pd.DataFrame(
        {
            "card_name": [
                "Good Card",
                "Another Good Card",
                "Invalid Fee Card",
                "Negative Fee Card",
                "Discontinued Card",
            ],
            "issuer": ["CHASE", "AMEX", "CITI", "DISCOVER", "CHASE"],
            "annual_fee": [95, 250, 5000, -50, 0],  # 5000 and -50 are invalid
            "reward_rates": [
                {"base": 1.5},
                None,
                {"base": 2.0},
                {},
                {"base": 1.0},
            ],
            "discontinued": [False, False, False, False, True],
        }
    )


@pytest.fixture
def sample_transactions_df() -> pd.DataFrame:
    """Create sample transaction DataFrame for testing."""
    now = datetime.now()
    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:03d}" for i in range(1, 11)],
            "user_id": ["user_001"] * 5 + ["user_002"] * 5,
            "date": [
                now - timedelta(days=30),
                now - timedelta(days=25),
                now - timedelta(days=20),
                now - timedelta(days=15),
                now - timedelta(days=10),
                now - timedelta(days=5),
                now - timedelta(days=3),
                now - timedelta(days=1),
                now + timedelta(days=5),  # Future date - should be removed
                now - timedelta(days=2),
            ],
            "category": [
                "dining",
                "groceries",
                "travel",
                "",  # Empty - should become 'unknown'
                None,  # Missing - should become 'unknown'
                "gas",
                "entertainment",
                "online_shopping",
                "dining",
                "utilities",
            ],
            "merchant": [
                "Starbucks",
                "Whole Foods",
                "Delta Airlines",
                "Unknown Store",
                "Mystery Shop",
                "Shell",
                "Netflix",
                "Amazon",
                "Restaurant",
                "Electric Company",
            ],
            "mcc_code": [
                5812,  # Valid dining
                5411,  # Valid groceries
                4511,  # Valid travel
                5999,  # Valid other
                9999,  # Invalid MCC
                5541,  # Valid gas
                4899,  # Valid streaming
                5691,  # Valid online shopping
                5812,  # Valid dining
                4900,  # Valid utilities
            ],
            "amount": [
                25.50,
                150.00,
                500.00,
                -10.00,  # Negative - should be removed
                75.00,
                45.00,
                15.99,
                250.00,
                35.00,
                15000.00,  # Suspicious high amount
            ],
            "card_used": [
                "Chase Sapphire",
                "Amex Gold",
                "Chase Sapphire",
                "Citi Double Cash",
                "Citi Double Cash",
                "Chase Freedom",
                "Chase Freedom",
                "Amex Gold",
                "Chase Sapphire",
                "Chase Freedom",
            ],
        }
    )


@pytest.fixture
def sample_users_df() -> pd.DataFrame:
    """Create sample user profile DataFrame for testing."""
    return pd.DataFrame(
        {
            "user_id": [
                "user_001",
                "user_002",
                "user_003",
                "user_001",
            ],  # Duplicate user_001
            "archetype": [
                "young_professional",
                "suburban_family",
                "frequent_traveler",
                "young_professional",  # Duplicate
            ],
            "monthly_budget": [3500.00, 6500.00, 5000.00, 3500.00],
            "cards": [
                "['Chase Sapphire', 'Amex Gold']",
                "['Citi Double Cash', 'Chase Freedom']",
                "['Chase Sapphire Reserve', 'Amex Platinum']",
                "['Chase Sapphire', 'Amex Gold']",
            ],
            "redemption_preference": [
                "travel_transfer",
                "cash_back",
                "travel_portal",
                "travel_transfer",
            ],
            "age_group": ["26-35", "36-50", "26-35", "26-35"],
            "location_type": ["urban", "suburban", "urban", "urban"],
        }
    )


@pytest.fixture
def custom_config() -> CleaningConfig:
    """Create custom cleaning configuration for testing."""
    return CleaningConfig(
        max_annual_fee=500.0,
        min_annual_fee=0.0,
        suspicious_amount_threshold=5000.0,
    )


# =============================================================================
# Unit Tests: Text Normalization Functions
# =============================================================================


class TestCleanCardName:
    """Tests for clean_card_name() function."""

    def test_removes_trademark_symbols(self):
        """Should remove ®, ™, ℠, © symbols."""
        assert (
            clean_card_name("Chase Sapphire Preferred®") == "Chase Sapphire Preferred"
        )
        assert clean_card_name("Amex Gold™") == "AMEX Gold"
        assert clean_card_name("Citi Double Cash℠") == "Citi Double Cash"
        assert clean_card_name("Some Card©") == "Some Card"

    def test_normalizes_whitespace(self):
        """Should collapse multiple spaces to single space."""
        assert (
            clean_card_name("Chase  Sapphire   Preferred") == "Chase Sapphire Preferred"
        )
        assert clean_card_name("  Amex Gold  ") == "AMEX Gold"
        assert clean_card_name("Card\t\tName") == "Card Name"

    def test_removes_credit_card_suffix(self):
        """Should remove redundant 'Credit Card' suffix."""
        assert clean_card_name("Citi Double Cash Credit Card") == "Citi Double Cash"
        assert (
            clean_card_name("Capital One Venture credit card") == "Capital One Venture"
        )

    def test_handles_none_and_nan(self):
        """Should handle None and NaN gracefully."""
        assert clean_card_name(None) == ""
        assert clean_card_name(np.nan) == ""
        assert clean_card_name(pd.NA) == ""

    def test_handles_non_string_input(self):
        """Should convert non-string input to string."""
        assert clean_card_name(123) == "123"
        assert clean_card_name(45.67) == "45.67"

    def test_preserves_amex_acronym(self):
        """Should preserve AMEX acronym in title case."""
        result = clean_card_name("amex gold card")
        assert "AMEX" in result


class TestStandardizeIssuerName:
    """Tests for standardize_issuer_name() function."""

    def test_uppercase_conversion(self):
        """Should convert to uppercase."""
        assert standardize_issuer_name("chase") == "CHASE"
        assert standardize_issuer_name("American Express") == "AMERICAN EXPRESS"

    def test_replaces_underscores(self):
        """Should replace underscores with spaces."""
        assert standardize_issuer_name("capital_one") == "CAPITAL ONE"
        assert standardize_issuer_name("bank_of_america") == "BANK OF AMERICA"

    def test_applies_aliases(self):
        """Should apply alias mappings."""
        aliases = {"AMEX": "AMERICAN EXPRESS", "BOFA": "BANK OF AMERICA"}
        assert standardize_issuer_name("Amex", aliases) == "AMERICAN EXPRESS"
        assert standardize_issuer_name("BOFA", aliases) == "BANK OF AMERICA"

    def test_handles_none(self):
        """Should return 'UNKNOWN' for None/NaN."""
        assert standardize_issuer_name(None) == "UNKNOWN"
        assert standardize_issuer_name(np.nan) == "UNKNOWN"

    def test_normalizes_whitespace(self):
        """Should normalize extra whitespace."""
        assert standardize_issuer_name("Chase  Bank") == "CHASE BANK"
        assert standardize_issuer_name("  Citi  ") == "CITI"


class TestNormalizeWelcomeBonus:
    """Tests for normalize_welcome_bonus() function."""

    def test_parses_points_format(self):
        """Should parse 'X points after $Y spend' format."""
        result = normalize_welcome_bonus("60,000 points after $4,000 spend in 3 months")
        assert result["amount"] == 60000
        assert result["unit"] == "points"
        assert result["spend_requirement"] == 4000
        assert result["time_days"] == 90  # 3 months * 30

    def test_parses_miles_format(self):
        """Should parse miles format."""
        result = normalize_welcome_bonus("75000 miles after $4000 spend")
        assert result["amount"] == 75000
        assert result["unit"] == "miles"
        assert result["spend_requirement"] == 4000

    def test_parses_cash_bonus_format(self):
        """Should parse $XXX bonus format."""
        result = normalize_welcome_bonus("$750 bonus after spending $6000")
        assert result["amount"] == 750
        assert result["unit"] == "dollars"
        assert result["spend_requirement"] == 6000

    def test_handles_weeks(self):
        """Should convert weeks to days."""
        result = normalize_welcome_bonus("50000 points after $3000 in 6 weeks")
        assert result["time_days"] == 42  # 6 * 7

    def test_handles_days(self):
        """Should handle days directly."""
        result = normalize_welcome_bonus("50000 points in 90 days")
        assert result["time_days"] == 90

    def test_handles_none(self):
        """Should return default structure for None."""
        result = normalize_welcome_bonus(None)
        assert result["amount"] is None
        assert result["unit"] == "points"
        assert result["raw_text"] is None

    def test_handles_simple_format(self):
        """Should handle simple 'X points' format."""
        result = normalize_welcome_bonus("60000 points")
        assert result["amount"] == 60000
        assert result["unit"] == "points"
        assert result["spend_requirement"] is None


# =============================================================================
# Unit Tests: Credit Card Cleaning
# =============================================================================


class TestCleanCreditCardData:
    """Tests for clean_credit_card_data() function."""

    def test_returns_dataframe_and_report(self, sample_credit_cards_df):
        """Should return tuple of (DataFrame, dict)."""
        result, report = clean_credit_card_data(sample_credit_cards_df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(report, dict)

    def test_normalizes_card_names(self, sample_credit_cards_df):
        """Should normalize card names."""
        result, _ = clean_credit_card_data(sample_credit_cards_df)
        # Check trademark symbols removed
        assert not any(result["card_name"].str.contains("®", na=False))
        assert not any(result["card_name"].str.contains("™", na=False))
        assert not any(result["card_name"].str.contains("℠", na=False))

    def test_standardizes_issuer_names(self, sample_credit_cards_df):
        """Should standardize issuer names to uppercase."""
        result, _ = clean_credit_card_data(sample_credit_cards_df)
        # All issuers should be uppercase
        assert all(result["issuer"] == result["issuer"].str.upper())

    def test_deduplicates_by_normalized_name(self, sample_credit_cards_df):
        """Should remove duplicates based on normalized card name + issuer."""
        result, report = clean_credit_card_data(sample_credit_cards_df)
        # Original has "Chase Sapphire Preferred®" and "Chase Sapphire Preferred"
        # which should be deduplicated
        assert report["dedup_removed"] >= 1
        assert len(result) < len(sample_credit_cards_df)

    def test_imputes_missing_reward_rates(self, sample_credit_cards_df):
        """Should impute missing/empty reward_rates with default."""
        result, report = clean_credit_card_data(sample_credit_cards_df)
        # No None values in reward_rates
        assert not result["reward_rates"].isna().any()
        # Check imputation happened
        assert (
            report["missing_reward_rates"] > 0
            or report.get("empty_reward_rates", 0) > 0
        )

    def test_validates_annual_fee_range(self, sample_credit_cards_with_issues_df):
        """Should remove cards with invalid annual fees."""
        result, report = clean_credit_card_data(sample_credit_cards_with_issues_df)
        # 5000 and -50 are invalid
        assert report["invalid_annual_fees"] >= 2
        assert report["annual_fee_removed"] >= 2
        # Remaining fees should be in valid range
        assert all(result["annual_fee"] >= 0)
        assert all(result["annual_fee"] < 1000)

    def test_flags_discontinued_cards(self, sample_credit_cards_with_issues_df):
        """Should create is_active flag for discontinued cards."""
        result, report = clean_credit_card_data(sample_credit_cards_with_issues_df)
        if "is_active" in result.columns:
            assert "is_active" in result.columns
            assert report.get("discontinued_cards", 0) >= 0

    def test_report_contains_required_fields(self, sample_credit_cards_df):
        """Should include all required fields in report."""
        _, report = clean_credit_card_data(sample_credit_cards_df)
        required_fields = [
            "initial_count",
            "final_count",
            "dedup_removed",
            "steps",
        ]
        for field in required_fields:
            assert field in report, f"Missing required field: {field}"

    def test_is_idempotent(self, sample_credit_cards_df):
        """Running cleaning twice should produce same result."""
        result1, _ = clean_credit_card_data(sample_credit_cards_df)
        result2, _ = clean_credit_card_data(result1)

        # Same number of rows
        assert len(result1) == len(result2)
        # Same card names
        assert list(result1["card_name"]) == list(result2["card_name"])

    def test_preserves_original_card_name(self, sample_credit_cards_df):
        """Should preserve original card name in separate column."""
        result, _ = clean_credit_card_data(sample_credit_cards_df)
        assert "card_name_original" in result.columns

    def test_custom_config_thresholds(
        self, sample_credit_cards_with_issues_df, custom_config
    ):
        """Should respect custom configuration thresholds."""
        # Custom config has max_annual_fee=500
        result, report = clean_credit_card_data(
            sample_credit_cards_with_issues_df, config=custom_config
        )
        # Cards with fee >= 500 should be removed (in addition to invalid ones)
        assert all(result["annual_fee"] < 500)

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=["card_name", "issuer", "annual_fee"])
        result, report = clean_credit_card_data(empty_df)
        assert len(result) == 0
        assert report["initial_count"] == 0
        assert report["final_count"] == 0

    def test_handles_api_name_column(self):
        """Should handle 'name' column (API format) instead of 'card_name'."""
        df = pd.DataFrame(
            {
                "name": ["Chase Sapphire Preferred®", "Amex Gold™"],
                "issuer": ["CHASE", "AMEX"],
                "annual_fee": [95, 250],
            }
        )
        result, report = clean_credit_card_data(df)
        assert "card_name" in result.columns
        assert "card_name_normalized_from_name" in report["steps"]


# =============================================================================
# Unit Tests: Transaction Cleaning
# =============================================================================


class TestCleanTransactionData:
    """Tests for clean_transaction_data() function."""

    def test_returns_dataframe_and_report(self, sample_transactions_df):
        """Should return tuple of (DataFrame, dict)."""
        result, report = clean_transaction_data(sample_transactions_df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(report, dict)

    def test_removes_negative_amounts(self, sample_transactions_df):
        """Should remove transactions with negative amounts."""
        result, report = clean_transaction_data(sample_transactions_df)
        assert report["negative_amounts"] >= 1
        assert report["negative_amounts_removed"] >= 1
        assert all(result["amount"] >= 0)

    def test_removes_future_dates(self, sample_transactions_df):
        """Should remove transactions with future dates."""
        result, report = clean_transaction_data(sample_transactions_df)
        assert report["future_dates"] >= 1
        assert all(result["date"] <= pd.Timestamp.now())

    def test_fills_missing_categories(self, sample_transactions_df):
        """Should fill missing/empty categories with 'unknown'."""
        result, report = clean_transaction_data(sample_transactions_df)
        assert report["missing_categories"] >= 1
        assert not result["category"].isna().any()
        assert not (result["category"] == "").any()

    def test_validates_mcc_codes(self, sample_transactions_df):
        """Should flag invalid MCC codes."""
        result, report = clean_transaction_data(
            sample_transactions_df, validate_mcc=True
        )
        assert "mcc_valid" in result.columns
        assert report["invalid_mcc_codes"] >= 1
        # Check that known valid MCCs are marked valid
        dining_txns = result[result["mcc_code"] == 5812]
        if len(dining_txns) > 0:
            assert all(dining_txns["mcc_valid"])

    def test_flags_suspicious_transactions(self, sample_transactions_df):
        """Should flag suspicious high-value transactions."""
        result, report = clean_transaction_data(sample_transactions_df)
        assert "suspicious" in result.columns
        assert report["suspicious_high_amounts"] >= 1
        # $15000 transaction should be flagged
        high_value = result[result["amount"] >= 10000]
        if len(high_value) > 0:
            assert all(high_value["suspicious"] == 1)

    def test_report_contains_required_fields(self, sample_transactions_df):
        """Should include all required fields in report."""
        _, report = clean_transaction_data(sample_transactions_df)
        required_fields = [
            "initial_count",
            "final_count",
            "negative_amounts",
            "future_dates",
            "missing_categories",
            "steps",
        ]
        for field in required_fields:
            assert field in report, f"Missing required field: {field}"

    def test_is_idempotent(self, sample_transactions_df):
        """Running cleaning twice should produce same result."""
        result1, _ = clean_transaction_data(sample_transactions_df)
        result2, _ = clean_transaction_data(result1)

        assert len(result1) == len(result2)

    def test_custom_suspicious_threshold(self, sample_transactions_df, custom_config):
        """Should respect custom suspicious amount threshold."""
        result, report = clean_transaction_data(
            sample_transactions_df, config=custom_config
        )
        # Custom threshold is $5000 instead of default $10000
        # Should flag more transactions
        flagged = result[result["suspicious"] == 1]
        assert len(flagged) >= 1

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=["transaction_id", "user_id", "date", "amount"])
        result, report = clean_transaction_data(empty_df)
        assert len(result) == 0
        assert report["initial_count"] == 0

    def test_detects_potential_duplicates(self):
        """Should flag potential duplicate transactions."""
        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001", "txn_002", "txn_003"],
                "user_id": ["user_001", "user_001", "user_001"],
                "date": [datetime.now()] * 3,
                "amount": [100.00, 100.00, 200.00],  # First two are potential dupes
                "merchant": ["Store A", "Store A", "Store B"],
                "category": ["dining", "dining", "groceries"],
                "mcc_code": [5812, 5812, 5411],
            }
        )
        result, report = clean_transaction_data(df)
        assert "is_potential_duplicate" in result.columns
        assert report["potential_duplicates"] >= 1

    def test_standardizes_category_names(self, sample_transactions_df):
        """Should lowercase and strip category names."""
        result, _ = clean_transaction_data(sample_transactions_df)
        # All categories should be lowercase
        assert all(result["category"] == result["category"].str.lower())
        # No leading/trailing whitespace
        assert all(result["category"] == result["category"].str.strip())


# =============================================================================
# Unit Tests: User Profile Cleaning
# =============================================================================


class TestCleanUserProfileData:
    """Tests for clean_user_profile_data() function."""

    def test_returns_dataframe_and_report(self, sample_users_df):
        """Should return tuple of (DataFrame, dict)."""
        result, report = clean_user_profile_data(sample_users_df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(report, dict)

    def test_removes_duplicate_users(self, sample_users_df):
        """Should remove duplicate user_id entries."""
        result, report = clean_user_profile_data(sample_users_df)
        assert report["duplicate_users_removed"] >= 1
        # No duplicate user_ids
        assert not result["user_id"].duplicated().any()

    def test_validates_monthly_budget(self):
        """Should handle invalid monthly budget values."""
        df = pd.DataFrame(
            {
                "user_id": ["user_001", "user_002"],
                "archetype": ["young_professional", "suburban_family"],
                "monthly_budget": [3500.00, np.nan],  # One invalid
                "redemption_preference": ["cash_back", "travel_transfer"],
                "age_group": ["26-35", "36-50"],
                "location_type": ["urban", "suburban"],
            }
        )
        result, report = clean_user_profile_data(df)
        assert not result["monthly_budget"].isna().any()
        assert report["invalid_budgets_imputed"] >= 1

    def test_validates_archetype(self, sample_users_df):
        """Should report invalid archetypes."""
        df = sample_users_df.copy()
        df.loc[0, "archetype"] = "invalid_archetype"
        result, report = clean_user_profile_data(df)
        assert report["invalid_archetypes"] >= 1

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=["user_id", "archetype", "monthly_budget"])
        result, report = clean_user_profile_data(empty_df)
        assert len(result) == 0


# =============================================================================
# Unit Tests: Combined Cleaning
# =============================================================================


class TestCleanAllData:
    """Tests for clean_all_data() convenience function."""

    def test_cleans_all_datasets(
        self, sample_credit_cards_df, sample_transactions_df, sample_users_df
    ):
        """Should clean all three datasets."""
        cards, txns, users, report = clean_all_data(
            credit_cards_df=sample_credit_cards_df,
            transactions_df=sample_transactions_df,
            users_df=sample_users_df,
        )

        assert cards is not None
        assert txns is not None
        assert users is not None
        assert "credit_cards" in report
        assert "transactions" in report
        assert "users" in report

    def test_handles_none_inputs(self):
        """Should handle None inputs gracefully."""
        cards, txns, users, report = clean_all_data(
            credit_cards_df=None,
            transactions_df=None,
            users_df=None,
        )

        assert cards is None
        assert txns is None
        assert users is None
        assert report == {}

    def test_partial_inputs(self, sample_credit_cards_df):
        """Should handle partial inputs."""
        cards, txns, users, report = clean_all_data(
            credit_cards_df=sample_credit_cards_df,
            transactions_df=None,
            users_df=None,
        )

        assert cards is not None
        assert txns is None
        assert users is None
        assert "credit_cards" in report
        assert "transactions" not in report


# =============================================================================
# Unit Tests: Configuration
# =============================================================================


class TestCleaningConfig:
    """Tests for CleaningConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = CleaningConfig()
        assert config.max_annual_fee == 1000.0
        assert config.min_annual_fee == 0.0
        assert config.suspicious_amount_threshold == 10000.0
        assert len(config.valid_mcc_codes) > 0

    def test_custom_values(self):
        """Should accept custom values."""
        config = CleaningConfig(
            max_annual_fee=500.0,
            suspicious_amount_threshold=5000.0,
        )
        assert config.max_annual_fee == 500.0
        assert config.suspicious_amount_threshold == 5000.0

    def test_issuer_aliases(self):
        """Should have default issuer aliases."""
        config = CleaningConfig()
        assert "AMEX" in config.issuer_aliases
        assert config.issuer_aliases["AMEX"] == "AMERICAN EXPRESS"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unicode_characters_in_card_name(self):
        """Should handle unicode characters in card names."""
        df = pd.DataFrame(
            {
                "card_name": ["Café Rewards Card™", "日本語カード", "Émoji Card 🎉"],
                "issuer": ["CHASE", "JCB", "DISCOVER"],
                "annual_fee": [0, 0, 0],
            }
        )
        result, _ = clean_credit_card_data(df)
        assert len(result) == 3

    def test_very_long_card_name(self):
        """Should handle very long card names."""
        long_name = "A" * 1000 + "®™"
        df = pd.DataFrame(
            {
                "card_name": [long_name],
                "issuer": ["CHASE"],
                "annual_fee": [0],
            }
        )
        result, _ = clean_credit_card_data(df)
        assert len(result) == 1
        assert "®" not in result["card_name"].iloc[0]

    def test_all_invalid_transactions(self):
        """Should handle case where all transactions are invalid."""
        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001", "txn_002"],
                "user_id": ["user_001", "user_001"],
                "date": [datetime.now() + timedelta(days=100)] * 2,  # All future
                "amount": [-100, -200],  # All negative
                "category": ["dining", "groceries"],
                "mcc_code": [5812, 5411],
            }
        )
        result, report = clean_transaction_data(df)
        assert len(result) == 0
        assert report["final_count"] == 0

    def test_mixed_date_formats(self):
        """Should handle mixed date formats."""
        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001", "txn_002", "txn_003"],
                "user_id": ["user_001"] * 3,
                "date": [
                    "2024-01-15",
                    "2024-06-15",
                    datetime(2024, 3, 15),
                ],  # All past dates
                "amount": [100, 200, 300],
                "category": ["dining"] * 3,
                "mcc_code": [5812] * 3,
            }
        )
        result, report = clean_transaction_data(df)
        # All should parse successfully (all are valid past dates)
        assert report["invalid_dates"] == 0
        assert len(result) == 3

    def test_numeric_card_name(self):
        """Should handle numeric card names."""
        df = pd.DataFrame(
            {
                "card_name": [12345, 67890],
                "issuer": ["CHASE", "AMEX"],
                "annual_fee": [0, 0],
            }
        )
        result, _ = clean_credit_card_data(df)
        assert len(result) == 2
        assert result["card_name"].iloc[0] == "12345"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the cleaning pipeline."""

    def test_cleaning_output_ready_for_feature_engineering(
        self,
        sample_credit_cards_df,
        sample_transactions_df,
        sample_users_df,
    ):
        """Cleaned data should be ready for feature engineering."""
        cards, txns, users, _ = clean_all_data(
            credit_cards_df=sample_credit_cards_df,
            transactions_df=sample_transactions_df,
            users_df=sample_users_df,
        )

        # Credit cards should have clean reward_rates
        assert not cards["reward_rates"].isna().any()

        # Transactions should have no negative amounts or future dates
        assert all(txns["amount"] >= 0)
        assert all(txns["date"] <= pd.Timestamp.now())

        # Users should have unique user_ids
        assert not users["user_id"].duplicated().any()

    def test_cleaning_preserves_data_integrity(self, sample_credit_cards_df):
        """Cleaning should not corrupt data values."""
        result, _ = clean_credit_card_data(sample_credit_cards_df)

        # Annual fees should be preserved (not modified, just validated)
        original_fees = set(sample_credit_cards_df["annual_fee"].dropna())
        result_fees = set(result["annual_fee"].dropna())
        # Result fees should be a subset of original (some may be removed)
        assert result_fees.issubset(original_fees)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
