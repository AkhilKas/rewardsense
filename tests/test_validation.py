"""Tests for Great Expectations validation integration."""

import pandas as pd

from src.data_pipeline.validation.validator import DataValidator, validate_all_data


class TestDataValidator:
    """Test the DataValidator class."""

    def test_validate_transactions_valid_data(self):
        """Test validation passes for valid transaction data."""
        df = pd.DataFrame(
            [
                {
                    "transaction_id": "txn_0000001",
                    "user_id": "user_0001",
                    "date": "2025-08-01",
                    "category": "dining",
                    "merchant": "Starbucks",
                    "mcc_code": 5812,
                    "amount": 16.09,
                    "card_used": "Chase Sapphire Reserve",
                }
            ]
        )

        validator = DataValidator()
        success, results = validator.validate_transactions(df)

        assert success is True
        assert "statistics" in results

    def test_validate_users_valid_data(self):
        """Test validation passes for valid user data."""
        df = pd.DataFrame(
            [
                {
                    "user_id": "user_0001",
                    "archetype": "high_roller",
                    "monthly_budget": 13156.94,
                    "cards": "['Chase Sapphire Reserve']",
                    "redemption_preference": "travel_portal",
                    "age_group": "51-65",
                    "location_type": "urban",
                }
            ]
        )

        validator = DataValidator()
        success, results = validator.validate_user_profiles(df)

        assert success is True


class TestValidateAllData:
    """Test the convenience function."""

    def test_validate_all_data(self):
        """Test validating multiple datasets."""
        txns = pd.DataFrame(
            [
                {
                    "transaction_id": "txn_0000001",
                    "user_id": "user_0001",
                    "date": "2025-08-01",
                    "category": "dining",
                    "merchant": "Test",
                    "mcc_code": 5812,
                    "amount": 10.0,
                    "card_used": "Card",
                }
            ]
        )

        users = pd.DataFrame(
            [
                {
                    "user_id": "user_0001",
                    "archetype": "high_roller",
                    "monthly_budget": 5000,
                    "cards": "['Card']",
                    "redemption_preference": "cash_back",
                    "age_group": "26-35",
                    "location_type": "urban",
                }
            ]
        )

        results = validate_all_data(transactions_df=txns, users_df=users)

        assert "transactions" in results
        assert "users" in results
        assert results["transactions"] is True
        assert results["users"] is True
