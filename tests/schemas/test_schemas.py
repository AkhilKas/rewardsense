"""
Unit tests for RewardSense schemas.

Tests Story 6.1 acceptance criteria:
- All data artifacts have defined schemas
- Schemas are version controlled
- Schema validation works correctly

Coverage: All 10 schema classes and all 7 shared validators.
"""

import pytest
from pydantic import ValidationError

from schemas import (
    # Credit card schemas
    CreditCardCleaned,
    CreditCardFeatures,
    # Transaction schemas
    TransactionRaw,
    TransactionCleaned,
    TransactionFeatures,
    # User profile schemas
    UserProfileRaw,
    UserCardMapping,
    UserProfileFeatures,
    # Feature metadata
    FeatureMetadata,
    FeatureRegistry,
    # Validators
    validate_user_id_format,
    validate_transaction_id_format,
    validate_category,
    validate_mcc_code,
    validate_amount_positive,
    validate_redemption_preference,
    validate_archetype,
)


# =============================================================================
# Transaction Schemas
# =============================================================================


class TestTransactionRaw:
    """Test TransactionRaw schema."""

    def test_valid(self):
        """Test valid transaction passes validation."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
            "amount": 16.09,
            "card_used": "Chase Sapphire Reserve",
        }

        txn = TransactionRaw(**data)
        assert txn.transaction_id == "txn_0000123"
        assert txn.amount == 16.09

    def test_invalid_user_id(self):
        """Test invalid user_id format raises error."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "invalid_123",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
            "amount": 16.09,
            "card_used": "Card",
        }

        with pytest.raises(ValidationError):
            TransactionRaw(**data)

    def test_negative_amount(self):
        """Test negative amount raises error."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
            "amount": -10.0,
            "card_used": "Card",
        }

        with pytest.raises(ValidationError):
            TransactionRaw(**data)

    def test_zero_amount(self):
        """Test zero amount raises error."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
            "amount": 0,
            "card_used": "Card",
        }

        with pytest.raises(ValidationError):
            TransactionRaw(**data)

    def test_invalid_category(self):
        """Test invalid category raises error."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "nonexistent_category",
            "merchant": "Store",
            "mcc_code": 5812,
            "amount": 10.0,
            "card_used": "Card",
        }

        with pytest.raises(ValidationError):
            TransactionRaw(**data)

    def test_invalid_mcc_code(self):
        """Test invalid MCC code raises error."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 999,
            "amount": 16.09,
            "card_used": "Card",
        }

        with pytest.raises(ValidationError):
            TransactionRaw(**data)

    def test_missing_required_field(self):
        """Test missing required field raises error."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
            # amount missing
            "card_used": "Card",
        }

        with pytest.raises(ValidationError):
            TransactionRaw(**data)


class TestTransactionCleaned:
    """Test TransactionCleaned schema."""

    def test_has_suspicious_flag(self):
        """Test cleaned transaction includes suspicious flag."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
            "amount": 16.09,
            "card_used": "Card",
            "suspicious": False,
        }

        txn = TransactionCleaned(**data)
        assert hasattr(txn, "suspicious")
        assert txn.suspicious is False

    def test_suspicious_default(self):
        """Test suspicious defaults to False when not provided."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
            "amount": 16.09,
            "card_used": "Card",
        }

        txn = TransactionCleaned(**data)
        assert txn.suspicious is False

    def test_suspicious_true(self):
        """Test suspicious can be set to True."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "user_0001",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
            "amount": 15000.0,
            "card_used": "Card",
            "suspicious": True,
        }

        txn = TransactionCleaned(**data)
        assert txn.suspicious is True

    def test_inherits_raw_validation(self):
        """Test cleaned schema inherits Raw validation (e.g. user_id)."""
        data = {
            "transaction_id": "txn_0000123",
            "user_id": "bad_format",
            "date": "2025-08-01",
            "category": "dining",
            "merchant": "Store",
            "mcc_code": 5812,
            "amount": 10.0,
            "card_used": "Card",
            "suspicious": False,
        }

        with pytest.raises(ValidationError):
            TransactionCleaned(**data)


class TestTransactionFeatures:
    """Test TransactionFeatures schema."""

    @pytest.fixture()
    def valid_features(self):
        """Minimal valid TransactionFeatures data."""
        return {
            "user_id": "user_0001",
            "total_spending": 5000.0,
            "total_transactions": 100.0,
            "weekend_spending_ratio": 0.3,
            "peak_spending_month": 12,
            "peak_spending_day": 5,
            "avg_transaction_amount": 50.0,
            "transaction_amount_std": 25.0,
            "median_transaction_amount": 40.0,
            "total_spending_temporal": 5000.0,
            "num_cards_used": 3,
            "card_switch_rate": 0.4,
            "num_unique_mccs": 8,
            "avg_spending_per_mcc": 625.0,
            "num_unique_merchants": 20,
            "repeat_merchant_ratio": 0.6,
        }

    def test_valid(self, valid_features):
        """Test valid features pass validation."""
        feat = TransactionFeatures(**valid_features)
        assert feat.user_id == "user_0001"
        assert feat.total_spending == 5000.0
        assert feat.num_cards_used == 3

    def test_category_spending_defaults(self, valid_features):
        """Test category spending fields default to 0."""
        feat = TransactionFeatures(**valid_features)
        assert feat.dining_total_spent == 0
        assert feat.travel_total_spent == 0
        assert feat.groceries_total_spent == 0

    def test_with_category_spending(self, valid_features):
        """Test with explicit category spending values."""
        valid_features["dining_total_spent"] = 1200.0
        valid_features["travel_total_spent"] = 800.0
        feat = TransactionFeatures(**valid_features)
        assert feat.dining_total_spent == 1200.0
        assert feat.travel_total_spent == 800.0

    def test_weekend_ratio_bounds(self, valid_features):
        """Test weekend_spending_ratio must be 0-1."""
        valid_features["weekend_spending_ratio"] = 1.5
        with pytest.raises(ValidationError):
            TransactionFeatures(**valid_features)

    def test_peak_month_bounds(self, valid_features):
        """Test peak_spending_month must be 1-12."""
        valid_features["peak_spending_month"] = 13
        with pytest.raises(ValidationError):
            TransactionFeatures(**valid_features)

    def test_card_switch_rate_bounds(self, valid_features):
        """Test card_switch_rate must be 0-1."""
        valid_features["card_switch_rate"] = -0.1
        with pytest.raises(ValidationError):
            TransactionFeatures(**valid_features)

    def test_allows_extra_fields(self, valid_features):
        """Test extra category columns are allowed (extra='allow')."""
        valid_features["gas_total_spent"] = 500.0
        valid_features["streaming_total_spent"] = 120.0
        feat = TransactionFeatures(**valid_features)
        assert feat.gas_total_spent == 500.0
        assert feat.streaming_total_spent == 120.0

    def test_optional_suspicious_fields(self, valid_features):
        """Test optional suspicious fields."""
        valid_features["num_suspicious"] = 2
        valid_features["suspicious_rate"] = 0.02
        feat = TransactionFeatures(**valid_features)
        assert feat.num_suspicious == 2
        assert feat.suspicious_rate == 0.02

    def test_negative_total_spending_rejected(self, valid_features):
        """Test total_spending must be >= 0."""
        valid_features["total_spending"] = -100.0
        with pytest.raises(ValidationError):
            TransactionFeatures(**valid_features)


# =============================================================================
# Credit Card Schemas
# =============================================================================


class TestCreditCardCleaned:
    """Test CreditCardCleaned schema."""

    @pytest.fixture()
    def valid_cleaned_card(self):
        """Minimal valid CreditCardCleaned data."""
        return {
            "card_id": "abc123",
            "card_name": "Chase Sapphire Reserve",
            "issuer": "CHASE",
            "source": "creditcardbonuses",
            "annual_fee": 550.0,
            "reward_rates": {"universal_base_rate": 1.0},
            "last_updated": "2026-02-17T12:00:00",
        }

    def test_valid(self, valid_cleaned_card):
        """Test valid cleaned card passes validation."""
        card = CreditCardCleaned(**valid_cleaned_card)
        assert card.card_name == "Chase Sapphire Reserve"
        assert card.issuer == "CHASE"
        assert card.annual_fee == 550.0

    def test_annual_fee_zero(self, valid_cleaned_card):
        """Test zero annual fee is valid."""
        valid_cleaned_card["annual_fee"] = 0.0
        card = CreditCardCleaned(**valid_cleaned_card)
        assert card.annual_fee == 0.0

    def test_annual_fee_too_high(self, valid_cleaned_card):
        """Test annual fee >= 1000 is rejected."""
        valid_cleaned_card["annual_fee"] = 1000.0
        with pytest.raises(ValidationError):
            CreditCardCleaned(**valid_cleaned_card)

    def test_annual_fee_negative(self, valid_cleaned_card):
        """Test negative annual fee is rejected."""
        valid_cleaned_card["annual_fee"] = -50.0
        with pytest.raises(ValidationError):
            CreditCardCleaned(**valid_cleaned_card)

    def test_defaults(self, valid_cleaned_card):
        """Test default values for optional fields."""
        card = CreditCardCleaned(**valid_cleaned_card)
        assert card.discontinued is False
        assert card.offers == []
        assert card.credits == []
        assert card.historical_offers == []

    def test_missing_required_field(self):
        """Test missing required field raises error."""
        data = {
            "card_name": "Card",
            "issuer": "CHASE",
            "source": "test",
            "annual_fee": 0.0,
            "last_updated": "2026-01-01",
            # card_id missing
        }

        with pytest.raises(ValidationError):
            CreditCardCleaned(**data)

    def test_optional_fields(self, valid_cleaned_card):
        """Test optional fields can be set."""
        valid_cleaned_card["network"] = "VISA"
        valid_cleaned_card["currency"] = "POINTS"
        valid_cleaned_card["is_business"] = False
        valid_cleaned_card["is_annual_fee_waived"] = True
        card = CreditCardCleaned(**valid_cleaned_card)
        assert card.network == "VISA"
        assert card.is_annual_fee_waived is True


class TestCreditCardFeatures:
    """Test CreditCardFeatures schema."""

    @pytest.fixture()
    def valid_card_features(self):
        """Minimal valid CreditCardFeatures data."""
        return {
            "card_id": "abc123",
            "card_name": "Chase Sapphire Reserve",
            "issuer": "CHASE",
            "annual_fee": 550.0,
            "base_reward_rate": 1.0,
            "cashback_rate": 1.0,
            "effective_annual_fee": 250.0,
            "effective_fee_year1": 250.0,
            "net_annual_cost": 250.0,
            "expected_annual_rewards": 500.0,
            "net_value_annual": 250.0,
            "net_value_year1": 1000.0,
            "value_per_dollar": 0.02,
        }

    def test_valid(self, valid_card_features):
        """Test valid card features pass validation."""
        feat = CreditCardFeatures(**valid_card_features)
        assert feat.card_id == "abc123"
        assert feat.base_reward_rate == 1.0
        assert feat.net_value_annual == 250.0

    def test_welcome_bonus_defaults(self, valid_card_features):
        """Test welcome bonus fields default correctly."""
        feat = CreditCardFeatures(**valid_card_features)
        assert feat.welcome_bonus_spend_req == 0
        assert feat.welcome_bonus_amount == 0
        assert feat.welcome_bonus_days == 90
        assert feat.bonus_difficulty == "none"

    def test_status_flag_defaults(self, valid_card_features):
        """Test status flags default correctly."""
        feat = CreditCardFeatures(**valid_card_features)
        assert feat.is_active == 1
        assert feat.is_discontinued == 0
        assert feat.is_premium == 0
        assert feat.is_business == 0

    def test_with_welcome_bonus(self, valid_card_features):
        """Test with welcome bonus fields set."""
        valid_card_features["welcome_bonus_spend_req"] = 4000.0
        valid_card_features["welcome_bonus_amount"] = 60000.0
        valid_card_features["welcome_bonus_value_usd"] = 900.0
        valid_card_features["bonus_difficulty"] = "medium"
        feat = CreditCardFeatures(**valid_card_features)
        assert feat.welcome_bonus_spend_req == 4000.0
        assert feat.bonus_difficulty == "medium"

    def test_with_credits(self, valid_card_features):
        """Test credit fields are set properly."""
        valid_card_features["annual_credits_value"] = 300.0
        valid_card_features["num_credits"] = 3
        valid_card_features["has_credits"] = 1
        feat = CreditCardFeatures(**valid_card_features)
        assert feat.annual_credits_value == 300.0
        assert feat.has_credits == 1

    def test_allows_extra_onehot_fields(self, valid_card_features):
        """Test extra one-hot encoded columns are allowed."""
        valid_card_features["issuer_CHASE"] = 1
        valid_card_features["issuer_AMEX"] = 0
        valid_card_features["network_VISA"] = 1
        feat = CreditCardFeatures(**valid_card_features)
        assert feat.issuer_CHASE == 1
        assert feat.issuer_AMEX == 0

    def test_missing_required_field(self):
        """Test missing required field raises error."""
        data = {
            "card_id": "abc",
            "card_name": "Card",
            "issuer": "CHASE",
            "annual_fee": 0.0,
            "base_reward_rate": 1.0,
            "cashback_rate": 1.0,
            # Missing effective_annual_fee, net_value_annual, etc.
        }

        with pytest.raises(ValidationError):
            CreditCardFeatures(**data)


# =============================================================================
# User Profile Schemas
# =============================================================================


class TestUserProfileRaw:
    """Test UserProfileRaw schema."""

    def test_valid(self):
        """Test valid user profile passes validation."""
        data = {
            "user_id": "user_0001",
            "archetype": "high_roller",
            "monthly_budget": 13156.94,
            "cards": "['Chase Sapphire Reserve']",
            "redemption_preference": "travel_portal",
            "age_group": "51-65",
            "location_type": "urban",
        }

        user = UserProfileRaw(**data)
        assert user.archetype == "high_roller"
        assert user.monthly_budget > 0

    def test_invalid_archetype(self):
        """Test invalid archetype raises error."""
        data = {
            "user_id": "user_0001",
            "archetype": "invalid_type",
            "monthly_budget": 1000,
            "cards": "['Card']",
            "redemption_preference": "cash_back",
            "age_group": "26-35",
            "location_type": "urban",
        }

        with pytest.raises(ValidationError):
            UserProfileRaw(**data)

    def test_invalid_age_group(self):
        """Test invalid age group raises error."""
        data = {
            "user_id": "user_0001",
            "archetype": "high_roller",
            "monthly_budget": 1000,
            "cards": "['Card']",
            "redemption_preference": "cash_back",
            "age_group": "100-200",
            "location_type": "urban",
        }

        with pytest.raises(ValidationError):
            UserProfileRaw(**data)

    def test_invalid_location_type(self):
        """Test invalid location type raises error."""
        data = {
            "user_id": "user_0001",
            "archetype": "high_roller",
            "monthly_budget": 1000,
            "cards": "['Card']",
            "redemption_preference": "cash_back",
            "age_group": "26-35",
            "location_type": "space_station",
        }

        with pytest.raises(ValidationError):
            UserProfileRaw(**data)

    def test_zero_budget_rejected(self):
        """Test zero monthly_budget is rejected (must be > 0)."""
        data = {
            "user_id": "user_0001",
            "archetype": "high_roller",
            "monthly_budget": 0,
            "cards": "['Card']",
            "redemption_preference": "cash_back",
            "age_group": "26-35",
            "location_type": "urban",
        }

        with pytest.raises(ValidationError):
            UserProfileRaw(**data)

    def test_invalid_redemption_preference(self):
        """Test invalid redemption preference raises error."""
        data = {
            "user_id": "user_0001",
            "archetype": "high_roller",
            "monthly_budget": 5000,
            "cards": "['Card']",
            "redemption_preference": "bitcoin",
            "age_group": "26-35",
            "location_type": "urban",
        }

        with pytest.raises(ValidationError):
            UserProfileRaw(**data)


class TestUserCardMapping:
    """Test UserCardMapping schema."""

    def test_valid(self):
        """Test valid user-card mapping passes validation."""
        data = {
            "user_id": "user_0001",
            "card_id": "Chase Sapphire Reserve",
            "redemption_preference": "travel_portal",
        }

        mapping = UserCardMapping(**data)
        assert mapping.user_id == "user_0001"
        assert mapping.card_id == "Chase Sapphire Reserve"

    def test_invalid_user_id(self):
        """Test invalid user_id raises error."""
        data = {
            "user_id": "bad_id",
            "card_id": "Chase Sapphire Reserve",
            "redemption_preference": "cash_back",
        }

        with pytest.raises(ValidationError):
            UserCardMapping(**data)

    def test_invalid_redemption_preference(self):
        """Test invalid redemption preference raises error."""
        data = {
            "user_id": "user_0001",
            "card_id": "Chase Sapphire Reserve",
            "redemption_preference": "not_real",
        }

        with pytest.raises(ValidationError):
            UserCardMapping(**data)

    def test_all_redemption_preferences(self):
        """Test all valid redemption preferences are accepted."""
        valid_prefs = [
            "cash_back",
            "travel_transfer",
            "statement_credit",
            "gift_cards",
            "merchandise",
            "travel_portal",
        ]

        for pref in valid_prefs:
            mapping = UserCardMapping(
                user_id="user_0001",
                card_id="Test Card",
                redemption_preference=pref,
            )
            assert mapping.redemption_preference == pref


class TestUserProfileFeatures:
    """Test UserProfileFeatures schema."""

    @pytest.fixture()
    def valid_profile_features(self):
        """Minimal valid UserProfileFeatures data."""
        return {
            "user_id": "user_0001",
            "archetype": "high_roller",
            "monthly_budget": 10000.0,
            "redemption_preference": "travel_portal",
            "age_group": "51-65",
            "location_type": "urban",
            "cards_list": ["Chase Sapphire Reserve", "Amex Platinum"],
            "num_cards": 2,
            "monthly_budget_log": 9.21,
            "annual_budget": 120000.0,
            "age_group_ordinal": 4,
            "estimated_point_value": 0.02,
        }

    def test_valid(self, valid_profile_features):
        """Test valid profile features pass validation."""
        feat = UserProfileFeatures(**valid_profile_features)
        assert feat.user_id == "user_0001"
        assert feat.num_cards == 2
        assert feat.annual_budget == 120000.0

    def test_cards_list_parsed(self, valid_profile_features):
        """Test cards_list is a proper list."""
        feat = UserProfileFeatures(**valid_profile_features)
        assert isinstance(feat.cards_list, list)
        assert len(feat.cards_list) == 2
        assert "Amex Platinum" in feat.cards_list

    def test_num_cards_must_be_positive(self, valid_profile_features):
        """Test num_cards must be >= 1."""
        valid_profile_features["num_cards"] = 0
        with pytest.raises(ValidationError):
            UserProfileFeatures(**valid_profile_features)

    def test_age_ordinal_bounds(self, valid_profile_features):
        """Test age_group_ordinal bounds 0-5."""
        valid_profile_features["age_group_ordinal"] = 6
        with pytest.raises(ValidationError):
            UserProfileFeatures(**valid_profile_features)

    def test_point_value_bounds(self, valid_profile_features):
        """Test estimated_point_value must be 0-1."""
        valid_profile_features["estimated_point_value"] = 1.5
        with pytest.raises(ValidationError):
            UserProfileFeatures(**valid_profile_features)

    def test_budget_quartile_optional(self, valid_profile_features):
        """Test budget_quartile is optional."""
        feat = UserProfileFeatures(**valid_profile_features)
        assert feat.budget_quartile is None

        valid_profile_features["budget_quartile"] = "Q4"
        feat2 = UserProfileFeatures(**valid_profile_features)
        assert feat2.budget_quartile == "Q4"

    def test_allows_extra_onehot_fields(self, valid_profile_features):
        """Test dynamic one-hot encoded fields are allowed."""
        valid_profile_features["archetype_high_roller"] = 1
        valid_profile_features["archetype_young_professional"] = 0
        valid_profile_features["location_urban"] = 1
        feat = UserProfileFeatures(**valid_profile_features)
        assert feat.archetype_high_roller == 1


# =============================================================================
# Validators
# =============================================================================


class TestValidators:
    """Test shared validation utility functions."""

    # --- user_id ---
    def test_validate_user_id_valid(self):
        """Test valid user_id passes."""
        assert validate_user_id_format("user_0001") == "user_0001"
        assert validate_user_id_format("user_9999") == "user_9999"

    def test_validate_user_id_invalid_prefix(self):
        """Test user_id with wrong prefix raises error."""
        with pytest.raises(ValueError):
            validate_user_id_format("invalid_001")

    def test_validate_user_id_non_numeric_suffix(self):
        """Test user_id with non-numeric suffix raises error."""
        with pytest.raises(ValueError):
            validate_user_id_format("user_abc")

    def test_validate_user_id_non_string(self):
        """Test non-string user_id raises error."""
        with pytest.raises(ValueError):
            validate_user_id_format(12345)

    # --- transaction_id ---
    def test_validate_transaction_id_valid(self):
        """Test valid transaction_id passes."""
        assert validate_transaction_id_format("txn_0000123") == "txn_0000123"

    def test_validate_transaction_id_invalid_prefix(self):
        """Test transaction_id with wrong prefix raises error."""
        with pytest.raises(ValueError):
            validate_transaction_id_format("trans_001")

    def test_validate_transaction_id_non_string(self):
        """Test non-string transaction_id raises error."""
        with pytest.raises(ValueError):
            validate_transaction_id_format(99999)

    # --- category ---
    def test_validate_category_valid(self):
        """Test valid categories pass."""
        assert validate_category("dining") == "dining"
        assert validate_category("travel") == "travel"
        assert validate_category("groceries") == "groceries"
        assert validate_category("unknown") == "unknown"
        assert validate_category("other") == "other"

    def test_validate_category_invalid(self):
        """Test invalid category raises error."""
        with pytest.raises(ValueError):
            validate_category("fake_category")

    # --- mcc_code ---
    def test_validate_mcc_valid(self):
        """Test valid MCC code passes."""
        assert validate_mcc_code(5812) == 5812
        assert validate_mcc_code(3000) == 3000
        assert validate_mcc_code(1000) == 1000
        assert validate_mcc_code(9999) == 9999

    def test_validate_mcc_too_small(self):
        """Test MCC code < 1000 raises error."""
        with pytest.raises(ValueError):
            validate_mcc_code(999)

    def test_validate_mcc_too_large(self):
        """Test MCC code > 9999 raises error."""
        with pytest.raises(ValueError):
            validate_mcc_code(10000)

    def test_validate_mcc_string_coercion(self):
        """Test MCC code can be coerced from string."""
        assert validate_mcc_code("5812") == 5812

    def test_validate_mcc_invalid_type(self):
        """Test non-numeric MCC raises error."""
        with pytest.raises(ValueError):
            validate_mcc_code("abc")

    # --- amount ---
    def test_validate_amount_positive(self):
        """Test valid positive amount passes."""
        assert validate_amount_positive(100.50) == 100.50
        assert validate_amount_positive(0.01) == 0.01

    def test_validate_amount_zero_rejected(self):
        """Test zero amount is rejected."""
        with pytest.raises(ValueError):
            validate_amount_positive(0)

    def test_validate_amount_negative_rejected(self):
        """Test negative amount is rejected."""
        with pytest.raises(ValueError):
            validate_amount_positive(-10.0)

    def test_validate_amount_int_coercion(self):
        """Test integer amount is coerced to float."""
        result = validate_amount_positive(50)
        assert result == 50.0
        assert isinstance(result, float)

    def test_validate_amount_string_coercion(self):
        """Test string amount is coerced to float."""
        assert validate_amount_positive("25.50") == 25.50

    # --- redemption_preference ---
    def test_validate_redemption_valid(self):
        """Test valid redemption preferences pass."""
        assert validate_redemption_preference("cash_back") == "cash_back"
        assert validate_redemption_preference("travel_transfer") == "travel_transfer"
        assert validate_redemption_preference("travel_portal") == "travel_portal"

    def test_validate_redemption_invalid(self):
        """Test invalid redemption preference raises error."""
        with pytest.raises(ValueError):
            validate_redemption_preference("bitcoin")

    # --- archetype ---
    def test_validate_archetype_valid(self):
        """Test valid archetypes pass."""
        assert validate_archetype("high_roller") == "high_roller"
        assert validate_archetype("young_professional") == "young_professional"
        assert validate_archetype("minimal_user") == "minimal_user"

    def test_validate_archetype_invalid(self):
        """Test invalid archetype raises error."""
        with pytest.raises(ValueError):
            validate_archetype("nonexistent_archetype")


# =============================================================================
# Feature Metadata
# =============================================================================


class TestFeatureMetadata:
    """Test feature metadata schemas."""

    def test_valid(self):
        """Test valid feature metadata."""
        meta = FeatureMetadata(
            name="net_value_annual",
            data_type="numeric",
            description="Net annual value",
            source="credit_card",
            nullable=False,
            required_for_ml=True,
            min_value=-1000.0,
            max_value=5000.0,
        )

        assert meta.name == "net_value_annual"
        assert meta.data_type == "numeric"
        assert meta.required_for_ml is True

    def test_minimal(self):
        """Test feature metadata with only required fields."""
        meta = FeatureMetadata(
            name="test_feature",
            data_type="categorical",
            description="A test feature",
            source="transaction",
        )

        assert meta.nullable is False
        assert meta.required_for_ml is False
        assert meta.min_value is None
        assert meta.categories is None

    def test_invalid_data_type(self):
        """Test invalid data_type is rejected."""
        with pytest.raises(ValidationError):
            FeatureMetadata(
                name="test",
                data_type="complex_number",
                description="bad type",
                source="credit_card",
            )

    def test_invalid_source(self):
        """Test invalid source is rejected."""
        with pytest.raises(ValidationError):
            FeatureMetadata(
                name="test",
                data_type="numeric",
                description="bad source",
                source="unknown_source",
            )


class TestFeatureRegistry:
    """Test feature registry."""

    @pytest.fixture()
    def sample_registry(self):
        """Build a sample registry with features from all sources."""
        return FeatureRegistry(
            version="1.0.0",
            credit_card_features=[
                FeatureMetadata(
                    name="base_reward_rate",
                    data_type="numeric",
                    description="Base reward rate",
                    source="credit_card",
                    required_for_ml=True,
                ),
                FeatureMetadata(
                    name="net_value_annual",
                    data_type="numeric",
                    description="Net annual value",
                    source="credit_card",
                ),
            ],
            transaction_features=[
                FeatureMetadata(
                    name="total_spending",
                    data_type="numeric",
                    description="Total spending",
                    source="transaction",
                    required_for_ml=True,
                ),
            ],
            user_profile_features=[
                FeatureMetadata(
                    name="archetype",
                    data_type="categorical",
                    description="User archetype",
                    source="user_profile",
                    categories=["high_roller", "young_professional"],
                ),
            ],
        )

    def test_basic_creation(self):
        """Test basic registry creation."""
        registry = FeatureRegistry(
            version="1.0.0",
            credit_card_features=[
                FeatureMetadata(
                    name="base_reward_rate",
                    data_type="numeric",
                    description="Base reward rate",
                    source="credit_card",
                )
            ],
        )

        assert registry.version == "1.0.0"
        assert len(registry.credit_card_features) == 1

    def test_get_feature_found(self, sample_registry):
        """Test get_feature returns matching feature."""
        feat = sample_registry.get_feature("base_reward_rate")
        assert feat is not None
        assert feat.name == "base_reward_rate"

    def test_get_feature_not_found(self, sample_registry):
        """Test get_feature returns None for unknown feature."""
        feat = sample_registry.get_feature("nonexistent_feature")
        assert feat is None

    def test_get_features_by_type(self, sample_registry):
        """Test get_features_by_type returns correct features."""
        numeric = sample_registry.get_features_by_type("numeric")
        assert len(numeric) == 3

        categorical = sample_registry.get_features_by_type("categorical")
        assert len(categorical) == 1
        assert categorical[0].name == "archetype"

    def test_get_required_features(self, sample_registry):
        """Test get_required_features returns only required features."""
        required = sample_registry.get_required_features()
        assert len(required) == 2
        names = {f.name for f in required}
        assert "base_reward_rate" in names
        assert "total_spending" in names

    def test_empty_registry(self):
        """Test empty registry methods work."""
        registry = FeatureRegistry(version="1.0.0")
        assert registry.get_feature("anything") is None
        assert registry.get_features_by_type("numeric") == []
        assert registry.get_required_features() == []
