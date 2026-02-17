
"""
Unit tests for feature engineering module.

Covers Story 3.2 acceptance criteria:
  - Feature engineering is deterministic and reproducible
  - All features documented with descriptions
  - Features stored in standardized format

Follows the same pattern as test_cleaning.py and test_generators.py
"""

import pandas as pd
import pytest
import numpy as np

from src.data_pipeline.preprocessing.feature_engineering import (
    CreditCardFeatureEngineer,
    TransactionFeatureEngineer,
    UserProfileFeatureEngineer,
    engineer_all_features
)


# =====================================================================
# Fixtures - Sample Data
# =====================================================================

@pytest.fixture
def sample_credit_cards():
    """Sample credit card data matching API structure."""
    return pd.DataFrame([
        {
            'card_id': 'card_001',
            'card_name': 'Chase Sapphire Reserve',
            'issuer': 'CHASE',
            'network': 'VISA',
            'annual_fee': 550.0,
            'is_annual_fee_waived': False,
            'is_business': False,
            'currency': 'POINTS',
            'reward_rates': {'universal_base_rate': 1.0},
            'universal_cashback_percent': 1.0,
            'offers': [
                {
                    'spend': 4000,
                    'amount': [{'amount': 60000}],
                    'days': 90,
                    'credits': []
                }
            ],
            'credits': [
                {'description': '$300 travel credit', 'value': 300, 'weight': 1.0}
            ],
            'discontinued': False
        },
        {
            'card_id': 'card_002',
            'card_name': 'Citi Double Cash',
            'issuer': 'CITI',
            'network': 'MASTERCARD',
            'annual_fee': 0.0,
            'is_annual_fee_waived': False,
            'is_business': False,
            'currency': 'CASHBACK',
            'reward_rates': {'universal_base_rate': 2.0},
            'universal_cashback_percent': 2.0,
            'offers': [],
            'credits': [],
            'discontinued': False
        },
        {
            'card_id': 'card_003',
            'card_name': 'Old Discontinued Card',
            'issuer': 'AMERICAN_EXPRESS',
            'network': 'AMERICAN_EXPRESS',
            'annual_fee': 95.0,
            'is_annual_fee_waived': True,
            'is_business': False,
            'currency': 'POINTS',
            'reward_rates': {'universal_base_rate': 1.5},
            'universal_cashback_percent': 1.5,
            'offers': [],
            'credits': [],
            'discontinued': True
        }
    ])


@pytest.fixture
def sample_transactions():
    """Sample transaction data matching generated structure."""
    return pd.DataFrame([
        {
            'transaction_id': 'txn_001',
            'user_id': 'user_0001',
            'date': '2025-08-01',
            'category': 'dining',
            'merchant': 'Starbucks',
            'mcc_code': 5812,
            'amount': 16.09,
            'card_used': 'Chase Sapphire Reserve'
        },
        {
            'transaction_id': 'txn_002',
            'user_id': 'user_0001',
            'date': '2025-08-02',
            'category': 'travel',
            'merchant': 'United Airlines',
            'mcc_code': 3000,
            'amount': 283.54,
            'card_used': 'Chase Sapphire Reserve'
        },
        {
            'transaction_id': 'txn_003',
            'user_id': 'user_0001',
            'date': '2025-08-05',
            'category': 'utilities',
            'merchant': 'Electric Company',
            'mcc_code': 4814,
            'amount': 158.15,
            'card_used': 'Citi Double Cash'
        },
        {
            'transaction_id': 'txn_004',
            'user_id': 'user_0002',
            'date': '2025-08-10',
            'category': 'dining',
            'merchant': 'Starbucks',
            'mcc_code': 5812,
            'amount': 12.50,
            'card_used': 'Citi Double Cash'
        },
        {
            'transaction_id': 'txn_005',
            'user_id': 'user_0002',
            'date': '2025-08-15',
            'category': 'entertainment',
            'merchant': 'AMC Theatres',
            'mcc_code': 7941,
            'amount': 25.00,
            'card_used': 'Citi Double Cash'
        },
        {
            'transaction_id': 'txn_006',
            'user_id': 'user_0002',
            'date': '2025-08-16',
            'category': 'dining',
            'merchant': 'Starbucks',
            'mcc_code': 5812,
            'amount': 8.50,
            'card_used': 'Citi Double Cash'
        }
    ])


@pytest.fixture
def sample_user_profiles():
    """Sample user profile data matching generated structure."""
    return pd.DataFrame([
        {
            'user_id': 'user_0001',
            'archetype': 'high_roller',
            'monthly_budget': 13156.94,
            'cards': "['Chase Sapphire Reserve', 'Amex Platinum', 'Capital One Venture X']",
            'redemption_preference': 'travel_portal',
            'age_group': '51-65',
            'location_type': 'urban'
        },
        {
            'user_id': 'user_0002',
            'archetype': 'budget_conscious',
            'monthly_budget': 1920.92,
            'cards': "['Citi Double Cash', 'Chase Freedom Flex']",
            'redemption_preference': 'cash_back',
            'age_group': '26-35',
            'location_type': 'rural'
        },
        {
            'user_id': 'user_0003',
            'archetype': 'young_professional',
            'monthly_budget': 3500.00,
            'cards': "['Amex Gold']",
            'redemption_preference': 'gift_cards',
            'age_group': '18-25',
            'location_type': 'urban'
        }
    ])


# =====================================================================
# CreditCardFeatureEngineer Tests
# =====================================================================

class TestCreditCardFeatureEngineer:
    """Test credit card feature engineering."""
    
    def test_extract_base_reward_rate(self, sample_credit_cards):
        engineer = CreditCardFeatureEngineer()
        result = engineer.extract_base_reward_rate(sample_credit_cards)
        
        assert 'base_reward_rate' in result.columns
        assert 'cashback_rate' in result.columns
        assert result['base_reward_rate'].iloc[0] == 1.0
        assert result['base_reward_rate'].iloc[1] == 2.0
    
    def test_parse_welcome_bonus_offers(self, sample_credit_cards):
        engineer = CreditCardFeatureEngineer()
        result = engineer.parse_welcome_bonus_offers(sample_credit_cards)
        
        assert 'welcome_bonus_spend_req' in result.columns
        assert 'welcome_bonus_amount' in result.columns
        assert 'welcome_bonus_days' in result.columns
        
        # First card: 60k bonus for $4k spend
        assert result['welcome_bonus_spend_req'].iloc[0] == 4000
        assert result['welcome_bonus_amount'].iloc[0] == 60000
        assert result['welcome_bonus_days'].iloc[0] == 90
        
        # Second card: no bonus
        assert result['welcome_bonus_amount'].iloc[1] == 0
    
    def test_calculate_welcome_bonus_value(self, sample_credit_cards):
        engineer = CreditCardFeatureEngineer()
        df = engineer.parse_welcome_bonus_offers(sample_credit_cards)
        result = engineer.calculate_welcome_bonus_value(df)
        
        assert 'welcome_bonus_value_usd' in result.columns
        assert 'welcome_bonus_roi' in result.columns
        assert 'bonus_difficulty' in result.columns
        
        # 60k points at 1 cent = $600
        assert result['welcome_bonus_value_usd'].iloc[0] == 600.0
        # ROI = 600 / 4000 = 0.15
        assert abs(result['welcome_bonus_roi'].iloc[0] - 0.15) < 0.01
    
    def test_parse_credits_benefits(self, sample_credit_cards):
        engineer = CreditCardFeatureEngineer()
        result = engineer.parse_credits_benefits(sample_credit_cards)
        
        assert 'annual_credits_value' in result.columns
        assert 'num_credits' in result.columns
        assert 'has_credits' in result.columns
        
        # First card: $300 credit
        assert result['annual_credits_value'].iloc[0] == 300
        assert result['num_credits'].iloc[0] == 1
        assert result['has_credits'].iloc[0] == 1
        
        # Second card: no credits
        assert result['annual_credits_value'].iloc[1] == 0
        assert result['has_credits'].iloc[1] == 0
    
    def test_calculate_effective_annual_fee(self, sample_credit_cards):
        engineer = CreditCardFeatureEngineer()
        df = engineer.parse_credits_benefits(sample_credit_cards)
        result = engineer.calculate_effective_annual_fee(df)
        
        assert 'effective_annual_fee' in result.columns
        assert 'effective_fee_year1' in result.columns
        
        # Card 1: $550 fee - $300 credit = $250
        assert result['effective_annual_fee'].iloc[0] == 250
        
        # Card 2: $0 fee
        assert result['effective_annual_fee'].iloc[1] == 0
        
        # Card 3: Waived first year
        assert result['effective_fee_year1'].iloc[2] == 0
    
    def test_filter_active_cards(self, sample_credit_cards):
        engineer = CreditCardFeatureEngineer()
        result = engineer.filter_active_cards(sample_credit_cards)
        
        assert 'is_active' in result.columns
        assert 'is_discontinued' in result.columns
        
        # First two cards active
        assert result['is_active'].iloc[0] == 1
        assert result['is_active'].iloc[1] == 1
        
        # Third card discontinued
        assert result['is_discontinued'].iloc[2] == 1
        assert result['is_active'].iloc[2] == 0
    
    def test_create_issuer_network_features(self, sample_credit_cards):
        engineer = CreditCardFeatureEngineer()
        result = engineer.create_issuer_network_features(sample_credit_cards)
        
        # Check issuer encoding
        assert 'issuer_CHASE' in result.columns
        assert 'issuer_CITI' in result.columns
        
        # Check network encoding
        assert 'network_VISA' in result.columns
        assert 'network_MASTERCARD' in result.columns
        
        # Check tier flags
        assert 'is_premium' in result.columns
        assert 'is_no_annual_fee' in result.columns
        
        # Verify tiers
        assert result['is_premium'].iloc[0] == 1  # $550 fee
        assert result['is_no_annual_fee'].iloc[1] == 1  # $0 fee
    
    def test_engineer_features_full_pipeline(self, sample_credit_cards):
        engineer = CreditCardFeatureEngineer()
        result = engineer.engineer_features(sample_credit_cards, annual_spending=25000)
        
        # Check all expected features exist
        expected_features = [
            'base_reward_rate',
            'welcome_bonus_value_usd',
            'annual_credits_value',
            'effective_annual_fee',
            'net_value_annual',
            'net_value_year1',
            'is_active',
            'is_premium'
        ]
        
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"
        
        # Check output shape
        assert len(result) == len(sample_credit_cards)
        assert result.shape[1] > sample_credit_cards.shape[1]


# =====================================================================
# TransactionFeatureEngineer Tests
# =====================================================================

class TestTransactionFeatureEngineer:
    """Test transaction feature engineering."""
    
    def test_aggregate_spending_by_category(self, sample_transactions):
        engineer = TransactionFeatureEngineer()
        result = engineer.aggregate_spending_by_category(sample_transactions)
        
        # One row per user
        assert len(result) == sample_transactions['user_id'].nunique()
        
        # Check for category columns
        assert 'dining_total_spent' in result.columns
        assert 'travel_total_spent' in result.columns
        assert 'total_spending' in result.columns
        assert 'total_transactions' in result.columns
        
        # User 1: dining + travel + utilities
        user1 = result[result['user_id'] == 'user_0001'].iloc[0]
        assert user1['dining_total_spent'] > 0
        assert user1['travel_total_spent'] > 0
        
        # Check totals are correct
        user1_original = sample_transactions[sample_transactions['user_id'] == 'user_0001']
        assert abs(user1['total_spending'] - user1_original['amount'].sum()) < 0.01
    
    def test_extract_temporal_patterns(self, sample_transactions):
        engineer = TransactionFeatureEngineer()
        result = engineer.extract_temporal_patterns(sample_transactions)
        
        expected_cols = [
            'user_id',
            'weekend_spending_ratio',
            'peak_spending_month',
            'avg_transaction_amount',
            'total_spending_temporal'
        ]
        
        for col in expected_cols:
            assert col in result.columns
        
        # Check data types
        assert result['peak_spending_month'].dtype in [np.int64, np.int32, int]
        assert result['weekend_spending_ratio'].dtype == float
        
        # Weekend ratio should be between 0 and 1
        assert (result['weekend_spending_ratio'] >= 0).all()
        assert (result['weekend_spending_ratio'] <= 1).all()
    
    def test_analyze_card_usage_patterns(self, sample_transactions):
        engineer = TransactionFeatureEngineer()
        result = engineer.analyze_card_usage_patterns(sample_transactions)
        
        assert 'num_cards_used' in result.columns
        assert 'primary_card' in result.columns
        assert 'card_switch_rate' in result.columns
        
        # User 1 uses 2 different cards
        user1 = result[result['user_id'] == 'user_0001'].iloc[0]
        assert user1['num_cards_used'] == 2
        assert user1['primary_card'] in ['Chase Sapphire Reserve', 'Citi Double Cash']
        
        # User 2 uses 1 card
        user2 = result[result['user_id'] == 'user_0002'].iloc[0]
        assert user2['num_cards_used'] == 1
    
    def test_analyze_mcc_patterns(self, sample_transactions):
        engineer = TransactionFeatureEngineer()
        result = engineer.analyze_mcc_patterns(sample_transactions)
        
        assert 'num_unique_mccs' in result.columns
        assert 'primary_mcc' in result.columns
        assert 'avg_spending_per_mcc' in result.columns
        
        # Check values are reasonable
        assert (result['num_unique_mccs'] >= 1).all()
        assert result['primary_mcc'].notna().all()
        assert (result['avg_spending_per_mcc'] > 0).all()
    
    # def test_create_merchant_features(self, sample_transactions):
    #     engineer = TransactionFeatureEngineer()
    #     result = engineer.create_merchant_features(sample_transactions)
        
    #     assert 'num_unique_merchants' in result.columns
    #     assert 'favorite_merchant' in result.columns
    #     assert 'repeat_merchant_ratio' in result.columns
        
    #     # User 1 has 3 unique merchants
    #     user1 = result[result['user_id'] == 'user_0001'].iloc[0]
    #     assert user1['num_unique_merchants'] == 3
        
    #     # User 2 has repeat merchant (Starbucks appears twice)
    #     user2 = result[result['user_id'] == 'user_0002'].iloc[0]
    #     assert user2['num_unique_merchants'] == 3  # Starbucks, Subway, AMC
    #     # 3 transactions, 3 unique = no repeats actually, OR
    #     # if Starbucks appears twice: 4 transactions, 3 unique = 0.25 repeat ratio
    #     assert user2['repeat_merchant_ratio'] >= 0  # Just check it's non-negative
    
    def test_handle_suspicious_transactions(self):
        """Test suspicious transaction handling."""
        # Create data with suspicious flag
        df = pd.DataFrame([
            {'transaction_id': 'txn_001', 'user_id': 'user_0001', 'amount': 50, 'suspicious': False, 
             'date': '2025-08-01', 'category': 'dining', 'merchant': 'Test', 'mcc_code': 5812, 'card_used': 'Card A'},
            {'transaction_id': 'txn_002', 'user_id': 'user_0001', 'amount': 15000, 'suspicious': True,
             'date': '2025-08-01', 'category': 'dining', 'merchant': 'Test', 'mcc_code': 5812, 'card_used': 'Card A'},
            {'transaction_id': 'txn_003', 'user_id': 'user_0002', 'amount': 30, 'suspicious': False,
             'date': '2025-08-01', 'category': 'dining', 'merchant': 'Test', 'mcc_code': 5812, 'card_used': 'Card A'}
        ])
        
        engineer = TransactionFeatureEngineer()
        result = engineer.handle_suspicious_transactions(df)
        
        assert 'num_suspicious' in result.columns
        assert 'suspicious_rate' in result.columns
        
        # User 1 has 1 suspicious out of 2
        user1 = result[result['user_id'] == 'user_0001'].iloc[0]
        assert user1['num_suspicious'] == 1
        assert user1['suspicious_rate'] == 0.5
    
    def test_engineer_features_full_pipeline(self, sample_transactions):
        engineer = TransactionFeatureEngineer()
        result = engineer.engineer_features(sample_transactions)
        
        # One row per user
        assert len(result) == sample_transactions['user_id'].nunique()
        
        # Check all feature types present
        assert any('_total_spent' in c for c in result.columns)
        assert 'total_spending' in result.columns
        assert 'avg_transaction_amount' in result.columns
        assert 'num_cards_used' in result.columns
        assert 'num_unique_merchants' in result.columns
        assert 'spending_diversity' in result.columns


# =====================================================================
# UserProfileFeatureEngineer Tests
# =====================================================================

class TestUserProfileFeatureEngineer:
    """Test user profile feature engineering."""
    
    def test_parse_cards_column(self, sample_user_profiles):
        engineer = UserProfileFeatureEngineer()
        result = engineer.parse_cards_column(sample_user_profiles)
        
        assert 'cards_list' in result.columns
        assert 'num_cards' in result.columns
        
        # Check parsed values
        assert isinstance(result['cards_list'].iloc[0], list)
        assert result['num_cards'].iloc[0] == 3
        assert result['num_cards'].iloc[1] == 2
    
    def test_encode_archetype(self, sample_user_profiles):
        engineer = UserProfileFeatureEngineer()
        result = engineer.encode_archetype(sample_user_profiles)
        
        assert 'archetype_high_roller' in result.columns
        assert 'archetype_budget_conscious' in result.columns
        
        # Check encoding
        assert result['archetype_high_roller'].iloc[0] == 1
        assert result['archetype_budget_conscious'].iloc[1] == 1
    
    def test_encode_age_group(self, sample_user_profiles):
        engineer = UserProfileFeatureEngineer()
        result = engineer.encode_age_group(sample_user_profiles)
        
        assert 'age_group_ordinal' in result.columns
        assert 'age_51-65' in result.columns
        assert 'age_26-35' in result.columns
        
        # Check ordinal values
        assert result['age_group_ordinal'].iloc[0] == 4  # 51-65
        assert result['age_group_ordinal'].iloc[1] == 2  # 26-35
    
    def test_encode_location_type(self, sample_user_profiles):
        engineer = UserProfileFeatureEngineer()
        result = engineer.encode_location_type(sample_user_profiles)
        
        assert 'location_urban' in result.columns
        assert 'location_rural' in result.columns
        
        assert result['location_urban'].iloc[0] == 1
        assert result['location_rural'].iloc[1] == 1
    
    def test_create_budget_features(self, sample_user_profiles):
        engineer = UserProfileFeatureEngineer()
        result = engineer.create_budget_features(sample_user_profiles)
        
        assert 'monthly_budget_log' in result.columns
        assert 'annual_budget' in result.columns
        assert 'budget_quartile' in result.columns
        
        # Check annual budget
        assert result['annual_budget'].iloc[0] == result['monthly_budget'].iloc[0] * 12
        
        # Log should be positive
        assert (result['monthly_budget_log'] >= 0).all()
    
    def test_estimate_point_valuations(self, sample_user_profiles):
        engineer = UserProfileFeatureEngineer()
        result = engineer.estimate_point_valuations(sample_user_profiles)
        
        assert 'estimated_point_value' in result.columns
        
        # travel_portal = 0.015
        assert result['estimated_point_value'].iloc[0] == 0.015
        # cash_back = 0.01
        assert result['estimated_point_value'].iloc[1] == 0.01
    
    def test_encode_redemption_preferences(self, sample_user_profiles):
        engineer = UserProfileFeatureEngineer()
        result = engineer.encode_redemption_preferences(sample_user_profiles)
        
        assert 'redemption_travel_portal' in result.columns
        assert 'redemption_cash_back' in result.columns
        
        assert result['redemption_travel_portal'].iloc[0] == 1
        assert result['redemption_cash_back'].iloc[1] == 1
    
    def test_engineer_features_full_pipeline(self, sample_user_profiles):
        engineer = UserProfileFeatureEngineer()
        result = engineer.engineer_features(sample_user_profiles)
        
        # Check key features exist
        expected_features = [
            'num_cards',
            'archetype_high_roller',
            'age_group_ordinal',
            'location_urban',
            'monthly_budget_log',
            'estimated_point_value',
            'redemption_travel_portal'
        ]
        
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"
        
        # Check shape
        assert len(result) == len(sample_user_profiles)


# =====================================================================
# Determinism & Reproducibility Tests
# =====================================================================

class TestReproducibility:
    """Test that feature engineering is deterministic and reproducible."""
    
    def test_credit_card_features_deterministic(self, sample_credit_cards):
        """Same input should produce same output."""
        engineer1 = CreditCardFeatureEngineer()
        engineer2 = CreditCardFeatureEngineer()
        
        result1 = engineer1.engineer_features(sample_credit_cards, annual_spending=25000)
        result2 = engineer2.engineer_features(sample_credit_cards, annual_spending=25000)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_transaction_features_deterministic(self, sample_transactions):
        """Same input should produce same output."""
        engineer1 = TransactionFeatureEngineer()
        engineer2 = TransactionFeatureEngineer()
        
        result1 = engineer1.engineer_features(sample_transactions)
        result2 = engineer2.engineer_features(sample_transactions)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_user_features_deterministic(self, sample_user_profiles):
        """Same input should produce same output."""
        engineer1 = UserProfileFeatureEngineer()
        engineer2 = UserProfileFeatureEngineer()
        
        result1 = engineer1.engineer_features(sample_user_profiles)
        result2 = engineer2.engineer_features(sample_user_profiles)
        
        pd.testing.assert_frame_equal(result1, result2)


# =====================================================================
# Integration: engineer_all_features
# =====================================================================

class TestEngineerAllFeatures:
    """Test the convenience function."""
    
    def test_engineer_all_features(self, sample_credit_cards, sample_transactions, sample_user_profiles):
        cards_f, txns_f, users_f = engineer_all_features(
            credit_cards_df=sample_credit_cards,
            transactions_df=sample_transactions,
            users_df=sample_user_profiles,
            annual_spending=25000
        )
        
        # All should return DataFrames
        assert cards_f is not None
        assert txns_f is not None
        assert users_f is not None
        
        # Check shapes
        assert len(cards_f) == len(sample_credit_cards)
        assert len(txns_f) == sample_transactions['user_id'].nunique()
        assert len(users_f) == len(sample_user_profiles)
    
    def test_engineer_all_features_with_none(self):
        """Test handling of None inputs."""
        cards_f, txns_f, users_f = engineer_all_features(
            credit_cards_df=None,
            transactions_df=None,
            users_df=None
        )
        
        assert cards_f is None
        assert txns_f is None
        assert users_f is None
    
    def test_engineer_all_features_saves_output(self, tmp_path, sample_credit_cards, sample_transactions, sample_user_profiles):
        """Test saving engineered features to files."""
        output_dir = tmp_path / "features"
        
        cards_f, txns_f, users_f = engineer_all_features(
            credit_cards_df=sample_credit_cards,
            transactions_df=sample_transactions,
            users_df=sample_user_profiles,
            output_dir=output_dir
        )
        
        # Check files were created
        assert (output_dir / 'credit_cards_features.csv').exists()
        assert (output_dir / 'transactions_features.csv').exists()
        assert (output_dir / 'users_features.csv').exists()
        
        # Verify files can be read back
        loaded_cards = pd.read_csv(output_dir / 'credit_cards_features.csv')
        assert len(loaded_cards) == len(cards_f)


# =====================================================================
# Edge Cases
# =====================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_transactions_dataframe(self):
        """Test handling of empty transaction DataFrame."""
        engineer = TransactionFeatureEngineer()
        empty_df = pd.DataFrame(columns=[
            'transaction_id', 'user_id', 'date', 'category', 
            'merchant', 'mcc_code', 'amount', 'card_used'
        ])
        
        # Should not crash
        result = engineer.aggregate_spending_by_category(empty_df)
        
        # Should return empty DataFrame with user_id column
        assert len(result) == 0
        assert 'user_id' in result.columns
    
    # def test_missing_optional_credit_card_columns(self):
    #     """Test handling when optional columns are missing."""
    #     df = pd.DataFrame([
    #         {
    #             'card_id': 'card_001',
    #             'card_name': 'Test Card',
    #             'issuer': 'TEST',
    #             'annual_fee': 0,
    #             'reward_rates': {'universal_base_rate': 1.0},
    #             'offers': [],  # Empty offers instead of missing
    #             'credits': [],  # Empty credits instead of missing
    #             # Missing: discontinued, is_business
    #         }
    #     ])
        
    #     engineer = CreditCardFeatureEngineer()
    #     result = engineer.engineer_features(df)
        
    #     # Should have default values
    #     assert 'is_active' in result.columns
    #     assert result['is_active'].iloc[0] == 1  # Default to active
    
    def test_null_monthly_budget(self):
        """Test handling of null budget values."""
        # Need at least 4 rows for quartiles to work
        df = pd.DataFrame([
            {
                'user_id': 'user_001',
                'archetype': 'high_roller',
                'monthly_budget': None,
                'cards': "['Card A']",
                'redemption_preference': 'cash_back',
                'age_group': '26-35',
                'location_type': 'urban'
            },
            {
                'user_id': 'user_002',
                'archetype': 'budget_conscious',
                'monthly_budget': 1000,
                'cards': "['Card B']",
                'redemption_preference': 'cash_back',
                'age_group': '26-35',
                'location_type': 'urban'
            },
            {
                'user_id': 'user_003',
                'archetype': 'young_professional',
                'monthly_budget': 2000,
                'cards': "['Card C']",
                'redemption_preference': 'travel_portal',
                'age_group': '36-50',
                'location_type': 'suburban'
            },
            {
                'user_id': 'user_004',
                'archetype': 'high_roller',
                'monthly_budget': 5000,
                'cards': "['Card D']",
                'redemption_preference': 'travel_transfer',
                'age_group': '51-65',
                'location_type': 'rural'
            }
        ])
        
        engineer = UserProfileFeatureEngineer()
        result = engineer.engineer_features(df)
        
        # Should handle null budget
        assert 'monthly_budget_log' in result.columns
        assert result['monthly_budget_log'].iloc[0] == 0  # log1p(0) = 0
    
    def test_malformed_cards_string(self):
        """Test handling of malformed cards string."""
        df = pd.DataFrame([
            {
                'user_id': 'user_001',
                'archetype': 'high_roller',
                'monthly_budget': 5000,
                'cards': "Card A, Card B",  # Not proper list format
                'redemption_preference': 'cash_back',
                'age_group': '26-35',
                'location_type': 'urban'
            }
        ])
        
        engineer = UserProfileFeatureEngineer()
        result = engineer.parse_cards_column(df)
        
        # Should parse as list
        assert isinstance(result['cards_list'].iloc[0], list)
        assert result['num_cards'].iloc[0] >= 1
    
    def test_unknown_currency_type(self):
        """Test handling of unknown currency in welcome bonus."""
        df = pd.DataFrame([
            {
                'card_id': 'card_001',
                'card_name': 'Mystery Card',
                'issuer': 'UNKNOWN',
                'network': 'VISA',
                'annual_fee': 0,
                'currency': 'MYSTERY_POINTS',  # Unknown currency
                'reward_rates': {'universal_base_rate': 1.0},
                'offers': [{'spend': 1000, 'amount': [{'amount': 10000}], 'days': 90}],
                'credits': [],
                'discontinued': False
            }
        ])
        
        engineer = CreditCardFeatureEngineer()
        df = engineer.parse_welcome_bonus_offers(df)
        result = engineer.calculate_welcome_bonus_value(df)
        
        # Should use default valuation (1.0 cent per point)
        assert result['welcome_bonus_value_usd'].iloc[0] == 100.0


# =====================================================================
# Data Quality Tests
# =====================================================================

class TestDataQuality:
    """Test data quality of engineered features."""
    
    def test_no_nan_in_critical_features(self, sample_credit_cards):
        """Critical features should not have NaN values."""
        engineer = CreditCardFeatureEngineer()
        result = engineer.engineer_features(sample_credit_cards)
        
        critical_features = ['base_reward_rate', 'net_value_annual', 'is_active']
        
        for feat in critical_features:
            assert not result[feat].isna().any(), f"Found NaN in {feat}"
    
    def test_positive_amounts_preserved(self, sample_transactions):
        """All aggregated amounts should be non-negative."""
        engineer = TransactionFeatureEngineer()
        result = engineer.aggregate_spending_by_category(sample_transactions)
        
        spending_cols = [c for c in result.columns if c.endswith('_total_spent')]
        for col in spending_cols:
            assert (result[col] >= 0).all(), f"Negative values in {col}"
    
    def test_feature_ranges_reasonable(self, sample_user_profiles):
        """Check that feature values are in reasonable ranges."""
        engineer = UserProfileFeatureEngineer()
        result = engineer.engineer_features(sample_user_profiles)
        
        # Point values should be between 0 and 1 dollar
        assert (result['estimated_point_value'] >= 0).all()
        assert (result['estimated_point_value'] <= 1.0).all()
        
        # Age ordinal should be 0-5
        assert (result['age_group_ordinal'] >= 0).all()
        assert (result['age_group_ordinal'] <= 5).all()
