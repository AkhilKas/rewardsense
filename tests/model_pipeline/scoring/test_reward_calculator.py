"""
TDD Tests for RewardCalculator - Story 2.1

Following TDD: These tests are written BEFORE implementation.
They define the expected behavior of the reward calculation logic.
"""

from datetime import datetime


class TestRewardCalculator:
    """Test core reward calculation logic."""
    
    def test_calculate_base_reward(self):
        """Test basic reward calculation with universal rate."""
        # This test will fail until we implement RewardCalculator
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        # Card with 1% cashback on everything
        card = {
            'card_id': 'card_001',
            'card_name': 'Test Card',
            'reward_rates': {'universal_base_rate': 1.0},
            'annual_fee': 0
        }
        
        transaction = {
            'amount': 100.0,
            'category': 'dining',
            'merchant': 'Starbucks',
            'mcc_code': 5812
        }
        
        reward = calculator.calculate_reward(card, transaction)
        
        # 1% of $100 = $1.00
        assert reward == 1.0
    
    def test_calculate_reward_with_annual_fee_amortization(self):
        """Test that annual fee is amortized over spending."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator(amortize_annual_fee=True, amortization_period_months=12)
        
        # Card with $550 annual fee, 1% cashback
        card = {
            'card_id': 'card_002',
            'card_name': 'Premium Card',
            'reward_rates': {'universal_base_rate': 1.0},
            'annual_fee': 550
        }
        
        transaction = {
            'amount': 100.0,
            'category': 'dining',
            'merchant': 'Starbucks',
            'mcc_code': 5812
        }
        
        # For a single transaction, effective reward considers annual fee
        # This is complex - for now just test it calculates something
        reward = calculator.calculate_reward(card, transaction)
        
        # Reward should be less than 1.0 due to annual fee impact
        assert reward < 1.0
    
    def test_calculate_reward_returns_zero_for_zero_amount(self):
        """Test edge case: zero amount transaction."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        card = {
            'card_id': 'card_001',
            'reward_rates': {'universal_base_rate': 1.0},
            'annual_fee': 0
        }
        
        transaction = {
            'amount': 0.0,
            'category': 'dining',
            'merchant': 'Test',
            'mcc_code': 5812
        }
        
        reward = calculator.calculate_reward(card, transaction)
        assert reward == 0.0


class TestMerchantCategoryMapper:
    """Test MCC to category mapping."""
    
    def test_map_mcc_to_category(self):
        """Test mapping MCC code to spending category."""
        from src.model_pipeline.scoring.merchant_mapper import MerchantCategoryMapper
        
        mapper = MerchantCategoryMapper()
        
        # Dining MCC
        category = mapper.map_mcc_to_category(5812)
        assert category == 'dining'
        
        # Travel MCC
        category = mapper.map_mcc_to_category(3000)
        assert category == 'travel'
    
    def test_map_unknown_mcc_returns_general(self):
        """Test unknown MCC returns 'general' category."""
        from src.model_pipeline.scoring.merchant_mapper import MerchantCategoryMapper
        
        mapper = MerchantCategoryMapper()
        
        # Unknown MCC
        category = mapper.map_mcc_to_category(9999)
        assert category in ['general', 'other', 'unknown']


class TestSpendingCapTracker:
    """Test spending cap tracking logic."""
    
    def test_track_spending_within_cap(self):
        """Test tracking spending that's within the cap."""
        from src.model_pipeline.scoring.spending_cap_tracker import SpendingCapTracker
        
        tracker = SpendingCapTracker(user_id='user_0001')
        
        # Card with $6000 quarterly dining cap
        card_id = 'card_001'
        category = 'dining'
        cap = 6000.0
        
        # First transaction
        remaining = tracker.get_remaining_cap(card_id, category, cap, spent_so_far=0)
        assert remaining == 6000.0
        
        # After $100 spending
        tracker.record_transaction(card_id, category, 100.0)
        remaining = tracker.get_remaining_cap(card_id, category, cap, spent_so_far=100)
        assert remaining == 5900.0
    
    def test_spending_exceeds_cap(self):
        """Test behavior when spending exceeds cap."""
        from src.model_pipeline.scoring.spending_cap_tracker import SpendingCapTracker
        
        tracker = SpendingCapTracker(user_id='user_0001')
        
        card_id = 'card_001'
        category = 'dining'
        cap = 100.0
        
        # Already spent $150 (exceeds $100 cap)
        remaining = tracker.get_remaining_cap(card_id, category, cap, spent_so_far=150)
        assert remaining == 0.0  # No cap remaining


class TestEdgeCases:
    """Test edge cases for reward calculation."""
    
    def test_foreign_transaction_fee(self):
        """Test that foreign transaction fees are deducted from rewards."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        # Card with 3% foreign transaction fee
        card = {
            'card_id': 'card_001',
            'reward_rates': {'universal_base_rate': 2.0},
            'annual_fee': 0,
            'foreign_transaction_fee_pct': 3.0
        }
        
        transaction = {
            'amount': 100.0,
            'category': 'travel',
            'merchant': 'Foreign Merchant',
            'mcc_code': 3000,
            'is_foreign': True
        }
        
        reward = calculator.calculate_reward(card, transaction)
        
        # 2% rewards - 3% fee = -1% = -$1.00
        assert reward < 0  # Net negative due to fee
    
    def test_missing_reward_rates_uses_default(self):
        """Test graceful handling of missing reward_rates."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        card = {
            'card_id': 'card_001',
            'card_name': 'Test Card',
            # Missing reward_rates!
            'annual_fee': 0
        }
        
        transaction = {
            'amount': 100.0,
            'category': 'dining',
            'merchant': 'Test',
            'mcc_code': 5812
        }
        
        # Should not crash, should use default rate
        reward = calculator.calculate_reward(card, transaction)
        assert reward >= 0  # Should calculate something reasonable


class TestCategoryBonuses:
    """Test category-specific bonus rates."""
    
    def test_category_bonus_dining(self):
        """Test card with 3x dining bonus."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        # Card with 3x on dining, 1x on everything else
        card = {
            'card_id': 'card_003',
            'card_name': 'Dining Rewards Card',
            'reward_rates': {
                'universal_base_rate': 1.0,
                'category_bonuses': {
                    'dining': 3.0,
                    'travel': 1.0
                }
            },
            'annual_fee': 0
        }
        
        # Dining transaction
        transaction = {
            'amount': 100.0,
            'category': 'dining',
            'merchant': 'Restaurant',
            'mcc_code': 5812
        }
        
        reward = calculator.calculate_reward(card, transaction)
        
        # 3% of $100 = $3.00
        assert reward == 3.0
    
    def test_category_bonus_fallback_to_base(self):
        """Test fallback to base rate for non-bonus category."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        # Card with 3x dining, but transaction is groceries
        card = {
            'card_id': 'card_003',
            'reward_rates': {
                'universal_base_rate': 1.0,
                'category_bonuses': {
                    'dining': 3.0
                }
            },
            'annual_fee': 0
        }
        
        transaction = {
            'amount': 100.0,
            'category': 'groceries',  # Not a bonus category
            'merchant': 'Whole Foods',
            'mcc_code': 5411
        }
        
        reward = calculator.calculate_reward(card, transaction)
        
        # Falls back to 1% base rate
        assert reward == 1.0


class TestRotatingBonuses:
    """Test rotating quarterly bonus categories."""
    
    def test_rotating_bonus_active_quarter(self):
        """Test rotating bonus applies in correct quarter."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        # Card with 5% rotating bonus on gas in Q3 (Jul-Sep)
        card = {
            'card_id': 'card_004',
            'reward_rates': {
                'universal_base_rate': 1.0,
                'rotating_bonuses': {
                    'Q3': {
                        'categories': ['gas'],
                        'rate': 5.0
                    }
                }
            },
            'annual_fee': 0
        }
        
        # Transaction in Q3 (August)
        transaction = {
            'amount': 100.0,
            'category': 'gas',
            'merchant': 'Shell',
            'mcc_code': 5541,
            'date': datetime(2025, 8, 15)  # August = Q3
        }
        
        reward = calculator.calculate_reward(card, transaction)
        
        # 5% of $100 = $5.00
        assert reward == 5.0
    
    def test_rotating_bonus_inactive_quarter(self):
        """Test rotating bonus doesn't apply in wrong quarter."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        # Same card as above
        card = {
            'card_id': 'card_004',
            'reward_rates': {
                'universal_base_rate': 1.0,
                'rotating_bonuses': {
                    'Q3': {
                        'categories': ['gas'],
                        'rate': 5.0
                    }
                }
            },
            'annual_fee': 0
        }
        
        # Transaction in Q1 (January) - rotating bonus not active
        transaction = {
            'amount': 100.0,
            'category': 'gas',
            'merchant': 'Shell',
            'mcc_code': 5541,
            'date': datetime(2025, 1, 15)  # January = Q1
        }
        
        reward = calculator.calculate_reward(card, transaction)
        
        # Falls back to 1% base rate
        assert reward == 1.0


class TestWelcomeBonusEligibility:
    """Test sign-up bonus eligibility logic."""
    
    def test_user_eligible_for_welcome_bonus(self):
        """Test welcome bonus calculation for eligible user."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        card = {
            'card_id': 'card_005',
            'reward_rates': {'universal_base_rate': 1.0},
            'annual_fee': 0,
            'welcome_bonus': {
                'amount': 60000,  # 60k points
                'spend_requirement': 4000,
                'days_to_complete': 90,
                'currency': 'POINTS'
            }
        }
        
        user_status = {
            'user_id': 'user_0001',
            'card_tenure_days': 0,  # New card
            'total_spent_on_card': 0
        }
        
        eligible = calculator.is_welcome_bonus_eligible(card, user_status)
        assert eligible is True
    
    def test_user_ineligible_already_has_bonus(self):
        """Test user who already received the bonus."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        card = {
            'card_id': 'card_005',
            'welcome_bonus': {
                'amount': 60000,
                'spend_requirement': 4000
            }
        }
        
        user_status = {
            'user_id': 'user_0001',
            'card_tenure_days': 200,  # Had card for 200 days
            'total_spent_on_card': 5000,  # Already met requirement
            'welcome_bonus_received': True
        }
        
        eligible = calculator.is_welcome_bonus_eligible(card, user_status)
        assert eligible is False


class TestStatementCredits:
    """Test statement credit offsets."""
    
    def test_statement_credit_offset(self):
        """Test that statement credits increase effective reward value."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator(include_statement_credits=True)
        
        # Card with $10/month dining credit
        card = {
            'card_id': 'card_006',
            'reward_rates': {'universal_base_rate': 1.0},
            'annual_fee': 250,
            'statement_credits': {
                'dining': {
                    'amount': 10.0,
                    'frequency': 'monthly'
                }
            }
        }
        
        transaction = {
            'amount': 50.0,
            'category': 'dining',
            'merchant': 'Restaurant',
            'mcc_code': 5812
        }
        
        # Should factor in the $10 monthly credit for dining
        reward = calculator.calculate_reward_with_credits(card, transaction)
        
        # Reward should be higher than base 1% due to statement credit
        assert reward > 0.5  # More than just 1% of $50
