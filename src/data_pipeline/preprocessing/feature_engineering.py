



"""
Feature Engineering Module for RewardSense

This module provides feature engineering transformations for:
- Credit card data (reward rates, net value, welcome bonuses, credits)
- Transaction data (spending patterns, temporal features, card usage)
- User profile data (point valuations, redemption preferences)

All transformations are deterministic and reproducible.
Matches actual data structure from CreditCardBonuses API and synthetic generators.

Author: RewardSense Team
Date: 2026-02-17
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging
from pathlib import Path
import ast
import json

# Set up logging
logger = logging.getLogger(__name__)


class CreditCardFeatureEngineer:
    """
    Feature engineering for credit card data from CreditCardBonuses API.
    
    Expected schema (from actual API response):
      - card_id, card_name, issuer, network, currency
      - annual_fee, is_annual_fee_waived, is_business, discontinued
      - reward_rates: {"universal_base_rate": float}
      - offers: [{"spend": int, "amount": [{"amount": int}], "days": int}]
      - credits: [{"description": str, "value": float}]
      - universal_cashback_percent
    """
    
    def __init__(self):
        """Initialize the credit card feature engineer."""
        logger.info("Initialized CreditCardFeatureEngineer")
    
    def extract_base_reward_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract universal_base_rate from reward_rates dict.
        
        Args:
            df: DataFrame with reward_rates column
        
        Returns:
            DataFrame with base_reward_rate column
        """
        df = df.copy()
        
        def safe_extract_rate(reward_rates):
            """Safely extract base rate from reward_rates dict."""
            if pd.isna(reward_rates):
                return 1.0  # Default
            
            if isinstance(reward_rates, dict):
                return float(reward_rates.get('universal_base_rate', 1.0))
            
            if isinstance(reward_rates, str):
                try:
                    rates_dict = json.loads(reward_rates)
                    return float(rates_dict.get('universal_base_rate', 1.0))
                except (json.JSONDecodeError, ValueError, KeyError):
                    return 1.0
            
            return 1.0
        
        df['base_reward_rate'] = df['reward_rates'].apply(safe_extract_rate)
        
        # Also use universal_cashback_percent if available
        if 'universal_cashback_percent' in df.columns:
            df['cashback_rate'] = pd.to_numeric(df['universal_cashback_percent'], errors='coerce').fillna(df['base_reward_rate'])
        else:
            df['cashback_rate'] = df['base_reward_rate']
        
        logger.info(f"Extracted base reward rates (mean: {df['base_reward_rate'].mean():.2f}%)")
        return df
    
    def parse_welcome_bonus_offers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse welcome bonus from 'offers' array.
        
        Structure: [{"spend": int, "amount": [{"amount": int}], "days": int, "credits": []}]
        
        Args:
            df: DataFrame with offers column
        
        Returns:
            DataFrame with welcome bonus feature columns
        """
        df = df.copy()
        
        def extract_primary_offer(offers):
            """Extract the primary (current) welcome bonus offer."""
            if pd.isna(offers) or not offers:
                return None
            
            if isinstance(offers, str):
                try:
                    offers = json.loads(offers)
                except (json.JSONDecodeError, ValueError):
                    return None
            
            if isinstance(offers, list) and len(offers) > 0:
                return offers[0]  # First offer is current
            
            return None
        
        # Extract primary offer
        df['_primary_offer'] = df['offers'].apply(extract_primary_offer)
        
        # Extract spend requirement
        df['welcome_bonus_spend_req'] = df['_primary_offer'].apply(
            lambda x: float(x.get('spend', 0)) if x else 0
        )
        
        # Extract bonus amount
        def extract_amount(offer):
            if not offer:
                return 0
            amount_list = offer.get('amount', [])
            if amount_list and isinstance(amount_list, list) and len(amount_list) > 0:
                return float(amount_list[0].get('amount', 0))
            return 0
        
        df['welcome_bonus_amount'] = df['_primary_offer'].apply(extract_amount)
        
        # Extract time limit
        df['welcome_bonus_days'] = df['_primary_offer'].apply(
            lambda x: int(x.get('days', 90)) if x else 90
        )
        
        # Drop temporary column
        df = df.drop('_primary_offer', axis=1)
        
        logger.info(f"Parsed welcome bonus offers for {len(df)} cards")
        return df
    
    def calculate_welcome_bonus_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the USD value of welcome bonuses based on currency type.
        
        Args:
            df: DataFrame with welcome_bonus_amount and currency columns
        
        Returns:
            DataFrame with welcome_bonus_value_usd column
        """
        df = df.copy()
        
        # Currency valuation (cents per point/mile)
        currency_values = {
            'CASHBACK': 100,    # $1 = 100 cents
            'USD': 100,
            'POINTS': 1,        # 1 point = 1 cent (conservative)
            'MILES': 1.2,       # 1 mile = 1.2 cents
            'DELTA': 1.3,       # Delta SkyMiles
            'UNITED': 1.2,      # United MileagePlus
            'AMERICAN': 1.5,    # AA miles
            'MARRIOTT': 0.8,    # Marriott Bonvoy
            'HILTON': 0.5,      # Hilton Honors
            'HYATT': 1.7,       # Hyatt points
        }
        
        # Ensure currency column exists
        if 'currency' not in df.columns:
            df['currency'] = 'POINTS'
        
        # Calculate value in cents
        df['currency'] = df['currency'].fillna('POINTS')
        df['_cents_per_unit'] = df['currency'].map(currency_values).fillna(1.0)
        df['welcome_bonus_value_cents'] = df['welcome_bonus_amount'] * df['_cents_per_unit']
        
        # Convert to USD
        df['welcome_bonus_value_usd'] = df['welcome_bonus_value_cents'] / 100
        
        # Calculate ROI (return on spend requirement)
        df['welcome_bonus_roi'] = np.where(
            df['welcome_bonus_spend_req'] > 0,
            df['welcome_bonus_value_usd'] / df['welcome_bonus_spend_req'],
            0
        )
        
        # Categorize bonus difficulty
        def categorize_difficulty(row):
            spend = row.get('welcome_bonus_spend_req', 0)
            days = row.get('welcome_bonus_days', 90)
            
            if spend == 0:
                return 'none'
            elif spend < 2000 and days >= 90:
                return 'easy'
            elif spend > 5000 or days < 60:
                return 'hard'
            else:
                return 'medium'
        
        df['bonus_difficulty'] = df.apply(categorize_difficulty, axis=1)
        
        # Clean up
        df = df.drop('_cents_per_unit', axis=1)
        
        logger.info(f"Calculated welcome bonus values (avg: ${df['welcome_bonus_value_usd'].mean():.2f})")
        return df
    
    def parse_credits_benefits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse credits/benefits array and calculate total value.
        
        Structure: [{"description": str, "value": float, "weight": float}]
        
        Args:
            df: DataFrame with credits column
        
        Returns:
            DataFrame with credits feature columns
        """
        df = df.copy()
        
        def extract_credits_value(credits):
            """Extract total annual value from credits."""
            if pd.isna(credits) or not credits:
                return 0, 0
            
            if isinstance(credits, str):
                try:
                    credits = json.loads(credits)
                except (json.JSONDecodeError, ValueError):
                    return 0, 0
            
            if not isinstance(credits, list):
                return 0, 0
            
            total_value = sum(c.get('value', 0) for c in credits)
            count = len(credits)
            
            return total_value, count
        
        df['_credits_parsed'] = df['credits'].apply(extract_credits_value)
        df['annual_credits_value'] = df['_credits_parsed'].apply(lambda x: x[0])
        df['num_credits'] = df['_credits_parsed'].apply(lambda x: x[1])
        
        # Has credits flag
        df['has_credits'] = (df['num_credits'] > 0).astype(int)
        
        df = df.drop('_credits_parsed', axis=1)
        
        logger.info(f"Parsed credits/benefits (avg value: ${df['annual_credits_value'].mean():.2f})")
        return df
    
    def calculate_effective_annual_fee(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate effective annual fee after credits and first-year waiver.
        
        Args:
            df: DataFrame with annual_fee, is_annual_fee_waived, annual_credits_value
        
        Returns:
            DataFrame with effective fee columns
        """
        df = df.copy()
        
        # Ensure columns exist
        annual_fee = pd.to_numeric(df.get('annual_fee', 0), errors='coerce').fillna(0)
        is_waived = df.get('is_annual_fee_waived', False).fillna(False)
        credits_value = df.get('annual_credits_value', 0)
        
        # Effective fee after credits
        df['effective_annual_fee'] = annual_fee - credits_value
        
        # First year effective fee (considering waiver)
        df['effective_fee_year1'] = np.where(
            is_waived,
            -credits_value,  # Negative = profit!
            df['effective_annual_fee']
        )
        
        # Net annual cost (ongoing)
        df['net_annual_cost'] = df['effective_annual_fee']
        
        logger.info(f"Calculated effective fees (avg: ${df['effective_annual_fee'].mean():.2f})")
        return df
    
    def calculate_net_value(self, df: pd.DataFrame,
                          annual_spending: float = 25000) -> pd.DataFrame:
        """
        Calculate net value: (expected rewards - effective annual fee).
        
        Args:
            df: DataFrame with reward rates and fees
            annual_spending: Assumed annual spending (default: $25,000)
        
        Returns:
            DataFrame with net value columns
        """
        df = df.copy()
        
        # Calculate expected annual rewards
        # Using base_reward_rate as percentage
        df['expected_annual_rewards'] = annual_spending * (df['base_reward_rate'] / 100)
        
        # Net value (rewards - effective fee)
        df['net_value_annual'] = df['expected_annual_rewards'] - df['effective_annual_fee']
        
        # Net value first year (including welcome bonus)
        df['net_value_year1'] = (
            df['expected_annual_rewards'] - 
            df['effective_fee_year1'] + 
            df['welcome_bonus_value_usd']
        )
        
        # Value per dollar spent
        df['value_per_dollar'] = df['net_value_annual'] / annual_spending
        
        logger.info(f"Calculated net values (avg annual: ${df['net_value_annual'].mean():.2f})")
        return df
    
    def filter_active_cards(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out discontinued cards or create flag.
        
        Args:
            df: DataFrame with discontinued column
        
        Returns:
            DataFrame with is_active flag
        """
        df = df.copy()
        
        if 'discontinued' in df.columns:
            df['is_active'] = (~df['discontinued']).astype(int)
            df['is_discontinued'] = df['discontinued'].astype(int)
            
            discontinued_count = df['discontinued'].sum()
            logger.info(f"Found {discontinued_count} discontinued cards out of {len(df)}")
        else:
            df['is_active'] = 1
            df['is_discontinued'] = 0
        
        return df
    
    def create_issuer_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from issuer and network information.
        
        Args:
            df: DataFrame with issuer and network columns
        
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        # Normalize issuer names (they're already uppercase with underscores)
        if 'issuer' in df.columns:
            df['issuer_clean'] = df['issuer'].str.replace('_', ' ').str.title()
            
            # One-hot encode issuer
            issuer_dummies = pd.get_dummies(df['issuer'], prefix='issuer', prefix_sep='_')
            df = pd.concat([df, issuer_dummies], axis=1)
            logger.info(f"Encoded {len(issuer_dummies.columns)} issuers")
        
        # One-hot encode network
        if 'network' in df.columns:
            network_dummies = pd.get_dummies(df['network'], prefix='network', prefix_sep='_')
            df = pd.concat([df, network_dummies], axis=1)
            logger.info(f"Encoded {len(network_dummies.columns)} networks")
        
        # Card tier features based on annual fee
        if 'annual_fee' in df.columns:
            df['is_premium'] = (df['annual_fee'] >= 450).astype(int)
            df['is_mid_tier'] = ((df['annual_fee'] >= 95) & (df['annual_fee'] < 450)).astype(int)
            df['is_no_annual_fee'] = (df['annual_fee'] == 0).astype(int)
        
        # Business card flag
        if 'is_business' in df.columns:
            df['is_business'] = df['is_business'].fillna(False).astype(int)
        
        # Currency type encoding
        if 'currency' in df.columns:
            currency_dummies = pd.get_dummies(df['currency'], prefix='currency', prefix_sep='_')
            df = pd.concat([df, currency_dummies], axis=1)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame,
                         annual_spending: float = 25000) -> pd.DataFrame:
        """
        Apply all credit card feature engineering transformations.
        
        Args:
            df: Raw credit card DataFrame from API
            annual_spending: Assumed annual spending for calculations
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Engineering features for {len(df)} credit cards")
        
        # Filter/flag discontinued cards (DO THIS FIRST)
        df = self.filter_active_cards(df)
        
        # Extract base reward rates
        df = self.extract_base_reward_rate(df)
        
        # Parse welcome bonus offers
        df = self.parse_welcome_bonus_offers(df)
        
        # Calculate welcome bonus value
        df = self.calculate_welcome_bonus_value(df)
        
        # Parse credits/benefits
        df = self.parse_credits_benefits(df)
        
        # Calculate effective annual fee
        df = self.calculate_effective_annual_fee(df)
        
        # Calculate net value
        df = self.calculate_net_value(df, annual_spending)
        
        # Create issuer/network features
        df = self.create_issuer_network_features(df)
        
        logger.info(f"Credit card feature engineering complete: {df.shape}")
        return df


class TransactionFeatureEngineer:
    """
    Feature engineering for transaction data.
    
    Schema: transaction_id, user_id, date, category, merchant, mcc_code, amount, card_used
    Categories: dining, travel, online_shopping, utilities, entertainment, groceries, gas
    """
    
    def __init__(self):
        """Initialize the transaction feature engineer."""
        # Use actual categories from generated data
        self.standard_categories = [
            'dining', 'travel', 'online_shopping', 'utilities', 
            'entertainment', 'groceries', 'gas'
        ]
        logger.info("Initialized TransactionFeatureEngineer")
    
    def aggregate_spending_by_category(self, df: pd.DataFrame,
                                      user_id_col: str = 'user_id',
                                      category_col: str = 'category',
                                      amount_col: str = 'amount') -> pd.DataFrame:
        """
        Aggregate transaction spending by user and category.
        
        Args:
            df: Transaction DataFrame
            user_id_col: User ID column
            category_col: Category column
            amount_col: Amount column
        
        Returns:
            DataFrame with aggregated spending per user per category
        """
        logger.info(f"Aggregating spending for {df[user_id_col].nunique()} users")
        
        # Group by user and category
        agg_df = df.groupby([user_id_col, category_col])[amount_col].agg([
            ('total_spent', 'sum'),
            ('transaction_count', 'count'),
            ('avg_transaction', 'mean'),
            ('max_transaction', 'max'),
            ('min_transaction', 'min'),
            ('std_transaction', 'std')
        ]).reset_index()
        
        # Pivot to wide format
        spending_pivot = agg_df.pivot(
            index=user_id_col,
            columns=category_col,
            values='total_spent'
        ).fillna(0)
        spending_pivot.columns = [f'{col}_total_spent' for col in spending_pivot.columns]
        
        # Pivot transaction counts
        count_pivot = agg_df.pivot(
            index=user_id_col,
            columns=category_col,
            values='transaction_count'
        ).fillna(0)
        count_pivot.columns = [f'{col}_txn_count' for col in count_pivot.columns]
        
        # Merge
        result = spending_pivot.reset_index()
        result = result.merge(count_pivot.reset_index(), on=user_id_col, how='left')
        
        # Add total spending and transaction count
        spending_cols = [c for c in result.columns if c.endswith('_total_spent')]
        result['total_spending'] = result[spending_cols].sum(axis=1)
        
        count_cols = [c for c in result.columns if c.endswith('_txn_count')]
        result['total_transactions'] = result[count_cols].sum(axis=1)
        
        # Add spending diversity (entropy)
        def calculate_spending_entropy(row):
            """Calculate entropy of spending distribution across categories."""
            spending_vals = [row[c] for c in spending_cols if row[c] > 0]
            if not spending_vals or sum(spending_vals) == 0:
                return 0
            probs = np.array(spending_vals) / sum(spending_vals)
            return -np.sum(probs * np.log2(probs + 1e-10))
        
        result['spending_diversity'] = result.apply(calculate_spending_entropy, axis=1)
        
        logger.info(f"Created spending aggregations for {len(result)} users")
        return result
    
    def extract_temporal_patterns(self, df: pd.DataFrame,
                                  date_col: str = 'date',
                                  amount_col: str = 'amount',
                                  user_id_col: str = 'user_id') -> pd.DataFrame:
        """
        Extract temporal spending patterns.
        
        Args:
            df: Transaction DataFrame
            date_col: Date column
            amount_col: Amount column
            user_id_col: User ID column
        
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Ensure datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract temporal features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)
        
        # Aggregate by user
        temporal_features = df.groupby(user_id_col).agg({
            'is_weekend': 'mean',  # Weekend spending ratio
            'month': lambda x: x.mode()[0] if len(x) > 0 else 1,  # Peak month
            'day_of_week': lambda x: x.mode()[0] if len(x) > 0 else 0,  # Peak day
            amount_col: ['mean', 'std', 'median', 'sum']
        }).reset_index()
        
        temporal_features.columns = [
            user_id_col,
            'weekend_spending_ratio',
            'peak_spending_month',
            'peak_spending_day',
            'avg_transaction_amount',
            'transaction_amount_std',
            'median_transaction_amount',
            'total_spending_temporal'
        ]
        
        logger.info(f"Extracted temporal patterns for {len(temporal_features)} users")
        return temporal_features
    
    def analyze_card_usage_patterns(self, df: pd.DataFrame,
                                   user_id_col: str = 'user_id',
                                   card_col: str = 'card_used',
                                   category_col: str = 'category') -> pd.DataFrame:
        """
        Analyze card usage patterns.
        
        Args:
            df: Transaction DataFrame
            user_id_col: User ID column
            card_col: Card used column
            category_col: Category column
        
        Returns:
            DataFrame with card usage features
        """
        logger.info("Analyzing card usage patterns")
        
        # Unique cards per user
        cards_per_user = df.groupby(user_id_col)[card_col].nunique().reset_index()
        cards_per_user.columns = [user_id_col, 'num_cards_used']
        
        # Most used card
        most_used_card = df.groupby(user_id_col)[card_col].agg(
            lambda x: x.mode()[0] if len(x) > 0 else None
        ).reset_index()
        most_used_card.columns = [user_id_col, 'primary_card']
        
        # Card switching frequency
        def calc_switch_rate(cards):
            if len(cards) <= 1:
                return 0
            switches = sum(1 for i in range(1, len(cards)) if cards.iloc[i] != cards.iloc[i-1])
            return switches / (len(cards) - 1)
        
        switch_rate = df.groupby(user_id_col)[card_col].apply(calc_switch_rate).reset_index()
        switch_rate.columns = [user_id_col, 'card_switch_rate']
        
        # Merge
        card_features = cards_per_user.merge(most_used_card, on=user_id_col)
        card_features = card_features.merge(switch_rate, on=user_id_col)
        
        logger.info(f"Created card usage features for {len(card_features)} users")
        return card_features
    
    def analyze_mcc_patterns(self, df: pd.DataFrame,
                            user_id_col: str = 'user_id',
                            mcc_col: str = 'mcc_code',
                            amount_col: str = 'amount') -> pd.DataFrame:
        """
        Analyze MCC code patterns.
        
        Args:
            df: Transaction DataFrame
            user_id_col: User ID column
            mcc_col: MCC code column
            amount_col: Amount column
        
        Returns:
            DataFrame with MCC features
        """
        logger.info("Analyzing MCC patterns")
        
        # Unique MCCs
        mcc_diversity = df.groupby(user_id_col)[mcc_col].nunique().reset_index()
        mcc_diversity.columns = [user_id_col, 'num_unique_mccs']
        
        # Most common MCC
        most_common_mcc = df.groupby(user_id_col)[mcc_col].agg(
            lambda x: x.mode()[0] if len(x) > 0 else None
        ).reset_index()
        most_common_mcc.columns = [user_id_col, 'primary_mcc']
        
        # Average spending per MCC
        avg_per_mcc = df.groupby(user_id_col).apply(
            lambda x: x.groupby(mcc_col)[amount_col].mean().mean() if len(x) > 0 else 0
        ).reset_index()
        avg_per_mcc.columns = [user_id_col, 'avg_spending_per_mcc']
        
        # Merge
        mcc_features = mcc_diversity.merge(most_common_mcc, on=user_id_col)
        mcc_features = mcc_features.merge(avg_per_mcc, on=user_id_col)
        
        logger.info(f"Created MCC features for {len(mcc_features)} users")
        return mcc_features
    
    def create_merchant_features(self, df: pd.DataFrame,
                                merchant_col: str = 'merchant',
                                user_id_col: str = 'user_id') -> pd.DataFrame:
        """
        Create merchant pattern features.
        
        Args:
            df: Transaction DataFrame
            merchant_col: Merchant column
            user_id_col: User ID column
        
        Returns:
            DataFrame with merchant features
        """
        logger.info("Creating merchant features")
        
        # Unique merchants
        merchant_diversity = df.groupby(user_id_col)[merchant_col].nunique().reset_index()
        merchant_diversity.columns = [user_id_col, 'num_unique_merchants']
        
        # Favorite merchant
        favorite_merchant = df.groupby(user_id_col)[merchant_col].agg(
            lambda x: x.mode()[0] if len(x) > 0 else None
        ).reset_index()
        favorite_merchant.columns = [user_id_col, 'favorite_merchant']
        
        # Repeat merchant ratio
        def calc_repeat_ratio(merchants):
            total = len(merchants)
            unique = merchants.nunique()
            return (total - unique) / total if total > 0 else 0
        
        repeat_ratio = df.groupby(user_id_col)[merchant_col].apply(calc_repeat_ratio).reset_index()
        repeat_ratio.columns = [user_id_col, 'repeat_merchant_ratio']
        
        # Merge
        merchant_features = merchant_diversity.merge(favorite_merchant, on=user_id_col)
        merchant_features = merchant_features.merge(repeat_ratio, on=user_id_col)
        
        logger.info(f"Created merchant features for {len(merchant_features)} users")
        return merchant_features
    
    def handle_suspicious_transactions(self, df: pd.DataFrame,
                                      user_id_col: str = 'user_id') -> pd.DataFrame:
        """
        Handle suspicious flag from cleaning module.
        
        Args:
            df: Transaction DataFrame (possibly with 'suspicious' column from cleaning)
            user_id_col: User ID column
        
        Returns:
            DataFrame with suspicious transaction features
        """
        if 'suspicious' not in df.columns:
            return pd.DataFrame()  # Return empty DataFrame
        
        logger.info("Handling suspicious transactions")
        
        # Aggregate by user
        suspicious_features = df.groupby(user_id_col)['suspicious'].agg([
            ('num_suspicious', 'sum'),
            ('suspicious_rate', 'mean')
        ]).reset_index()
        
        return suspicious_features
    
    def engineer_features(self, df: pd.DataFrame,
                         user_id_col: str = 'user_id',
                         date_col: str = 'date',
                         amount_col: str = 'amount',
                         merchant_col: str = 'merchant',
                         category_col: str = 'category',
                         card_col: str = 'card_used',
                         mcc_col: str = 'mcc_code') -> pd.DataFrame:
        """
        Apply all transaction feature engineering transformations.
        
        Args:
            df: Raw transaction DataFrame
            All column names with defaults
        
        Returns:
            DataFrame with all engineered features per user
        """
        logger.info(f"Engineering features for {len(df)} transactions")
        
        # Create all feature sets
        spending_features = self.aggregate_spending_by_category(df, user_id_col, category_col, amount_col)
        temporal_features = self.extract_temporal_patterns(df, date_col, amount_col, user_id_col)
        card_features = self.analyze_card_usage_patterns(df, user_id_col, card_col, category_col)
        mcc_features = self.analyze_mcc_patterns(df, user_id_col, mcc_col, amount_col)
        merchant_features = self.create_merchant_features(df, merchant_col, user_id_col)
        
        # Handle suspicious transactions if column exists
        suspicious_features = self.handle_suspicious_transactions(df, user_id_col)
        
        # Merge all
        features_df = spending_features
        for feat_df in [temporal_features, card_features, mcc_features, merchant_features]:
            features_df = features_df.merge(feat_df, on=user_id_col, how='outer')
        
        if not suspicious_features.empty:
            features_df = features_df.merge(suspicious_features, on=user_id_col, how='left')
        
        logger.info(f"Transaction feature engineering complete: {features_df.shape}")
        return features_df


class UserProfileFeatureEngineer:
    """
    Feature engineering for user profile data.
    
    Schema: user_id, archetype, monthly_budget, cards, redemption_preference, age_group, location_type
    """
    
    def __init__(self):
        """Initialize the user profile feature engineer."""
        logger.info("Initialized UserProfileFeatureEngineer")
    
    def parse_cards_column(self, df: pd.DataFrame,
                          cards_col: str = 'cards') -> pd.DataFrame:
        """Parse the 'cards' column (string representation of list)."""
        df = df.copy()
        
        def safe_parse_cards(cards_str):
            if pd.isna(cards_str):
                return []
            try:
                return ast.literal_eval(str(cards_str))
            except (ValueError, SyntaxError):
                # Try splitting by comma as fallback
                return [c.strip() for c in str(cards_str).strip("[]'\"").split(',') if c.strip()]
        
        df['cards_list'] = df[cards_col].apply(safe_parse_cards)
        df['num_cards'] = df['cards_list'].apply(len)
        
        logger.info(f"Parsed cards for {len(df)} users (avg: {df['num_cards'].mean():.2f})")
        return df
    
    def encode_archetype(self, df: pd.DataFrame,
                        archetype_col: str = 'archetype') -> pd.DataFrame:
        """One-hot encode user archetype."""
        df = df.copy()
        archetype_dummies = pd.get_dummies(df[archetype_col], prefix='archetype', prefix_sep='_')
        df = pd.concat([df, archetype_dummies], axis=1)
        logger.info(f"Encoded {len(archetype_dummies.columns)} archetypes")
        return df
    
    def encode_age_group(self, df: pd.DataFrame,
                        age_col: str = 'age_group') -> pd.DataFrame:
        """Encode age group (ordinal + one-hot)."""
        df = df.copy()
        
        # Ordinal encoding
        age_order = {'18-25': 1, '26-35': 2, '36-50': 3, '51-65': 4, '65+': 5}
        df['age_group_ordinal'] = df[age_col].map(age_order).fillna(0)
        
        # One-hot encoding
        age_dummies = pd.get_dummies(df[age_col], prefix='age', prefix_sep='_')
        df = pd.concat([df, age_dummies], axis=1)
        
        logger.info(f"Encoded {len(age_dummies.columns)} age groups")
        return df
    
    def encode_location_type(self, df: pd.DataFrame,
                            location_col: str = 'location_type') -> pd.DataFrame:
        """One-hot encode location type."""
        df = df.copy()
        location_dummies = pd.get_dummies(df[location_col], prefix='location', prefix_sep='_')
        df = pd.concat([df, location_dummies], axis=1)
        logger.info(f"Encoded {len(location_dummies.columns)} location types")
        return df
    
    def create_budget_features(self, df: pd.DataFrame,
                              budget_col: str = 'monthly_budget') -> pd.DataFrame:
        """Create budget-derived features."""
        df = df.copy()
        
        df[budget_col] = pd.to_numeric(df[budget_col], errors='coerce').fillna(0)
        
        # Log transform
        df['monthly_budget_log'] = np.log1p(df[budget_col])
        
        # Annual budget
        df['annual_budget'] = df[budget_col] * 12
        
        # Budget quartiles
        df['budget_quartile'] = pd.qcut(
            df[budget_col], 
            q=4, 
            labels=['Q1_low', 'Q2_medium_low', 'Q3_medium_high', 'Q4_high'],
            duplicates='drop'
        )
        
        quartile_dummies = pd.get_dummies(df['budget_quartile'], prefix='budget', prefix_sep='_')
        df = pd.concat([df, quartile_dummies], axis=1)
        
        logger.info(f"Created budget features (range: ${df[budget_col].min():.2f} - ${df[budget_col].max():.2f})")
        return df
    
    def estimate_point_valuations(self, df: pd.DataFrame,
                                  redemption_col: str = 'redemption_preference') -> pd.DataFrame:
        """Estimate point valuations based on redemption preferences."""
        df = df.copy()
        
        # Valuation map
        valuation_map = {
            'cash_back': 0.01,
            'statement_credit': 0.01,
            'travel_portal': 0.015,
            'travel_transfer': 0.02,
            'gift_cards': 0.009,
            'merchandise': 0.008
        }
        
        df['estimated_point_value'] = df[redemption_col].map(valuation_map).fillna(0.01)
        
        logger.info(f"Estimated point valuations (avg: ${df['estimated_point_value'].mean():.4f})")
        return df
    
    def encode_redemption_preferences(self, df: pd.DataFrame,
                                     redemption_col: str = 'redemption_preference') -> pd.DataFrame:
        """One-hot encode redemption preferences."""
        df = df.copy()
        redemption_dummies = pd.get_dummies(df[redemption_col], prefix='redemption', prefix_sep='_')
        df = pd.concat([df, redemption_dummies], axis=1)
        logger.info(f"Encoded {len(redemption_dummies.columns)} redemption preferences")
        return df
    
    def engineer_features(self, df: pd.DataFrame,
                         user_id_col: str = 'user_id',
                         archetype_col: str = 'archetype',
                         budget_col: str = 'monthly_budget',
                         cards_col: str = 'cards',
                         redemption_col: str = 'redemption_preference',
                         age_col: str = 'age_group',
                         location_col: str = 'location_type') -> pd.DataFrame:
        """Apply all user profile feature engineering transformations."""
        logger.info(f"Engineering features for {len(df)} user profiles")
        
        df = self.parse_cards_column(df, cards_col)
        df = self.encode_archetype(df, archetype_col)
        df = self.encode_age_group(df, age_col)
        df = self.encode_location_type(df, location_col)
        df = self.create_budget_features(df, budget_col)
        df = self.estimate_point_valuations(df, redemption_col)
        df = self.encode_redemption_preferences(df, redemption_col)
        
        logger.info("User profile feature engineering complete")
        return df


# Convenience function
def engineer_all_features(credit_cards_df: Optional[pd.DataFrame] = None,
                         transactions_df: Optional[pd.DataFrame] = None,
                         users_df: Optional[pd.DataFrame] = None,
                         annual_spending: float = 25000,
                         output_dir: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Engineer features for all datasets.
    
    Matches actual data structure from:
    - CreditCardBonuses API (credit cards)
    - Synthetic generators (transactions, user profiles)
    
    Args:
        credit_cards_df: Credit card data from API
        transactions_df: Transaction data
        users_df: User profile data
        annual_spending: Annual spending for card value calculations
        output_dir: Optional directory to save features
    
    Returns:
        Tuple of (engineered_cards, engineered_transactions, engineered_users)
    """
    logger.info("=" * 70)
    logger.info("RewardSense Feature Engineering Pipeline")
    logger.info("Matches actual data structure from API and generators")
    logger.info("=" * 70)
    
    results = []
    
    # Credit cards
    if credit_cards_df is not None:
        logger.info("\n[1/3] Engineering credit card features...")
        card_engineer = CreditCardFeatureEngineer()
        cards_features = card_engineer.engineer_features(credit_cards_df, annual_spending)
        results.append(cards_features)
        if output_dir:
            output_path = Path(output_dir) / 'credit_cards_features.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cards_features.to_csv(output_path, index=False)
            logger.info(f"✅ Saved to {output_path}")
    else:
        results.append(None)
    
    # Transactions
    if transactions_df is not None:
        logger.info("\n[2/3] Engineering transaction features...")
        txn_engineer = TransactionFeatureEngineer()
        txn_features = txn_engineer.engineer_features(transactions_df)
        results.append(txn_features)
        if output_dir:
            output_path = Path(output_dir) / 'transactions_features.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            txn_features.to_csv(output_path, index=False)
            logger.info(f"✅ Saved to {output_path}")
    else:
        results.append(None)
    
    # User profiles
    if users_df is not None:
        logger.info("\n[3/3] Engineering user profile features...")
        user_engineer = UserProfileFeatureEngineer()
        user_features = user_engineer.engineer_features(users_df)
        results.append(user_features)
        if output_dir:
            output_path = Path(output_dir) / 'users_features.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            user_features.to_csv(output_path, index=False)
            logger.info(f"✅ Saved to {output_path}")
    else:
        results.append(None)
    
    logger.info("\n" + "=" * 70)
    logger.info("Feature Engineering Pipeline Complete!")
    logger.info("=" * 70)
    
    return tuple(results)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("RewardSense Feature Engineering Module")
    print("=" * 70)
    print("\n✅ Matches actual data structure:")
    print("  📄 Credit Cards: CreditCardBonuses API format")
    print("     - reward_rates.universal_base_rate")
    print("     - offers array for welcome bonuses")
    print("     - credits array for benefits")
    print("     - discontinued flag")
    print("  📄 Transactions: transaction_id, user_id, date, category, ...")
    print("     - Categories: dining, travel, utilities, entertainment, etc.")
    print("  📄 User Profiles: user_id, archetype, monthly_budget, ...")
    print("\nUsage:")
    print("  from src.data_pipeline.preprocessing.feature_engineering import engineer_all_features")
    print("  cards_f, txns_f, users_f = engineer_all_features(cards_df, txns_df, users_df)")
    print("=" * 70)
