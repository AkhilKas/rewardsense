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

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Credit cards
# =============================================================================


class CreditCardFeatureEngineer:
    """
    Feature engineering for credit card data from CreditCardBonuses API.

    Expected schema (from actual API response):
      - card_id, card_name, issuer, network, currency
      - annual_fee, is_annual_fee_waived, is_business, discontinued
      - reward_rates: {"universal_base_rate": float}  (or other variants)
      - offers: [{"spend": int, "amount": [{"amount": int}], "days": int}]
      - credits: [{"description": str, "value": float}]
      - universal_cashback_percent
    """

    def __init__(self):
        logger.info("Initialized CreditCardFeatureEngineer")

    # ----------------------------
    # Reward rate extraction
    # ----------------------------

    def extract_base_reward_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Populate df['base_reward_rate'] from whatever schema we received.

        Supported inputs:
        - reward_rates (object): dict/list/json-string
        - reward_rates.universal_base_rate (flattened)
        - universal_cashback_percent
        - base_reward_rate (already exists)
        - reward_rate / base_rate (common alternates)
        """
        if df is None or df.empty:
            return df

        # 1) Already computed
        if "base_reward_rate" in df.columns:
            return df

        # 2) Common flattened columns / alternates
        for c in [
            "reward_rates.universal_base_rate",
            "reward_rates_universal_base_rate",
            "reward_rates.base",
            "reward_rates_base",
            "universal_cashback_percent",
            "base_rate",
            "reward_rate",
        ]:
            if c in df.columns:
                df["base_reward_rate"] = pd.to_numeric(df[c], errors="coerce").fillna(
                    0.0
                )
                return df

        # 3) Object column case
        if "reward_rates" not in df.columns:
            df["base_reward_rate"] = 0.0
            return df

        def safe_extract_rate(x):
            if x is None or x is pd.NA:
                return np.nan

            # numpy scalar NaN
            if isinstance(x, (float, np.floating)) and np.isnan(x):
                return np.nan

            # If it's a JSON string, parse it
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return np.nan
                try:
                    x = json.loads(s)
                except Exception:
                    return np.nan

            # Dict: try common keys
            if isinstance(x, dict):
                for k in [
                    "universal_base_rate",
                    "universal_cashback_percent",
                    "base",
                    "base_rate",
                    "default",
                    "flat",
                    "all",
                    "everything_else",
                ]:
                    if k in x:
                        try:
                            return float(x[k])
                        except Exception:
                            pass

                # some schemas store lists under nested keys
                for k in ["rates", "reward_rates", "categories"]:
                    if k in x:
                        x = x[k]
                        break

            # normalize numpy arrays -> lists
            if isinstance(x, np.ndarray):
                x = x.tolist()

            # List: pick a sensible default
            if isinstance(x, list) and len(x) > 0:
                if isinstance(x[0], dict):
                    # try extracting numeric fields
                    vals = []
                    for item in x:
                        if not isinstance(item, dict):
                            continue
                        for key in ["universal_base_rate", "base", "rate", "value"]:
                            if key in item:
                                try:
                                    vals.append(float(item[key]))
                                except Exception:
                                    pass
                    if vals:
                        return float(max(vals))
                else:
                    vals = []
                    for item in x:
                        try:
                            vals.append(float(item))
                        except Exception:
                            pass
                    if vals:
                        return float(max(vals))

            return np.nan

        df["base_reward_rate"] = df["reward_rates"].apply(safe_extract_rate).fillna(0.0)
        return df

    # ----------------------------
    # Welcome bonus parsing
    # ----------------------------

    def parse_welcome_bonus_offers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse welcome bonus from 'offers' array.

        Structure: [{"spend": int, "amount": [{"amount": int}], "days": int, ...}]
        """
        df = df.copy()

        # If flattened and offers is missing, default safely
        if "offers" not in df.columns:
            df["welcome_bonus_spend_req"] = 0.0
            df["welcome_bonus_amount"] = 0.0
            df["welcome_bonus_days"] = 90
            logger.info("No 'offers' column found; defaulted welcome bonus features.")
            return df

        def extract_primary_offer(offers):
            # None / pandas NA
            if offers is None or offers is pd.NA:
                return None

            # numpy array -> list
            if isinstance(offers, np.ndarray):
                offers = offers.tolist()

            # scalar NaN
            if isinstance(offers, (float, np.floating)) and np.isnan(offers):
                return None

            # JSON string -> object
            if isinstance(offers, str):
                s = offers.strip()
                if not s:
                    return None
                try:
                    offers = json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    return None

            # Must be a list
            if not isinstance(offers, list) or len(offers) == 0:
                return None

            first = offers[0]
            return first if isinstance(first, dict) else None

        df["_primary_offer"] = df["offers"].apply(extract_primary_offer)

        df["welcome_bonus_spend_req"] = df["_primary_offer"].apply(
            lambda x: float(x.get("spend", 0)) if isinstance(x, dict) else 0.0
        )

        def extract_amount(offer):
            if not isinstance(offer, dict):
                return 0.0
            amount_list = offer.get("amount", [])

            if isinstance(amount_list, np.ndarray):
                amount_list = amount_list.tolist()

            if isinstance(amount_list, list) and len(amount_list) > 0:
                first = amount_list[0]
                if isinstance(first, dict):
                    try:
                        return float(first.get("amount", 0) or 0)
                    except Exception:
                        return 0.0
            return 0.0

        df["welcome_bonus_amount"] = df["_primary_offer"].apply(extract_amount)

        df["welcome_bonus_days"] = df["_primary_offer"].apply(
            lambda x: int(x.get("days", 90)) if isinstance(x, dict) else 90
        )

        df = df.drop("_primary_offer", axis=1)
        logger.info(f"Parsed welcome bonus offers for {len(df)} cards")
        return df

    # ----------------------------
    # Welcome bonus valuation
    # ----------------------------

    def calculate_welcome_bonus_value(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        currency_values = {
            "CASHBACK": 100,
            "USD": 100,
            "POINTS": 1.0,
            "MILES": 1.2,
            "DELTA": 1.3,
            "UNITED": 1.2,
            "AMERICAN": 1.5,
            "MARRIOTT": 0.8,
            "HILTON": 0.5,
            "HYATT": 1.7,
        }

        if "currency" not in df.columns:
            df["currency"] = "POINTS"

        # ensure required columns exist
        if "welcome_bonus_amount" not in df.columns:
            df["welcome_bonus_amount"] = 0.0
        if "welcome_bonus_spend_req" not in df.columns:
            df["welcome_bonus_spend_req"] = 0.0
        if "welcome_bonus_days" not in df.columns:
            df["welcome_bonus_days"] = 90

        df["currency"] = df["currency"].fillna("POINTS")
        df["_cents_per_unit"] = df["currency"].map(currency_values).fillna(1.0)

        df["welcome_bonus_value_cents"] = (
            df["welcome_bonus_amount"] * df["_cents_per_unit"]
        )
        df["welcome_bonus_value_usd"] = df["welcome_bonus_value_cents"] / 100.0

        df["welcome_bonus_roi"] = np.where(
            df["welcome_bonus_spend_req"] > 0,
            df["welcome_bonus_value_usd"] / df["welcome_bonus_spend_req"],
            0.0,
        )

        def categorize_difficulty(row):
            spend = float(row.get("welcome_bonus_spend_req", 0) or 0)
            days = int(row.get("welcome_bonus_days", 90) or 90)

            if spend == 0:
                return "none"
            if spend < 2000 and days >= 90:
                return "easy"
            if spend > 5000 or days < 60:
                return "hard"
            return "medium"

        df["bonus_difficulty"] = df.apply(categorize_difficulty, axis=1)

        df = df.drop("_cents_per_unit", axis=1)

        logger.info(
            f"Calculated welcome bonus values (avg: ${df['welcome_bonus_value_usd'].mean():.2f})"
        )
        return df

    # ----------------------------
    # Credits parsing (FIXED)
    # ----------------------------

    def parse_credits_benefits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse credits/benefits array and calculate total value.

        Structure: [{"description": str, "value": float, "weight": float}]
        """
        df = df.copy()

        # If flattened and "credits" doesn't exist, default safely
        if "credits" not in df.columns:
            df["annual_credits_value"] = 0.0
            df["num_credits"] = 0
            df["has_credits"] = 0
            logger.info("No 'credits' column found; defaulted credits features.")
            return df

        def extract_credits_value(credits):
            # None / pandas NA
            if credits is None or credits is pd.NA:
                return 0.0, 0

            # numpy array -> list
            if isinstance(credits, np.ndarray):
                credits = credits.tolist()

            # scalar NaN
            if isinstance(credits, (float, np.floating)) and np.isnan(credits):
                return 0.0, 0

            # JSON string -> object
            if isinstance(credits, str):
                s = credits.strip()
                if not s:
                    return 0.0, 0
                try:
                    credits = json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    return 0.0, 0

            # Must be list
            if not isinstance(credits, list) or len(credits) == 0:
                return 0.0, 0

            total_value = 0.0
            count = 0
            for c in credits:
                if not isinstance(c, dict):
                    continue
                try:
                    total_value += float(c.get("value", 0) or 0)
                except Exception:
                    total_value += 0.0
                count += 1

            return float(total_value), int(count)

        df["_credits_parsed"] = df["credits"].apply(extract_credits_value)
        df["annual_credits_value"] = df["_credits_parsed"].apply(lambda x: float(x[0]))
        df["num_credits"] = df["_credits_parsed"].apply(lambda x: int(x[1]))
        df["has_credits"] = (df["num_credits"] > 0).astype(int)

        df = df.drop("_credits_parsed", axis=1)

        logger.info(
            f"Parsed credits/benefits (avg value: ${df['annual_credits_value'].mean():.2f})"
        )
        return df

    # ----------------------------
    # Fee + net value
    # ----------------------------

    def calculate_effective_annual_fee(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        annual_fee = pd.to_numeric(df.get("annual_fee", 0), errors="coerce").fillna(0.0)
        is_waived = df.get("is_annual_fee_waived", False)
        if isinstance(is_waived, pd.Series):
            is_waived = is_waived.fillna(False).astype(bool)
        else:
            is_waived = bool(is_waived)

        credits_value = pd.to_numeric(
            df.get("annual_credits_value", 0), errors="coerce"
        ).fillna(0.0)

        df["effective_annual_fee"] = annual_fee - credits_value

        df["effective_fee_year1"] = np.where(
            is_waived,
            -credits_value,  # negative means profit from credits
            df["effective_annual_fee"],
        )

        df["net_annual_cost"] = df["effective_annual_fee"]

        logger.info(
            f"Calculated effective fees (avg: ${df['effective_annual_fee'].mean():.2f})"
        )
        return df

    def calculate_net_value(
        self, df: pd.DataFrame, annual_spending: float = 25000
    ) -> pd.DataFrame:
        df = df.copy()

        # Ensure required columns exist
        for col, default in [
            ("base_reward_rate", 0.0),
            ("effective_annual_fee", 0.0),
            ("effective_fee_year1", 0.0),
            ("welcome_bonus_value_usd", 0.0),
        ]:
            if col not in df.columns:
                df[col] = default

        df["expected_annual_rewards"] = annual_spending * (
            df["base_reward_rate"] / 100.0
        )

        df["net_value_annual"] = (
            df["expected_annual_rewards"] - df["effective_annual_fee"]
        )

        df["net_value_year1"] = (
            df["expected_annual_rewards"]
            - df["effective_fee_year1"]
            + df["welcome_bonus_value_usd"]
        )

        df["value_per_dollar"] = df["net_value_annual"] / float(annual_spending)

        logger.info(
            f"Calculated net values (avg annual: ${df['net_value_annual'].mean():.2f})"
        )
        return df

    # ----------------------------
    # Flags + categorical encodings
    # ----------------------------

    def filter_active_cards(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "discontinued" in df.columns:
            disc = df["discontinued"].fillna(False).astype(bool)
            df["is_active"] = (~disc).astype(int)
            df["is_discontinued"] = disc.astype(int)
            logger.info(f"Found {int(disc.sum())} discontinued cards out of {len(df)}")
        else:
            df["is_active"] = 1
            df["is_discontinued"] = 0

        return df

    def create_issuer_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "issuer" in df.columns:
            df["issuer_clean"] = (
                df["issuer"].astype(str).str.replace("_", " ").str.title()
            )
            issuer_dummies = pd.get_dummies(
                df["issuer"], prefix="issuer", prefix_sep="_"
            )
            df = pd.concat([df, issuer_dummies], axis=1)
            logger.info(f"Encoded {len(issuer_dummies.columns)} issuers")

        if "network" in df.columns:
            network_dummies = pd.get_dummies(
                df["network"], prefix="network", prefix_sep="_"
            )
            df = pd.concat([df, network_dummies], axis=1)
            logger.info(f"Encoded {len(network_dummies.columns)} networks")

        if "annual_fee" in df.columns:
            fee = pd.to_numeric(df["annual_fee"], errors="coerce").fillna(0.0)
            df["is_premium"] = (fee >= 450).astype(int)
            df["is_mid_tier"] = ((fee >= 95) & (fee < 450)).astype(int)
            df["is_no_annual_fee"] = (fee == 0).astype(int)

        if "is_business" in df.columns:
            df["is_business"] = df["is_business"].fillna(False).astype(int)

        if "currency" in df.columns:
            currency_dummies = pd.get_dummies(
                df["currency"], prefix="currency", prefix_sep="_"
            )
            df = pd.concat([df, currency_dummies], axis=1)

        return df

    # ----------------------------
    # Full pipeline
    # ----------------------------

    def engineer_features(
        self, df: pd.DataFrame, annual_spending: float = 25000
    ) -> pd.DataFrame:
        if df is None or df.empty:
            logger.info("No credit card rows to engineer.")
            return df

        logger.info(f"Engineering features for {len(df)} credit cards")

        df = self.filter_active_cards(df)
        df = self.extract_base_reward_rate(df)
        df = self.parse_welcome_bonus_offers(df)
        df = self.calculate_welcome_bonus_value(df)
        df = self.parse_credits_benefits(df)
        df = self.calculate_effective_annual_fee(df)
        df = self.calculate_net_value(df, annual_spending)
        df = self.create_issuer_network_features(df)

        logger.info(f"Credit card feature engineering complete: {df.shape}")
        return df


# =============================================================================
# Transactions
# =============================================================================


class TransactionFeatureEngineer:
    """
    Feature engineering for transaction data.

    Schema: transaction_id, user_id, date, category, merchant, mcc_code, amount, card_used
    """

    def __init__(self):
        self.standard_categories = [
            "dining",
            "travel",
            "online_shopping",
            "utilities",
            "entertainment",
            "groceries",
            "gas",
        ]
        logger.info("Initialized TransactionFeatureEngineer")

    def aggregate_spending_by_category(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        category_col: str = "category",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        logger.info(f"Aggregating spending for {df[user_id_col].nunique()} users")

        agg_df = (
            df.groupby([user_id_col, category_col])[amount_col]
            .agg(
                [
                    ("total_spent", "sum"),
                    ("transaction_count", "count"),
                    ("avg_transaction", "mean"),
                    ("max_transaction", "max"),
                    ("min_transaction", "min"),
                    ("std_transaction", "std"),
                ]
            )
            .reset_index()
        )

        spending_pivot = agg_df.pivot(
            index=user_id_col, columns=category_col, values="total_spent"
        ).fillna(0)
        spending_pivot.columns = [
            f"{col}_total_spent" for col in spending_pivot.columns
        ]

        count_pivot = agg_df.pivot(
            index=user_id_col, columns=category_col, values="transaction_count"
        ).fillna(0)
        count_pivot.columns = [f"{col}_txn_count" for col in count_pivot.columns]

        result = spending_pivot.reset_index()
        result = result.merge(count_pivot.reset_index(), on=user_id_col, how="left")

        spending_cols = [c for c in result.columns if c.endswith("_total_spent")]
        result["total_spending"] = result[spending_cols].sum(axis=1)

        count_cols = [c for c in result.columns if c.endswith("_txn_count")]
        result["total_transactions"] = result[count_cols].sum(axis=1)

        def calculate_spending_entropy(row):
            vals = [row[c] for c in spending_cols if row[c] > 0]
            s = float(sum(vals))
            if s <= 0:
                return 0.0
            probs = np.array(vals, dtype=float) / s
            return float(-np.sum(probs * np.log2(probs + 1e-10)))

        result["spending_diversity"] = result.apply(calculate_spending_entropy, axis=1)

        logger.info(f"Created spending aggregations for {len(result)} users")
        return result

    def extract_temporal_patterns(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        amount_col: str = "amount",
        user_id_col: str = "user_id",
    ) -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        df["day_of_week"] = df[date_col].dt.dayofweek
        df["day_of_month"] = df[date_col].dt.day
        df["month"] = df[date_col].dt.month
        df["quarter"] = df[date_col].dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = (df["day_of_month"] <= 7).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 24).astype(int)

        temporal_features = (
            df.groupby(user_id_col)
            .agg(
                {
                    "is_weekend": "mean",
                    "month": lambda x: x.mode()[0] if len(x) > 0 else 1,
                    "day_of_week": lambda x: x.mode()[0] if len(x) > 0 else 0,
                    amount_col: ["mean", "std", "median", "sum"],
                }
            )
            .reset_index()
        )

        temporal_features.columns = [
            user_id_col,
            "weekend_spending_ratio",
            "peak_spending_month",
            "peak_spending_day",
            "avg_transaction_amount",
            "transaction_amount_std",
            "median_transaction_amount",
            "total_spending_temporal",
        ]

        logger.info(f"Extracted temporal patterns for {len(temporal_features)} users")
        return temporal_features

    def analyze_card_usage_patterns(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        card_col: str = "card_used",
        category_col: str = "category",
    ) -> pd.DataFrame:
        logger.info("Analyzing card usage patterns")

        cards_per_user = df.groupby(user_id_col)[card_col].nunique().reset_index()
        cards_per_user.columns = [user_id_col, "num_cards_used"]

        most_used_card = (
            df.groupby(user_id_col)[card_col]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else None)
            .reset_index()
        )
        most_used_card.columns = [user_id_col, "primary_card"]

        def calc_switch_rate(cards):
            if len(cards) <= 1:
                return 0.0
            switches = sum(
                1 for i in range(1, len(cards)) if cards.iloc[i] != cards.iloc[i - 1]
            )
            return float(switches) / float(len(cards) - 1)

        switch_rate = (
            df.groupby(user_id_col)[card_col].apply(calc_switch_rate).reset_index()
        )
        switch_rate.columns = [user_id_col, "card_switch_rate"]

        card_features = cards_per_user.merge(most_used_card, on=user_id_col)
        card_features = card_features.merge(switch_rate, on=user_id_col)

        logger.info(f"Created card usage features for {len(card_features)} users")
        return card_features

    def analyze_mcc_patterns(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        mcc_col: str = "mcc_code",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        logger.info("Analyzing MCC patterns")

        mcc_diversity = df.groupby(user_id_col)[mcc_col].nunique().reset_index()
        mcc_diversity.columns = [user_id_col, "num_unique_mccs"]

        most_common_mcc = (
            df.groupby(user_id_col)[mcc_col]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else None)
            .reset_index()
        )
        most_common_mcc.columns = [user_id_col, "primary_mcc"]

        avg_per_mcc = (
            df.groupby(user_id_col)
            .apply(
                lambda x: (
                    x.groupby(mcc_col)[amount_col].mean().mean() if len(x) > 0 else 0.0
                )
            )
            .reset_index()
        )
        avg_per_mcc.columns = [user_id_col, "avg_spending_per_mcc"]

        mcc_features = mcc_diversity.merge(most_common_mcc, on=user_id_col)
        mcc_features = mcc_features.merge(avg_per_mcc, on=user_id_col)

        logger.info(f"Created MCC features for {len(mcc_features)} users")
        return mcc_features

    def create_merchant_features(
        self,
        df: pd.DataFrame,
        merchant_col: str = "merchant",
        user_id_col: str = "user_id",
    ) -> pd.DataFrame:
        logger.info("Creating merchant features")

        merchant_diversity = (
            df.groupby(user_id_col)[merchant_col].nunique().reset_index()
        )
        merchant_diversity.columns = [user_id_col, "num_unique_merchants"]

        favorite_merchant = (
            df.groupby(user_id_col)[merchant_col]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else None)
            .reset_index()
        )
        favorite_merchant.columns = [user_id_col, "favorite_merchant"]

        def calc_repeat_ratio(merchants):
            total = len(merchants)
            unique = merchants.nunique()
            return float(total - unique) / float(total) if total > 0 else 0.0

        repeat_ratio = (
            df.groupby(user_id_col)[merchant_col].apply(calc_repeat_ratio).reset_index()
        )
        repeat_ratio.columns = [user_id_col, "repeat_merchant_ratio"]

        merchant_features = merchant_diversity.merge(favorite_merchant, on=user_id_col)
        merchant_features = merchant_features.merge(repeat_ratio, on=user_id_col)

        logger.info(f"Created merchant features for {len(merchant_features)} users")
        return merchant_features

    def handle_suspicious_transactions(
        self, df: pd.DataFrame, user_id_col: str = "user_id"
    ) -> pd.DataFrame:
        if "suspicious" not in df.columns:
            return pd.DataFrame()

        logger.info("Handling suspicious transactions")
        suspicious_features = (
            df.groupby(user_id_col)["suspicious"]
            .agg([("num_suspicious", "sum"), ("suspicious_rate", "mean")])
            .reset_index()
        )
        return suspicious_features

    def engineer_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        date_col: str = "date",
        amount_col: str = "amount",
        merchant_col: str = "merchant",
        category_col: str = "category",
        card_col: str = "card_used",
        mcc_col: str = "mcc_code",
    ) -> pd.DataFrame:
        logger.info(f"Engineering features for {len(df)} transactions")

        spending_features = self.aggregate_spending_by_category(
            df, user_id_col, category_col, amount_col
        )
        temporal_features = self.extract_temporal_patterns(
            df, date_col, amount_col, user_id_col
        )
        card_features = self.analyze_card_usage_patterns(
            df, user_id_col, card_col, category_col
        )
        mcc_features = self.analyze_mcc_patterns(df, user_id_col, mcc_col, amount_col)
        merchant_features = self.create_merchant_features(df, merchant_col, user_id_col)
        suspicious_features = self.handle_suspicious_transactions(df, user_id_col)

        features_df = spending_features
        for feat_df in [
            temporal_features,
            card_features,
            mcc_features,
            merchant_features,
        ]:
            features_df = features_df.merge(feat_df, on=user_id_col, how="outer")

        if not suspicious_features.empty:
            features_df = features_df.merge(
                suspicious_features, on=user_id_col, how="left"
            )

        logger.info(f"Transaction feature engineering complete: {features_df.shape}")
        return features_df


# =============================================================================
# Users
# =============================================================================


class UserProfileFeatureEngineer:
    """
    Feature engineering for user profile data.

    Schema: user_id, archetype, monthly_budget, cards, redemption_preference, age_group, location_type
    """

    def __init__(self):
        logger.info("Initialized UserProfileFeatureEngineer")

    def parse_cards_column(
        self, df: pd.DataFrame, cards_col: str = "cards"
    ) -> pd.DataFrame:
        df = df.copy()

        def safe_parse_cards(cards_str):
            if pd.isna(cards_str):
                return []
            try:
                return ast.literal_eval(str(cards_str))
            except (ValueError, SyntaxError):
                return [
                    c.strip()
                    for c in str(cards_str).strip("[]'\"").split(",")
                    if c.strip()
                ]

        df["cards_list"] = df[cards_col].apply(safe_parse_cards)
        df["num_cards"] = df["cards_list"].apply(len)

        logger.info(
            f"Parsed cards for {len(df)} users (avg: {df['num_cards'].mean():.2f})"
        )
        return df

    def encode_archetype(
        self, df: pd.DataFrame, archetype_col: str = "archetype"
    ) -> pd.DataFrame:
        df = df.copy()
        archetype_dummies = pd.get_dummies(
            df[archetype_col], prefix="archetype", prefix_sep="_"
        )
        df = pd.concat([df, archetype_dummies], axis=1)
        logger.info(f"Encoded {len(archetype_dummies.columns)} archetypes")
        return df

    def encode_age_group(
        self, df: pd.DataFrame, age_col: str = "age_group"
    ) -> pd.DataFrame:
        df = df.copy()
        age_order = {"18-25": 1, "26-35": 2, "36-50": 3, "51-65": 4, "65+": 5}
        df["age_group_ordinal"] = df[age_col].map(age_order).fillna(0)

        age_dummies = pd.get_dummies(df[age_col], prefix="age", prefix_sep="_")
        df = pd.concat([df, age_dummies], axis=1)

        logger.info(f"Encoded {len(age_dummies.columns)} age groups")
        return df

    def encode_location_type(
        self, df: pd.DataFrame, location_col: str = "location_type"
    ) -> pd.DataFrame:
        df = df.copy()
        location_dummies = pd.get_dummies(
            df[location_col], prefix="location", prefix_sep="_"
        )
        df = pd.concat([df, location_dummies], axis=1)
        logger.info(f"Encoded {len(location_dummies.columns)} location types")
        return df

    def create_budget_features(
        self, df: pd.DataFrame, budget_col: str = "monthly_budget"
    ) -> pd.DataFrame:
        df = df.copy()
        df[budget_col] = pd.to_numeric(df[budget_col], errors="coerce").fillna(0.0)

        df["monthly_budget_log"] = np.log1p(df[budget_col])
        df["annual_budget"] = df[budget_col] * 12.0

        df["budget_quartile"] = pd.qcut(
            df[budget_col],
            q=4,
            labels=["Q1_low", "Q2_medium_low", "Q3_medium_high", "Q4_high"],
            duplicates="drop",
        )

        quartile_dummies = pd.get_dummies(
            df["budget_quartile"], prefix="budget", prefix_sep="_"
        )
        df = pd.concat([df, quartile_dummies], axis=1)

        logger.info(
            f"Created budget features (range: ${df[budget_col].min():.2f} - ${df[budget_col].max():.2f})"
        )
        return df

    def estimate_point_valuations(
        self, df: pd.DataFrame, redemption_col: str = "redemption_preference"
    ) -> pd.DataFrame:
        df = df.copy()

        valuation_map = {
            "cash_back": 0.01,
            "statement_credit": 0.01,
            "travel_portal": 0.015,
            "travel_transfer": 0.02,
            "gift_cards": 0.009,
            "merchandise": 0.008,
        }

        df["estimated_point_value"] = df[redemption_col].map(valuation_map).fillna(0.01)

        logger.info(
            f"Estimated point valuations (avg: ${df['estimated_point_value'].mean():.4f})"
        )
        return df

    def encode_redemption_preferences(
        self, df: pd.DataFrame, redemption_col: str = "redemption_preference"
    ) -> pd.DataFrame:
        df = df.copy()
        redemption_dummies = pd.get_dummies(
            df[redemption_col], prefix="redemption", prefix_sep="_"
        )
        df = pd.concat([df, redemption_dummies], axis=1)
        logger.info(f"Encoded {len(redemption_dummies.columns)} redemption preferences")
        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        archetype_col: str = "archetype",
        budget_col: str = "monthly_budget",
        cards_col: str = "cards",
        redemption_col: str = "redemption_preference",
        age_col: str = "age_group",
        location_col: str = "location_type",
    ) -> pd.DataFrame:
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


# =============================================================================
# Convenience function
# =============================================================================


def engineer_all_features(
    credit_cards_df: Optional[pd.DataFrame] = None,
    transactions_df: Optional[pd.DataFrame] = None,
    users_df: Optional[pd.DataFrame] = None,
    annual_spending: float = 25000,
    output_dir: Optional[Path] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    logger.info("=" * 70)
    logger.info("RewardSense Feature Engineering Pipeline")
    logger.info("Matches actual data structure from API and generators")
    logger.info("=" * 70)

    results = []

    # Credit cards
    if credit_cards_df is not None:
        logger.info("\n[1/3] Engineering credit card features...")
        card_engineer = CreditCardFeatureEngineer()
        cards_features = card_engineer.engineer_features(
            credit_cards_df, annual_spending
        )
        results.append(cards_features)
        if output_dir:
            output_path = Path(output_dir) / "credit_cards_features.csv"
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
            output_path = Path(output_dir) / "transactions_features.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            txn_features.to_csv(output_path, index=False)
            logger.info(f"✅ Saved to {output_path}")
    else:
        results.append(None)

    # Users
    if users_df is not None:
        logger.info("\n[3/3] Engineering user profile features...")
        user_engineer = UserProfileFeatureEngineer()
        user_features = user_engineer.engineer_features(users_df)
        results.append(user_features)
        if output_dir:
            output_path = Path(output_dir) / "users_features.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            user_features.to_csv(output_path, index=False)
            logger.info(f"✅ Saved to {output_path}")
    else:
        results.append(None)

    logger.info("\n" + "=" * 70)
    logger.info("Feature Engineering Pipeline Complete!")
    logger.info("=" * 70)

    return tuple(results)


# =============================================================================
# CLI/debug helper
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("RewardSense Feature Engineering Module")
    print("=" * 70)
    print("\n✅ Matches actual data structure:")
    print("  📄 Credit Cards: CreditCardBonuses API format")
    print("     - reward_rates.universal_base_rate (or object reward_rates)")
    print("     - offers array for welcome bonuses")
    print("     - credits array for benefits")
    print("     - discontinued flag")
    print("  📄 Transactions: transaction_id, user_id, date, category, ...")
    print("  📄 User Profiles: user_id, archetype, monthly_budget, ...")
    print("\nUsage:")
    print(
        "  from src.data_pipeline.preprocessing.feature_engineering import engineer_all_features"
    )
    print(
        "  cards_f, txns_f, users_f = engineer_all_features(cards_df, txns_df, users_df)"
    )
    print("=" * 70)
