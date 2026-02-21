"""
Custom domain-specific statistics for RewardSense data.

Provides metrics beyond standard profiling.
"""

import pandas as pd
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CreditCardStatistics:
    """Generate domain-specific statistics for credit card data."""

    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate credit card-specific statistics."""
        logger.info(f"Calculating statistics for {len(df)} credit cards...")

        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_cards": len(df),
            "unique_issuers": (
                int(df["issuer"].nunique()) if "issuer" in df.columns else 0
            ),
            "unique_networks": (
                int(df["network"].nunique()) if "network" in df.columns else 0
            ),
        }

        # Annual fee statistics
        if "annual_fee" in df.columns:
            fees = df["annual_fee"]
            stats["annual_fee"] = {
                "mean": float(fees.mean()),
                "median": float(fees.median()),
                "min": float(fees.min()),
                "max": float(fees.max()),
                "std": float(fees.std()),
                "no_fee_count": int((fees == 0).sum()),
                "no_fee_pct": float((fees == 0).sum() / len(fees) * 100),
                "premium_count": int((fees >= 450).sum()),
                "premium_pct": float((fees >= 450).sum() / len(fees) * 100),
            }

        # Reward rates
        if "base_reward_rate" in df.columns:
            rates = df["base_reward_rate"]
            stats["reward_rates"] = {
                "mean": float(rates.mean()),
                "median": float(rates.median()),
                "min": float(rates.min()),
                "max": float(rates.max()),
            }

        # Currency distribution
        if "currency" in df.columns:
            currency_dist = df["currency"].value_counts().to_dict()
            stats["currency_distribution"] = {
                str(k): int(v) for k, v in currency_dist.items()
            }

        # Business vs personal
        if "is_business" in df.columns:
            stats["business_cards"] = {
                "count": int(df["is_business"].sum()),
                "pct": float(df["is_business"].sum() / len(df) * 100),
            }

        # Discontinued cards
        if "discontinued" in df.columns:
            stats["discontinued_cards"] = {
                "count": int(df["discontinued"].sum()),
                "pct": float(df["discontinued"].sum() / len(df) * 100),
            }

        logger.info("Credit card statistics calculated")
        return stats


class TransactionStatistics:
    """Generate domain-specific statistics for transaction data."""

    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate transaction-specific statistics."""
        logger.info(f"Calculating statistics for {len(df)} transactions...")

        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_transactions": len(df),
            "unique_users": (
                int(df["user_id"].nunique()) if "user_id" in df.columns else 0
            ),
        }

        # Amount statistics
        if "amount" in df.columns:
            amounts = df["amount"]
            stats["amount"] = {
                "total": float(amounts.sum()),
                "mean": float(amounts.mean()),
                "median": float(amounts.median()),
                "min": float(amounts.min()),
                "max": float(amounts.max()),
                "std": float(amounts.std()),
                "q25": float(amounts.quantile(0.25)),
                "q75": float(amounts.quantile(0.75)),
            }

        # Category distribution
        if "category" in df.columns:
            category_dist = df["category"].value_counts().to_dict()
            stats["category_distribution"] = {
                str(k): int(v) for k, v in category_dist.items()
            }

            # Category spending
            if "amount" in df.columns:
                category_spending = df.groupby("category")["amount"].sum().to_dict()
                stats["category_spending"] = {
                    str(k): float(v) for k, v in category_spending.items()
                }

        # Merchant diversity
        if "merchant" in df.columns:
            stats["merchant_diversity"] = {
                "unique_merchants": int(df["merchant"].nunique()),
                "avg_merchants_per_user": (
                    float(df.groupby("user_id")["merchant"].nunique().mean())
                    if "user_id" in df.columns
                    else 0
                ),
            }

        # Temporal patterns
        if "date" in df.columns:
            df_temp = df.copy()
            df_temp["date"] = pd.to_datetime(df_temp["date"])

            stats["temporal"] = {
                "date_range_start": df_temp["date"].min().isoformat(),
                "date_range_end": df_temp["date"].max().isoformat(),
                "days_covered": int(
                    (df_temp["date"].max() - df_temp["date"].min()).days
                ),
            }

        # Suspicious transactions
        if "suspicious" in df.columns:
            stats["suspicious"] = {
                "count": int(df["suspicious"].sum()),
                "pct": float(df["suspicious"].sum() / len(df) * 100),
            }

        logger.info("Transaction statistics calculated")
        return stats


class UserProfileStatistics:
    """Generate domain-specific statistics for user profile data."""

    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate user profile-specific statistics."""
        logger.info(f"Calculating statistics for {len(df)} users...")

        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_users": len(df),
        }

        # Budget statistics
        if "monthly_budget" in df.columns:
            budgets = df["monthly_budget"]
            stats["monthly_budget"] = {
                "mean": float(budgets.mean()),
                "median": float(budgets.median()),
                "min": float(budgets.min()),
                "max": float(budgets.max()),
                "std": float(budgets.std()),
                "total_annual_budget": float(budgets.sum() * 12),
            }

        # Archetype distribution
        if "archetype" in df.columns:
            archetype_dist = df["archetype"].value_counts().to_dict()
            stats["archetype_distribution"] = {
                str(k): int(v) for k, v in archetype_dist.items()
            }

        # Age group distribution
        if "age_group" in df.columns:
            age_dist = df["age_group"].value_counts().to_dict()
            stats["age_distribution"] = {str(k): int(v) for k, v in age_dist.items()}

        # Location distribution
        if "location_type" in df.columns:
            loc_dist = df["location_type"].value_counts().to_dict()
            stats["location_distribution"] = {
                str(k): int(v) for k, v in loc_dist.items()
            }

        # Redemption preferences
        if "redemption_preference" in df.columns:
            redemption_dist = df["redemption_preference"].value_counts().to_dict()
            stats["redemption_distribution"] = {
                str(k): int(v) for k, v in redemption_dist.items()
            }

        # Cards per user
        if "num_cards" in df.columns:
            cards = df["num_cards"]
            stats["cards_per_user"] = {
                "mean": float(cards.mean()),
                "median": float(cards.median()),
                "min": int(cards.min()),
                "max": int(cards.max()),
            }

        logger.info("User profile statistics calculated")
        return stats
