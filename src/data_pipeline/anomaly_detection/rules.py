"""
Domain-Specific Anomaly Rules

Custom anomaly rules for credit card, transaction, and user profile datasets.

These rules encode RewardSense business logic:
  - Reward rates within realistic ranges
  - Annual fees not exceeding known maximums
  - Transaction amounts within category norms
  - User budgets consistent with archetypes
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from src.data_pipeline.anomaly_detection.detectors import Anomaly, AnomalySeverity

logger = logging.getLogger(__name__)


class DomainRuleEngine:
    """Applies RewardSense-specific business rules to detect domain anomalies."""

    # ------------------------------------------------------------------
    # Credit card rules
    # ------------------------------------------------------------------

    def check_credit_card_rules(
        self,
        df: pd.DataFrame,
        dataset: str = "credit_cards",
    ) -> List[Anomaly]:
        """Domain rules for credit card data."""
        if df.empty:
            return []

        anomalies: List[Anomaly] = []

        # Rule 1: Reward rate sanity (no card gives > 10% base rate)
        if "base_reward_rate" in df.columns:
            unrealistic = df[df["base_reward_rate"] > 10.0]
            if len(unrealistic) > 0:
                anomalies.append(
                    Anomaly(
                        check_name="domain_reward_rate_too_high",
                        severity=AnomalySeverity.CRITICAL,
                        message=(
                            f"{len(unrealistic)} cards have base_reward_rate > 10%: "
                            f"likely a scraping/parsing error"
                        ),
                        dataset=dataset,
                        column="base_reward_rate",
                        details={
                            "count": len(unrealistic),
                            "max_value": round(
                                float(unrealistic["base_reward_rate"].max()), 2
                            ),
                            "card_names": (
                                unrealistic["card_name"].tolist()[:5]
                                if "card_name" in unrealistic.columns
                                else []
                            ),
                        },
                    )
                )

        # Rule 2: Annual fee sanity (no legitimate card > $5000)
        if "annual_fee" in df.columns:
            extreme_fee = df[df["annual_fee"] > 5000]
            if len(extreme_fee) > 0:
                anomalies.append(
                    Anomaly(
                        check_name="domain_annual_fee_extreme",
                        severity=AnomalySeverity.WARNING,
                        message=f"{len(extreme_fee)} cards have annual_fee > $5000",
                        dataset=dataset,
                        column="annual_fee",
                        details={
                            "count": len(extreme_fee),
                            "max_fee": round(float(extreme_fee["annual_fee"].max()), 2),
                        },
                    )
                )

        # Rule 3: Welcome bonus ROI sanity (ROI > 100% is suspicious)
        if "welcome_bonus_roi" in df.columns:
            suspicious_roi = df[df["welcome_bonus_roi"] > 1.0]
            if len(suspicious_roi) > 0:
                anomalies.append(
                    Anomaly(
                        check_name="domain_welcome_bonus_roi_extreme",
                        severity=AnomalySeverity.WARNING,
                        message=(
                            f"{len(suspicious_roi)} cards have welcome_bonus_roi > 100%"
                        ),
                        dataset=dataset,
                        column="welcome_bonus_roi",
                        details={
                            "count": len(suspicious_roi),
                            "max_roi": round(
                                float(suspicious_roi["welcome_bonus_roi"].max()), 4
                            ),
                        },
                    )
                )

        # Rule 4: Discontinued cards ratio (> 20% discontinued is unusual)
        if "discontinued" in df.columns:
            disc_ratio = float(df["discontinued"].sum()) / max(len(df), 1)
            if disc_ratio > 0.20:
                anomalies.append(
                    Anomaly(
                        check_name="domain_high_discontinued_ratio",
                        severity=AnomalySeverity.WARNING,
                        message=f"{disc_ratio:.1%} of cards are discontinued",
                        dataset=dataset,
                        column="discontinued",
                        details={"ratio": round(disc_ratio, 4)},
                    )
                )

        # Rule 5: Duplicate card names (after cleaning, should be 0)
        if "card_name" in df.columns:
            dup_count = int(df["card_name"].duplicated().sum())
            if dup_count > 0:
                anomalies.append(
                    Anomaly(
                        check_name="domain_duplicate_card_names",
                        severity=AnomalySeverity.WARNING,
                        message=f"{dup_count} duplicate card_name entries found",
                        dataset=dataset,
                        column="card_name",
                        details={"duplicate_count": dup_count},
                    )
                )

        logger.info(
            "[%s] Domain rules: %d anomalies from %d rules",
            dataset,
            len(anomalies),
            5,
        )
        return anomalies

    # ------------------------------------------------------------------
    # Transaction rules
    # ------------------------------------------------------------------

    def check_transaction_rules(
        self,
        df: pd.DataFrame,
        dataset: str = "transactions",
    ) -> List[Anomaly]:
        """Domain rules for transaction data."""
        if df.empty:
            return []

        anomalies: List[Anomaly] = []
        n = len(df)

        # Rule 1: Suspicious amount concentration
        if "amount" in df.columns:
            # > 1% of transactions over $10K is unusual
            high_val = df[df["amount"] > 10000]
            ratio = len(high_val) / max(n, 1)
            if ratio > 0.01:
                anomalies.append(
                    Anomaly(
                        check_name="domain_high_value_concentration",
                        severity=AnomalySeverity.WARNING,
                        message=f"{ratio:.2%} of transactions exceed $10,000",
                        dataset=dataset,
                        column="amount",
                        details={"count": len(high_val), "ratio": round(ratio, 4)},
                    )
                )

        # Rule 2: Category imbalance (one category > 50% of all transactions)
        if "category" in df.columns:
            cat_counts = df["category"].value_counts(normalize=True)
            dominant = cat_counts.iloc[0] if len(cat_counts) > 0 else 0
            if dominant > 0.50:
                anomalies.append(
                    Anomaly(
                        check_name="domain_category_imbalance",
                        severity=AnomalySeverity.WARNING,
                        message=(
                            f"Category '{cat_counts.index[0]}' accounts for "
                            f"{dominant:.1%} of transactions"
                        ),
                        dataset=dataset,
                        column="category",
                        details={
                            "dominant_category": cat_counts.index[0],
                            "ratio": round(float(dominant), 4),
                        },
                    )
                )

        # Rule 3: Users with suspiciously high transaction counts
        if "user_id" in df.columns:
            txn_per_user = df.groupby("user_id").size()
            mean_txns = float(txn_per_user.mean())
            std_txns = float(txn_per_user.std()) if len(txn_per_user) > 1 else 0
            if std_txns > 0:
                outlier_users = txn_per_user[txn_per_user > mean_txns + 3 * std_txns]
                if len(outlier_users) > 0:
                    anomalies.append(
                        Anomaly(
                            check_name="domain_user_transaction_outliers",
                            severity=AnomalySeverity.INFO,
                            message=(
                                f"{len(outlier_users)} users have transaction counts "
                                f"> 3 std above mean ({mean_txns:.0f} ± {std_txns:.0f})"
                            ),
                            dataset=dataset,
                            column="user_id",
                            details={
                                "outlier_user_count": len(outlier_users),
                                "mean_txns": round(mean_txns, 2),
                                "std_txns": round(std_txns, 2),
                            },
                        )
                    )

        # Rule 4: Future dates should not exist after cleaning
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            future = dates[dates > pd.Timestamp.now()]
            if len(future) > 0:
                anomalies.append(
                    Anomaly(
                        check_name="domain_future_dates_present",
                        severity=AnomalySeverity.CRITICAL,
                        message=f"{len(future)} transactions have future dates (post-cleaning)",
                        dataset=dataset,
                        column="date",
                        details={"count": len(future)},
                    )
                )

        # Rule 5: Duplicate transaction IDs
        if "transaction_id" in df.columns:
            dup_txns = int(df["transaction_id"].duplicated().sum())
            if dup_txns > 0:
                anomalies.append(
                    Anomaly(
                        check_name="domain_duplicate_transaction_ids",
                        severity=AnomalySeverity.CRITICAL,
                        message=f"{dup_txns} duplicate transaction_id values",
                        dataset=dataset,
                        column="transaction_id",
                        details={"duplicate_count": dup_txns},
                    )
                )

        logger.info(
            "[%s] Domain rules: %d anomalies from %d rules",
            dataset,
            len(anomalies),
            5,
        )
        return anomalies

    # ------------------------------------------------------------------
    # User profile rules
    # ------------------------------------------------------------------

    def check_user_rules(
        self,
        df: pd.DataFrame,
        dataset: str = "users",
    ) -> List[Anomaly]:
        """Domain rules for user profile data."""
        if df.empty:
            return []

        anomalies: List[Anomaly] = []

        # Rule 1: Budget-archetype consistency
        if "monthly_budget" in df.columns and "archetype" in df.columns:
            archetype_budget_ranges = {
                "minimal_user": (0, 2000),
                "budget_conscious": (500, 5000),
                "young_professional": (1500, 8000),
                "suburban_family": (3000, 15000),
                "frequent_traveler": (2500, 12000),
                "high_roller": (5000, 30000),
                "category_specialist": (1000, 10000),
            }
            inconsistent = 0
            for _, row in df.iterrows():
                arch = row.get("archetype", "")
                budget = float(row.get("monthly_budget", 0) or 0)
                expected_range = archetype_budget_ranges.get(arch)
                if expected_range and not (
                    expected_range[0] <= budget <= expected_range[1]
                ):
                    inconsistent += 1

            if inconsistent > 0:
                ratio = inconsistent / max(len(df), 1)
                severity = (
                    AnomalySeverity.CRITICAL
                    if ratio > 0.10
                    else AnomalySeverity.WARNING
                )
                anomalies.append(
                    Anomaly(
                        check_name="domain_budget_archetype_mismatch",
                        severity=severity,
                        message=(
                            f"{inconsistent} users ({ratio:.1%}) have budgets "
                            f"inconsistent with their archetype"
                        ),
                        dataset=dataset,
                        details={
                            "inconsistent_count": inconsistent,
                            "ratio": round(ratio, 4),
                        },
                    )
                )

        # Rule 2: Duplicate user IDs
        if "user_id" in df.columns:
            dup_users = int(df["user_id"].duplicated().sum())
            if dup_users > 0:
                anomalies.append(
                    Anomaly(
                        check_name="domain_duplicate_user_ids",
                        severity=AnomalySeverity.CRITICAL,
                        message=f"{dup_users} duplicate user_id values",
                        dataset=dataset,
                        column="user_id",
                        details={"duplicate_count": dup_users},
                    )
                )

        # Rule 3: Invalid archetype values
        if "archetype" in df.columns:
            valid_archetypes = {
                "young_professional",
                "suburban_family",
                "frequent_traveler",
                "budget_conscious",
                "high_roller",
                "minimal_user",
                "category_specialist",
            }
            invalid = df[~df["archetype"].isin(valid_archetypes)]
            if len(invalid) > 0:
                anomalies.append(
                    Anomaly(
                        check_name="domain_invalid_archetype",
                        severity=AnomalySeverity.WARNING,
                        message=f"{len(invalid)} users have unrecognized archetypes",
                        dataset=dataset,
                        column="archetype",
                        details={
                            "count": len(invalid),
                            "unknown_values": invalid["archetype"]
                            .unique()
                            .tolist()[:10],
                        },
                    )
                )

        logger.info(
            "[%s] Domain rules: %d anomalies from %d rules",
            dataset,
            len(anomalies),
            3,
        )
        return anomalies
