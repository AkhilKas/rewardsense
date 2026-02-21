"""
Synthetic transaction generator for RewardSense.

Produces realistic credit card transactions for each user profile,
incorporating MCC-aligned categories, seasonal spending patterns,
day-of-week effects, and per-category amount distributions.

All output is fully reproducible when using a fixed seed.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline.generators.config import (
    DEFAULT_HISTORY_MONTHS,
    DEFAULT_SEASONAL_MULTIPLIER,
    DEFAULT_SEED,
    MIN_TRANSACTION_AMOUNT,
    SEASONAL_MULTIPLIERS,
    SPENDING_ARCHETYPES,
    SPENDING_CATEGORIES,
    TRANSACTION_AMOUNT_PARAMS,
)

logger = logging.getLogger(__name__)


class TransactionGenerator:
    """Generates synthetic transactions for a set of user profiles.

    For each user, the generator:
      1. Determines monthly spend per category using archetype weights.
      2. Applies seasonal multipliers for realistic temporal variation.
      3. Distributes monthly spend into individual transactions with
         realistic amounts, merchants, and dates.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    history_months : int
        Number of months of transaction history to generate.
    start_date : datetime, optional
        Start date of the transaction window. Defaults to
        ``history_months`` months before today.
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        history_months: int = DEFAULT_HISTORY_MONTHS,
        start_date: Optional[datetime] = None,
    ) -> None:
        self.seed = seed
        self.history_months = history_months
        self._rng = np.random.default_rng(seed)
        self._archetype_map = {a.name: a for a in SPENDING_ARCHETYPES}

        if start_date is None:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            self.start_date = self._subtract_months(today, history_months)
        else:
            self.start_date = start_date

        # End date is exactly history_months calendar months after start,
        # but never past yesterday to avoid generating future-dated transactions.
        calculated_end = self._add_months(self.start_date, history_months)
        yesterday = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=1)
        self.end_date = min(calculated_end, yesterday)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, profiles_df: pd.DataFrame) -> pd.DataFrame:
        """Generate transactions for every user in *profiles_df*.

        Parameters
        ----------
        profiles_df : pd.DataFrame
            Output of ``UserProfileGenerator.generate()``.

        Returns
        -------
        pd.DataFrame
            Columns: transaction_id, user_id, date, category, merchant,
            mcc_code, amount, card_used
        """
        logger.info(
            "Generating transactions for %d users over %d months (seed=%d)",
            len(profiles_df),
            self.history_months,
            self.seed,
        )

        all_txns: List[Dict] = []
        txn_counter = 0

        for _, user in profiles_df.iterrows():
            user_txns, txn_counter = self._generate_user_transactions(
                user, txn_counter
            )
            all_txns.extend(user_txns)

        df = pd.DataFrame(all_txns)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)

        logger.info(
            "Generated %d total transactions across %d users",
            len(df),
            profiles_df["user_id"].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Calendar helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _days_in_month(year: int, month: int) -> int:
        """Return the number of days in a given month."""
        import calendar

        return calendar.monthrange(year, month)[1]

    @staticmethod
    def _add_months(dt: datetime, months: int) -> datetime:
        """Add calendar months to a datetime, clamping to valid day."""
        import calendar

        month = dt.month - 1 + months
        year = dt.year + month // 12
        month = month % 12 + 1
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return dt.replace(year=year, month=month, day=day)

    @staticmethod
    def _subtract_months(dt: datetime, months: int) -> datetime:
        """Subtract calendar months from a datetime, clamping to valid day."""
        import calendar

        month = dt.month - 1 - months
        year = dt.year + month // 12
        month = month % 12 + 1
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return dt.replace(year=year, month=month, day=day)

    # ------------------------------------------------------------------
    # Per-user generation
    # ------------------------------------------------------------------

    def _generate_user_transactions(
        self, user: pd.Series, txn_counter: int
    ) -> Tuple[List[Dict], int]:
        """Generate all transactions for a single user across the full window."""
        archetype = self._archetype_map[user["archetype"]]
        cards = user["cards"]
        txns: List[Dict] = []

        # iterate month by month
        current = self.start_date.replace(day=1)
        while current < self.end_date:
            month = current.month
            year = current.year
            days_in_month = self._days_in_month(year, month)

            # Clamp transaction days to not exceed end_date
            if year == self.end_date.year and month == self.end_date.month:
                days_in_month = min(days_in_month, self.end_date.day)

            for category, base_weight in archetype.category_weights.items():
                if base_weight <= 0:
                    continue

                # seasonal adjustment
                seasonal = SEASONAL_MULTIPLIERS.get(category, {}).get(
                    month, DEFAULT_SEASONAL_MULTIPLIER
                )

                # monthly budget for this category
                cat_budget = user["monthly_budget"] * base_weight * seasonal

                # add per-user noise (±15 %) so users within same archetype differ
                noise = self._rng.uniform(0.85, 1.15)
                cat_budget *= noise

                # split into individual transactions
                cat_txns = self._split_into_transactions(
                    user_id=user["user_id"],
                    category=category,
                    monthly_amount=cat_budget,
                    year=year,
                    month=month,
                    days_in_month=days_in_month,
                    cards=cards,
                    txn_counter=txn_counter,
                )
                txn_counter += len(cat_txns)
                txns.extend(cat_txns)

            # advance to next month (always go to 1st of next month)
            current = self._add_months(current.replace(day=1), 1)

        return txns, txn_counter

    # ------------------------------------------------------------------
    # Transaction splitting
    # ------------------------------------------------------------------

    def _split_into_transactions(
        self,
        user_id: str,
        category: str,
        monthly_amount: float,
        year: int,
        month: int,
        days_in_month: int,
        cards: List[str],
        txn_counter: int,
    ) -> List[Dict]:
        """Split a category's monthly budget into individual transactions."""
        mean_amt, std_amt = TRANSACTION_AMOUNT_PARAMS.get(category, (50.0, 25.0))
        cat_info = SPENDING_CATEGORIES.get(category, SPENDING_CATEGORIES["other"])

        # estimate number of transactions
        if monthly_amount < MIN_TRANSACTION_AMOUNT:
            return []
        est_num = max(1, int(round(monthly_amount / mean_amt)))
        # add some variation in count
        est_num = max(1, int(self._rng.poisson(est_num)))

        txns: List[Dict] = []
        remaining = monthly_amount

        for j in range(est_num):
            if remaining < MIN_TRANSACTION_AMOUNT:
                break

            # sample amount
            amt = float(self._rng.normal(mean_amt, std_amt))
            amt = max(MIN_TRANSACTION_AMOUNT, min(amt, remaining))
            amt = round(amt, 2)
            remaining -= amt

            # sample date with day-of-week weighting
            day = self._sample_transaction_day(category, days_in_month)
            date = datetime(year, month, day)

            # sample merchant and mcc
            merchant = str(self._rng.choice(cat_info["merchants"]))
            mcc_code = int(self._rng.choice(cat_info["mcc_codes"]))

            # sample which card was used (uniform random from portfolio)
            card_used = str(self._rng.choice(cards))

            txn_counter += 1
            txns.append(
                {
                    "transaction_id": f"txn_{txn_counter:07d}",
                    "user_id": user_id,
                    "date": date,
                    "category": category,
                    "merchant": merchant,
                    "mcc_code": mcc_code,
                    "amount": amt,
                    "card_used": card_used,
                }
            )

        return txns

    def _sample_transaction_day(self, category: str, days_in_month: int) -> int:
        """Sample a day within the month with realistic day-of-week effects.

        Weekends get higher weight for discretionary categories,
        weekdays for essentials like utilities and insurance.
        """
        days = np.arange(1, days_in_month + 1)

        # default: uniform
        weights = np.ones(days_in_month)

        weekend_heavy = {"dining", "entertainment", "online_shopping", "travel"}
        weekday_heavy = {"utilities", "insurance"}
        # monthly bills tend to cluster around the 1st or 15th
        bill_like = {"utilities", "insurance", "streaming"}

        if category in bill_like:
            # cluster around day 1-5 and 15-20
            for d in days:
                if d <= 5 or 15 <= d <= 20:
                    weights[d - 1] = 3.0
        elif category in weekend_heavy:
            # approximate: days 6,7,13,14,20,21,27,28 as weekends
            for d in days:
                if d % 7 in (6, 0):
                    weights[d - 1] = 1.8
        elif category in weekday_heavy:
            for d in days:
                if d % 7 not in (6, 0):
                    weights[d - 1] = 1.3

        weights = weights / weights.sum()
        return int(self._rng.choice(days, p=weights))