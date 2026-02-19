"""
Unit tests for the synthetic data generators.

Covers Story 2.3 acceptance criteria:
  - 100 synthetic users with diverse profiles
  - Transactions span 12+ months of simulated history
  - Data generation reproducible with fixed seed

Also covers Story 4.2 test requirements:
  - Test synthetic data reproducibility
  - Test data distribution properties
  - Validate generated schema
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.generators.config import (
    DEFAULT_HISTORY_MONTHS,
    DEFAULT_NUM_USERS,
    DEFAULT_SEED,
    MIN_TRANSACTION_AMOUNT,
    REDEMPTION_PREFERENCES,
    SPENDING_ARCHETYPES,
    SPENDING_CATEGORIES,
)
from src.data_pipeline.generators.transaction_generator import TransactionGenerator
from src.data_pipeline.generators.user_profile_generator import UserProfileGenerator


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def user_gen():
    """Default user profile generator."""
    return UserProfileGenerator(num_users=DEFAULT_NUM_USERS, seed=DEFAULT_SEED)


@pytest.fixture
def small_user_gen():
    """Small generator for fast tests."""
    return UserProfileGenerator(num_users=10, seed=DEFAULT_SEED)


@pytest.fixture
def profiles(user_gen):
    """Pre-generated default profiles."""
    return user_gen.generate()


@pytest.fixture
def small_profiles(small_user_gen):
    """Pre-generated small profiles for transaction tests."""
    return small_user_gen.generate()


@pytest.fixture
def txn_gen():
    """Default transaction generator."""
    return TransactionGenerator(
        seed=DEFAULT_SEED,
        history_months=DEFAULT_HISTORY_MONTHS,
        start_date=datetime(2024, 1, 1),
    )


@pytest.fixture
def transactions(txn_gen, small_profiles):
    """Pre-generated transactions on the small user set."""
    return txn_gen.generate(small_profiles)


# =====================================================================
# UserProfileGenerator — Schema Validation
# =====================================================================


class TestUserProfileSchema:
    """Validate the schema of generated user profiles."""

    def test_dataframe_columns(self, profiles):
        expected = {
            "user_id",
            "archetype",
            "monthly_budget",
            "cards",
            "redemption_preference",
            "age_group",
            "location_type",
        }
        assert set(profiles.columns) == expected

    def test_user_id_format(self, profiles):
        for uid in profiles["user_id"]:
            assert uid.startswith("user_")
            # numeric suffix should be zero-padded 4 digits
            assert len(uid.split("_")[1]) == 4

    def test_user_id_uniqueness(self, profiles):
        assert profiles["user_id"].is_unique

    def test_archetype_values(self, profiles):
        valid = {a.name for a in SPENDING_ARCHETYPES}
        assert set(profiles["archetype"].unique()).issubset(valid)

    def test_monthly_budget_positive(self, profiles):
        assert (profiles["monthly_budget"] > 0).all()

    def test_cards_non_empty(self, profiles):
        for cards in profiles["cards"]:
            assert isinstance(cards, list)
            assert len(cards) >= 1

    def test_redemption_preference_valid(self, profiles):
        assert set(profiles["redemption_preference"].unique()).issubset(
            set(REDEMPTION_PREFERENCES)
        )

    def test_age_group_valid(self, profiles):
        valid_groups = {"18-25", "26-35", "36-50", "51-65", "65+"}
        assert set(profiles["age_group"].unique()).issubset(valid_groups)

    def test_location_type_valid(self, profiles):
        valid_locs = {"urban", "suburban", "rural"}
        assert set(profiles["location_type"].unique()).issubset(valid_locs)


# =====================================================================
# UserProfileGenerator — Count & Diversity
# =====================================================================


class TestUserProfileDiversity:
    """Verify the generator produces the required volume and diversity."""

    def test_generates_correct_count(self, profiles):
        assert len(profiles) == DEFAULT_NUM_USERS

    def test_custom_count(self):
        gen = UserProfileGenerator(num_users=50, seed=99)
        df = gen.generate()
        assert len(df) == 50

    def test_multiple_archetypes_present(self, profiles):
        # at least 3 distinct archetypes with 100 users
        assert profiles["archetype"].nunique() >= 3

    def test_archetype_distribution_reasonable(self, profiles):
        """No single archetype should dominate >60 % of users."""
        counts = profiles["archetype"].value_counts(normalize=True)
        assert counts.max() < 0.60

    def test_redemption_preference_diversity(self, profiles):
        assert profiles["redemption_preference"].nunique() >= 2

    def test_budget_range_matches_archetypes(self, profiles):
        """Each user's budget should fall within their archetype's range."""
        arch_map = {a.name: a for a in SPENDING_ARCHETYPES}
        for _, row in profiles.iterrows():
            lo, hi = arch_map[row["archetype"]].monthly_budget_range
            assert lo <= row["monthly_budget"] <= hi


# =====================================================================
# UserProfileGenerator — Reproducibility
# =====================================================================


class TestUserProfileReproducibility:
    """Ensure deterministic output for the same seed."""

    def test_same_seed_same_output(self):
        g1 = UserProfileGenerator(num_users=20, seed=42)
        g2 = UserProfileGenerator(num_users=20, seed=42)
        pd.testing.assert_frame_equal(g1.generate(), g2.generate())

    def test_different_seed_different_output(self):
        g1 = UserProfileGenerator(num_users=20, seed=42)
        g2 = UserProfileGenerator(num_users=20, seed=99)
        df1, df2 = g1.generate(), g2.generate()
        # user_ids are the same (sequential), but archetypes / budgets differ
        assert not df1["archetype"].equals(df2["archetype"])


# =====================================================================
# UserProfileGenerator — User-Card Mapping
# =====================================================================


class TestUserCardMapping:
    """Validate the exploded user ↔ card mapping table."""

    def test_mapping_columns(self, user_gen, profiles):
        mapping = user_gen.generate_user_cards_mapping(profiles)
        assert set(mapping.columns) == {"user_id", "card_id", "redemption_preference"}

    def test_mapping_covers_all_users(self, user_gen, profiles):
        mapping = user_gen.generate_user_cards_mapping(profiles)
        assert set(mapping["user_id"].unique()) == set(profiles["user_id"].unique())

    def test_mapping_row_count(self, user_gen, profiles):
        mapping = user_gen.generate_user_cards_mapping(profiles)
        expected = sum(len(cards) for cards in profiles["cards"])
        assert len(mapping) == expected


# =====================================================================
# TransactionGenerator — Schema Validation
# =====================================================================


class TestTransactionSchema:
    """Validate the schema of generated transactions."""

    def test_dataframe_columns(self, transactions):
        expected = {
            "transaction_id",
            "user_id",
            "date",
            "category",
            "merchant",
            "mcc_code",
            "amount",
            "card_used",
        }
        assert set(transactions.columns) == expected

    def test_transaction_id_uniqueness(self, transactions):
        assert transactions["transaction_id"].is_unique

    def test_transaction_id_format(self, transactions):
        for tid in transactions["transaction_id"]:
            assert tid.startswith("txn_")

    def test_amount_positive(self, transactions):
        assert (transactions["amount"] >= MIN_TRANSACTION_AMOUNT).all()

    def test_amount_is_float_two_decimals(self, transactions):
        for amt in transactions["amount"]:
            assert round(amt, 2) == amt

    def test_category_valid(self, transactions):
        valid = set(SPENDING_CATEGORIES.keys())
        assert set(transactions["category"].unique()).issubset(valid)

    def test_mcc_codes_are_int(self, transactions):
        assert transactions["mcc_code"].dtype in (np.int64, np.int32, int)

    def test_merchant_non_empty(self, transactions):
        assert (transactions["merchant"].str.len() > 0).all()

    def test_dates_are_datetime(self, transactions):
        assert pd.api.types.is_datetime64_any_dtype(transactions["date"]) or all(
            isinstance(d, datetime) for d in transactions["date"]
        )


# =====================================================================
# TransactionGenerator — Temporal Coverage
# =====================================================================


class TestTransactionTemporalCoverage:
    """Verify transactions span the required time window."""

    def test_spans_at_least_12_months(self, transactions):
        """Acceptance criteria: transactions span 12+ months."""
        min_date = transactions["date"].min()
        max_date = transactions["date"].max()
        delta = max_date - min_date
        assert delta.days >= 365

    def test_every_month_has_transactions(self, transactions):
        """No month in the window should be empty."""
        months = transactions["date"].apply(lambda d: (d.year, d.month)).unique()
        assert len(months) >= 12

    def test_date_within_expected_window(self, txn_gen, transactions):
        assert transactions["date"].min() >= txn_gen.start_date
        assert transactions["date"].max() <= txn_gen.end_date + timedelta(days=7)


# =====================================================================
# TransactionGenerator — Distribution Properties
# =====================================================================


class TestTransactionDistributions:
    """Verify statistical properties of generated data."""

    def test_all_users_have_transactions(self, small_profiles, transactions):
        """Every user should have at least 1 transaction."""
        txn_users = set(transactions["user_id"].unique())
        profile_users = set(small_profiles["user_id"].unique())
        assert profile_users.issubset(txn_users)

    def test_category_diversity(self, transactions):
        """Should produce transactions across multiple categories."""
        assert transactions["category"].nunique() >= 5

    def test_high_spenders_have_more_transactions(self, small_profiles, transactions):
        """Users with higher budgets should tend to have more transactions."""
        txn_counts = transactions.groupby("user_id").size().reset_index(name="n_txns")
        merged = small_profiles.merge(txn_counts, on="user_id")
        corr = merged["monthly_budget"].corr(merged["n_txns"])
        # positive correlation expected (doesn't have to be perfect)
        assert corr > 0.0

    def test_seasonal_effect_visible(self):
        """Holiday months should show higher online_shopping spend."""
        gen_profiles = UserProfileGenerator(num_users=30, seed=42).generate()
        gen = TransactionGenerator(
            seed=42,
            history_months=14,
            start_date=datetime(2024, 1, 1),
        )
        txns = gen.generate(gen_profiles)

        online = txns[txns["category"] == "online_shopping"]
        online_by_month = online.groupby(online["date"].apply(lambda d: d.month))[
            "amount"
        ].sum()

        # November + December should be higher than, say, January
        if 11 in online_by_month.index and 1 in online_by_month.index:
            assert online_by_month[11] > online_by_month[1]

    def test_card_used_from_user_portfolio(self, small_profiles, transactions):
        """card_used must be one of the user's assigned cards."""
        card_map = dict(zip(small_profiles["user_id"], small_profiles["cards"]))
        for _, txn in transactions.iterrows():
            assert txn["card_used"] in card_map[txn["user_id"]]


# =====================================================================
# TransactionGenerator — Reproducibility
# =====================================================================


class TestTransactionReproducibility:

    def test_same_seed_same_transactions(self, small_profiles):
        g1 = TransactionGenerator(seed=42, start_date=datetime(2024, 1, 1))
        g2 = TransactionGenerator(seed=42, start_date=datetime(2024, 1, 1))
        t1 = g1.generate(small_profiles)
        t2 = g2.generate(small_profiles)
        pd.testing.assert_frame_equal(t1, t2)

    def test_different_seed_different_transactions(self, small_profiles):
        g1 = TransactionGenerator(seed=42, start_date=datetime(2024, 1, 1))
        g2 = TransactionGenerator(seed=99, start_date=datetime(2024, 1, 1))
        t1 = g1.generate(small_profiles)
        t2 = g2.generate(small_profiles)
        assert not t1["amount"].equals(t2["amount"])


# =====================================================================
# Edge Cases
# =====================================================================


class TestEdgeCases:

    def test_single_user(self):
        gen = UserProfileGenerator(num_users=1, seed=42)
        df = gen.generate()
        assert len(df) == 1

        txn_gen = TransactionGenerator(seed=42, start_date=datetime(2024, 1, 1))
        txns = txn_gen.generate(df)
        assert len(txns) > 0

    def test_minimal_user_has_few_transactions(self):
        """Minimal-use archetype should generate fewer transactions."""
        # create a profile forced to minimal_user
        gen = UserProfileGenerator(num_users=50, seed=10)
        profiles = gen.generate()
        minimal = profiles[profiles["archetype"] == "minimal_user"]
        others = profiles[profiles["archetype"] != "minimal_user"]

        if len(minimal) == 0 or len(others) == 0:
            pytest.skip("Archetype sampling didn't produce both groups")

        txn_gen = TransactionGenerator(seed=10, start_date=datetime(2024, 1, 1))
        txns = txn_gen.generate(profiles)

        min_avg = (
            txns[txns["user_id"].isin(minimal["user_id"])]
            .groupby("user_id")
            .size()
            .mean()
        )
        other_avg = (
            txns[txns["user_id"].isin(others["user_id"])]
            .groupby("user_id")
            .size()
            .mean()
        )
        assert min_avg < other_avg

    def test_zero_weight_category_produces_no_transactions(self):
        """If archetype gives 0 weight to a category, no txns for it."""
        gen = UserProfileGenerator(num_users=20, seed=42)
        profiles = gen.generate()
        # minimal_user has travel weight = 0.0
        minimal = profiles[profiles["archetype"] == "minimal_user"]

        if len(minimal) == 0:
            pytest.skip("No minimal_user in sample")

        txn_gen = TransactionGenerator(seed=42, start_date=datetime(2024, 1, 1))
        txns = txn_gen.generate(minimal)
        travel_txns = txns[txns["category"] == "travel"]
        assert len(travel_txns) == 0

    def test_empty_profiles_returns_empty_transactions(self):
        empty = pd.DataFrame(
            columns=[
                "user_id",
                "archetype",
                "monthly_budget",
                "cards",
                "redemption_preference",
                "age_group",
                "location_type",
            ]
        )
        txn_gen = TransactionGenerator(seed=42, start_date=datetime(2024, 1, 1))
        txns = txn_gen.generate(empty)
        assert len(txns) == 0
