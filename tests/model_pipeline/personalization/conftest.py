"""
Shared fixtures for personalization model tests.

Provides realistic-looking synthetic DataFrames that mimic
the Phase 1 feature-engineered outputs (users_features.csv and
transactions_features.csv).
"""

import numpy as np
import pandas as pd
import pytest

NUM_USERS = 60


def _make_user_id(i: int) -> str:
    return f"user_{i:04d}"


@pytest.fixture()
def users_features_df() -> pd.DataFrame:
    """Simulated users_features.csv from Phase 1."""
    rng = np.random.RandomState(42)
    n = NUM_USERS
    archetypes = ["high_roller", "budget_conscious", "travel_junkie", "everyday"]
    age_groups = ["18-25", "26-35", "36-50", "51-65", "65+"]
    locations = ["urban", "suburban", "rural"]
    redemptions = ["travel_portal", "cashback", "statement_credit", "gift_cards"]

    data = {
        "user_id": [_make_user_id(i) for i in range(n)],
        "archetype": rng.choice(archetypes, n),
        "monthly_budget": rng.uniform(500, 15000, n).round(2),
        "redemption_preference": rng.choice(redemptions, n),
        "age_group": rng.choice(age_groups, n),
        "location_type": rng.choice(locations, n),
        "num_cards": rng.randint(1, 8, n),
        "monthly_budget_log": None,
        "annual_budget": None,
        "budget_quartile": None,
        "age_group_ordinal": None,
        "estimated_point_value": rng.uniform(0.005, 0.03, n).round(4),
    }
    df = pd.DataFrame(data)
    df["monthly_budget_log"] = np.log1p(df["monthly_budget"]).round(4)
    df["annual_budget"] = (df["monthly_budget"] * 12).round(2)
    df["budget_quartile"] = pd.qcut(
        df["monthly_budget"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)

    age_ord = {"18-25": 1, "26-35": 2, "36-50": 3, "51-65": 4, "65+": 5}
    df["age_group_ordinal"] = df["age_group"].map(age_ord)

    for arch in archetypes:
        df[f"archetype_{arch}"] = (df["archetype"] == arch).astype(int)
    for ag in age_groups:
        df[f"age_{ag}"] = (df["age_group"] == ag).astype(int)
    for loc in locations:
        df[f"location_{loc}"] = (df["location_type"] == loc).astype(int)
    for red in redemptions:
        df[f"redemption_{red}"] = (df["redemption_preference"] == red).astype(int)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        df[f"budget_{q}"] = (df["budget_quartile"] == q).astype(int)

    return df


@pytest.fixture()
def transactions_features_df() -> pd.DataFrame:
    """Simulated transactions_features.csv from Phase 1."""
    rng = np.random.RandomState(42)
    n = NUM_USERS

    data = {
        "user_id": [_make_user_id(i) for i in range(n)],
        "total_spending": rng.uniform(500, 50000, n).round(2),
        "total_transactions": rng.randint(10, 500, n).astype(float),
        "avg_transaction_amount": rng.uniform(10, 200, n).round(2),
        "median_transaction_amount": rng.uniform(8, 150, n).round(2),
        "transaction_amount_std": rng.uniform(5, 100, n).round(2),
        "spending_diversity": rng.uniform(0.1, 3.0, n).round(4),
        "weekend_spending_ratio": rng.uniform(0.1, 0.5, n).round(4),
        "card_switch_rate": rng.uniform(0.0, 0.8, n).round(4),
        "num_cards_used": rng.randint(1, 6, n),
        "num_unique_mccs": rng.randint(3, 30, n),
        "num_unique_merchants": rng.randint(5, 50, n),
        "repeat_merchant_ratio": rng.uniform(0.1, 0.9, n).round(4),
    }
    return pd.DataFrame(data)


@pytest.fixture()
def joined_df(users_features_df, transactions_features_df) -> pd.DataFrame:
    """Merged user + transaction features (mimics DatasetBuilder.load_and_join)."""
    return users_features_df.merge(transactions_features_df, on="user_id", how="inner")


@pytest.fixture()
def xy_pair(joined_df):
    """(X, y) tuple from the joined fixture — all float64 for ML compatibility."""
    import numpy as np
    from model_pipeline.personalization.features import (
        get_feature_columns,
        TARGET_COLUMN,
    )

    feature_cols = get_feature_columns(joined_df)
    X = joined_df[feature_cols].copy().astype(np.float64)
    y = joined_df[TARGET_COLUMN].copy().astype(np.float64)
    return X, y
