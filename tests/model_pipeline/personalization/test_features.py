"""Tests for model_pipeline.personalization.features."""

import pandas as pd

from src.model_pipeline.personalization.features import (
    ALL_NUMERIC_FEATURES,
    TARGET_COLUMN,
    TRANSACTION_NUMERIC_FEATURES,
    USER_NUMERIC_FEATURES,
    compute_derived_features,
    detect_onehot_columns,
    get_feature_columns,
    validate_feature_frame,
)


class TestConstants:
    def test_target_column_name(self):
        assert TARGET_COLUMN == "estimated_point_value"

    def test_all_numeric_is_union(self):
        assert (
            ALL_NUMERIC_FEATURES == USER_NUMERIC_FEATURES + TRANSACTION_NUMERIC_FEATURES
        )

    def test_no_duplicate_features(self):
        assert len(ALL_NUMERIC_FEATURES) == len(set(ALL_NUMERIC_FEATURES))


class TestDetectOnehotColumns:
    def test_detects_matching_prefixes(self):
        df = pd.DataFrame(
            {
                "archetype_travel": [1],
                "age_26-35": [0],
                "location_urban": [1],
                "redemption_cashback": [0],
                "budget_Q1": [1],
                "unrelated_col": [99],
            }
        )
        result = detect_onehot_columns(df)
        assert "archetype_travel" in result
        assert "age_26-35" in result
        assert "unrelated_col" not in result

    def test_empty_dataframe(self):
        df = pd.DataFrame({"col_a": [], "col_b": []})
        assert detect_onehot_columns(df) == []


class TestGetFeatureColumns:
    def test_returns_numeric_and_onehot(self, joined_df):
        cols = get_feature_columns(joined_df)
        assert len(cols) > 0
        assert "monthly_budget" in cols
        assert "total_spending" in cols
        assert any(c.startswith("archetype_") for c in cols)

    def test_excludes_target(self, joined_df):
        cols = get_feature_columns(joined_df)
        assert TARGET_COLUMN not in cols


class TestValidateFeatureFrame:
    def test_valid_frame_no_warnings(self, joined_df):
        warnings = validate_feature_frame(joined_df)
        assert len(warnings) == 0

    def test_missing_target_produces_warning(self, joined_df):
        df = joined_df.drop(columns=[TARGET_COLUMN])
        warnings = validate_feature_frame(df)
        assert any("Target column" in w for w in warnings)

    def test_missing_numeric_produces_warning(self, joined_df):
        df = joined_df.drop(columns=["monthly_budget"])
        warnings = validate_feature_frame(df)
        assert any("Missing numeric" in w for w in warnings)


class TestComputeDerivedFeatures:
    def test_adds_spending_velocity(self):
        df = pd.DataFrame(
            {
                "avg_transaction_amount": [100.0, 50.0],
                "transaction_amount_std": [20.0, 10.0],
            }
        )
        result = compute_derived_features(df)
        assert "spending_velocity" in result.columns
        assert result["spending_velocity"].iloc[0] == round(20.0 / 100.0, 4)

    def test_adds_category_affinity_score(self):
        df = pd.DataFrame({"spending_diversity": [1.5, 2.8]})
        result = compute_derived_features(df)
        assert "category_affinity_score" in result.columns
        assert 0 <= result["category_affinity_score"].iloc[0] <= 1

    def test_defaults_peak_columns(self):
        df = pd.DataFrame({"col_a": [1]})
        result = compute_derived_features(df)
        assert "peak_spending_day" in result.columns
        assert "peak_spending_month" in result.columns

    def test_does_not_overwrite_existing(self):
        df = pd.DataFrame(
            {
                "spending_velocity": [0.99],
                "category_affinity_score": [0.55],
                "peak_spending_day": [3],
                "peak_spending_month": [7],
            }
        )
        result = compute_derived_features(df)
        assert result["spending_velocity"].iloc[0] == 0.99
        assert result["peak_spending_day"].iloc[0] == 3
