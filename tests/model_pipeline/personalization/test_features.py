"""Tests for model_pipeline.personalization.features."""

import pandas as pd

from model_pipeline.personalization.features import (
    ALL_NUMERIC_FEATURES,
    TARGET_COLUMN,
    TRANSACTION_NUMERIC_FEATURES,
    USER_NUMERIC_FEATURES,
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
