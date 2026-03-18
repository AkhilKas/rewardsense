"""Tests for model_pipeline.personalization.dataset_builder."""

import numpy as np
import pytest

from model_pipeline.personalization.dataset_builder import (
    DatasetBuildError,
    DatasetBuilder,
)
from model_pipeline.personalization.features import (
    TARGET_COLUMN,
)


class TestImputeMissing:
    def test_fills_numeric_with_median(self, joined_df):
        df = joined_df.copy()
        df.loc[0, "monthly_budget"] = np.nan
        df.loc[1, "monthly_budget"] = np.nan

        imputed, counts = DatasetBuilder.impute_missing(df)
        assert imputed["monthly_budget"].isna().sum() == 0
        assert counts["monthly_budget"] == 2

    def test_fills_onehot_with_zero(self, joined_df):
        df = joined_df.copy()
        onehot_col = [c for c in df.columns if c.startswith("archetype_")][0]
        df.loc[0, onehot_col] = np.nan

        imputed, counts = DatasetBuilder.impute_missing(df)
        assert imputed[onehot_col].isna().sum() == 0
        assert counts[onehot_col] == 1

    def test_no_missing_returns_empty_counts(self, joined_df):
        _, counts = DatasetBuilder.impute_missing(joined_df)
        assert len(counts) == 0


class TestBuildXY:
    def test_returns_correct_shapes(self, joined_df):
        X, y = DatasetBuilder.build_xy(joined_df)
        assert len(X) == len(y) == len(joined_df)
        assert TARGET_COLUMN not in X.columns

    def test_target_missing_raises(self, joined_df):
        df = joined_df.drop(columns=[TARGET_COLUMN])
        with pytest.raises(DatasetBuildError, match="Target column"):
            DatasetBuilder.build_xy(df)

    def test_x_has_only_feature_columns(self, joined_df):
        X, _ = DatasetBuilder.build_xy(joined_df)
        for col in X.columns:
            assert col != "user_id"
            assert col != TARGET_COLUMN


class TestLoadAndJoin:
    def test_join_produces_nonempty(
        self, users_features_df, transactions_features_df, mocker
    ):
        mock_loader = mocker.Mock()
        mock_loader.load_users_features.return_value = users_features_df
        mock_loader.load_transactions_features.return_value = transactions_features_df

        builder = DatasetBuilder(loader=mock_loader)
        merged = builder.load_and_join()
        assert len(merged) > 0
        assert "user_id" in merged.columns

    def test_empty_join_raises(
        self, users_features_df, transactions_features_df, mocker
    ):
        txn = transactions_features_df.copy()
        txn["user_id"] = "no_match"

        mock_loader = mocker.Mock()
        mock_loader.load_users_features.return_value = users_features_df
        mock_loader.load_transactions_features.return_value = txn

        builder = DatasetBuilder(loader=mock_loader)
        with pytest.raises(DatasetBuildError, match="zero rows"):
            builder.load_and_join()
