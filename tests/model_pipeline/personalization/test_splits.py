"""Tests for model_pipeline.personalization.splits."""

import numpy as np
import pandas as pd
import pytest

from model_pipeline.personalization.splits import (
    create_stratify_bins,
    split_data,
)


class TestCreateStratifyBins:
    def test_returns_correct_number_of_bins(self):
        y = pd.Series(np.random.uniform(0, 1, 100))
        bins = create_stratify_bins(y, n_bins=4)
        assert bins.nunique() <= 4

    def test_low_cardinality_returns_as_is(self):
        y = pd.Series([0.01, 0.01, 0.02, 0.02])
        bins = create_stratify_bins(y, n_bins=10)
        assert bins.dtype == object


class TestSplitData:
    def test_default_ratios(self, xy_pair):
        X, y = xy_pair
        result = split_data(X, y)
        total = len(result.X_train) + len(result.X_val) + len(result.X_test)
        assert total == len(X)

    def test_approximate_sizes(self, xy_pair):
        X, y = xy_pair
        n = len(X)
        result = split_data(X, y)

        assert abs(len(result.X_train) / n - 0.70) < 0.10
        assert abs(len(result.X_val) / n - 0.15) < 0.10
        assert abs(len(result.X_test) / n - 0.15) < 0.10

    def test_no_index_overlap(self, xy_pair):
        X, y = xy_pair
        result = split_data(X, y)

        train_idx = set(result.X_train.index)
        val_idx = set(result.X_val.index)
        test_idx = set(result.X_test.index)

        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_deterministic_with_same_seed(self, xy_pair):
        X, y = xy_pair
        r1 = split_data(X, y, random_seed=42)
        r2 = split_data(X, y, random_seed=42)
        assert list(r1.X_train.index) == list(r2.X_train.index)

    def test_invalid_ratios_raises(self, xy_pair):
        X, y = xy_pair
        with pytest.raises(ValueError, match="sum to"):
            split_data(X, y, train_size=0.5, val_size=0.1, test_size=0.1)

    def test_meta_is_split(self, joined_df, xy_pair):
        X, y = xy_pair
        meta = joined_df[["user_id"]].loc[X.index]
        result = split_data(X, y, meta=meta)
        assert result.meta_train is not None
        assert len(result.meta_train) == len(result.X_train)
