"""
Tests for DataSlicer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.bias_detection.slicer import DataSlicer, SliceReport


@pytest.fixture
def slicer():
    return DataSlicer()


@pytest.fixture
def user_features():
    """Mimics user feature-engineered output."""
    np.random.seed(42)
    n = 200
    archetypes = np.random.choice(
        ["young_professional", "suburban_family", "budget_conscious", "high_roller"],
        n,
        p=[0.35, 0.30, 0.25, 0.10],
    )
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "archetype": archetypes,
            "age_group": np.random.choice(["18-25", "26-35", "36-50", "51-65"], n),
            "location_type": np.random.choice(["urban", "suburban", "rural"], n),
            "monthly_budget": np.random.normal(4000, 2000, n).clip(500),
            "estimated_point_value": np.random.uniform(0.008, 0.02, n),
            "num_cards": np.random.randint(1, 6, n),
        }
    )


@pytest.fixture
def txn_features():
    """Mimics transaction feature output (user-level aggregated)."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "total_spending": np.random.normal(5000, 2000, n).clip(100),
            "avg_transaction_amount": np.random.normal(60, 25, n).clip(5),
            "spending_diversity": np.random.uniform(0, 1, n),
            "category": np.random.choice(["dining", "groceries", "travel", "gas"], n),
        }
    )


# =====================================================================
# Categorical slicing
# =====================================================================


class TestCategoricalSlicing:
    def test_slice_by_archetype(self, slicer, user_features):
        report = slicer.slice_by_column(user_features, "archetype")
        assert isinstance(report, SliceReport)
        assert report.num_slices == 4
        assert report.total_rows == 200
        assert sum(s.count for s in report.slices) == 200

    def test_slice_fractions_sum_to_one(self, slicer, user_features):
        report = slicer.slice_by_column(user_features, "archetype")
        total_frac = sum(s.fraction for s in report.slices)
        assert abs(total_frac - 1.0) < 0.01

    def test_slice_contains_metrics(self, slicer, user_features):
        report = slicer.slice_by_column(
            user_features, "archetype", metric_columns=["monthly_budget"]
        )
        for s in report.slices:
            assert "monthly_budget_mean" in s.metrics
            assert "monthly_budget_median" in s.metrics
            assert "monthly_budget_std" in s.metrics

    def test_imbalance_ratio_computed(self, slicer, user_features):
        report = slicer.slice_by_column(user_features, "archetype")
        assert report.imbalance_ratio >= 1.0

    def test_missing_column_returns_empty(self, slicer, user_features):
        report = slicer.slice_by_column(user_features, "nonexistent")
        assert report.num_slices == 0

    def test_slice_stats_to_dict(self, slicer, user_features):
        report = slicer.slice_by_column(user_features, "age_group")
        d = report.slices[0].to_dict()
        assert "slice_column" in d
        assert "count" in d
        assert "fraction" in d
        assert "metrics" in d

    def test_slice_report_to_dict(self, slicer, user_features):
        report = slicer.slice_by_column(user_features, "location_type")
        d = report.to_dict()
        assert d["column"] == "location_type"
        assert d["num_slices"] == 3
        assert len(d["slices"]) == 3


# =====================================================================
# Quantile slicing
# =====================================================================


class TestQuantileSlicing:
    def test_quartile_slicing(self, slicer, user_features):
        report = slicer.slice_by_quantiles(
            user_features, "monthly_budget", n_quantiles=4
        )
        assert report.num_slices == 4
        assert report.column == "monthly_budget_quantile"

    def test_custom_quantile_count(self, slicer, user_features):
        report = slicer.slice_by_quantiles(
            user_features, "monthly_budget", n_quantiles=5
        )
        assert report.num_slices == 5

    def test_custom_labels(self, slicer, user_features):
        labels = ["Low", "Medium", "High"]
        report = slicer.slice_by_quantiles(
            user_features, "monthly_budget", n_quantiles=3, labels=labels
        )
        values = {s.slice_value for s in report.slices}
        assert values == {"Low", "Medium", "High"}

    def test_missing_column(self, slicer, user_features):
        report = slicer.slice_by_quantiles(user_features, "nonexistent")
        assert report.num_slices == 0


# =====================================================================
# Custom bin slicing
# =====================================================================


class TestBinSlicing:
    def test_custom_bins(self, slicer, user_features):
        bins = [0, 2000, 4000, 6000, float("inf")]
        labels = ["low", "medium", "high", "very_high"]
        report = slicer.slice_by_bins(
            user_features, "monthly_budget", bins=bins, bin_labels=labels
        )
        assert report.num_slices >= 2
        assert report.column == "monthly_budget_binned"

    def test_missing_column(self, slicer, user_features):
        report = slicer.slice_by_bins(user_features, "nonexistent", bins=[0, 1])
        assert report.num_slices == 0


# =====================================================================
# Multi-dimension slicing
# =====================================================================


class TestMultiDimensionSlicing:
    def test_slice_all_dimensions(self, slicer, user_features):
        reports = slicer.slice_all_dimensions(
            user_features,
            categorical_columns=["archetype", "age_group", "location_type"],
            quantile_columns=["monthly_budget"],
            metric_columns=["estimated_point_value"],
        )
        assert "archetype" in reports
        assert "age_group" in reports
        assert "location_type" in reports
        assert "monthly_budget_quantile" in reports
        assert len(reports) == 4

    def test_empty_dimensions(self, slicer, user_features):
        reports = slicer.slice_all_dimensions(user_features)
        assert len(reports) == 0


# =====================================================================
# Edge cases
# =====================================================================


class TestSlicerEdgeCases:
    def test_empty_dataframe(self, slicer):
        df = pd.DataFrame({"cat": pd.Series(dtype=str), "val": pd.Series(dtype=float)})
        report = slicer.slice_by_column(df, "cat")
        assert report.num_slices == 0
        assert report.total_rows == 0

    def test_single_group(self, slicer):
        df = pd.DataFrame({"cat": ["A"] * 50, "val": range(50)})
        report = slicer.slice_by_column(df, "cat")
        assert report.num_slices == 1
        assert report.imbalance_ratio == 1.0

    def test_two_groups_equal(self, slicer):
        df = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 50, "val": range(100)})
        report = slicer.slice_by_column(df, "cat")
        assert report.num_slices == 2
        assert abs(report.imbalance_ratio - 1.0) < 0.01
