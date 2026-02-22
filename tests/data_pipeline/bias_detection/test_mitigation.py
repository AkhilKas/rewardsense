"""
Tests for bias mitigation strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.bias_detection.mitigation import BiasMitigator, MitigationResult


@pytest.fixture
def mitigator():
    return BiasMitigator()


@pytest.fixture
def imbalanced_df():
    """DataFrame with significant group imbalance."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(200)],
            "age_group": ["18-25"] * 120
            + ["26-35"] * 50
            + ["51-65"] * 20
            + ["65+"] * 10,
            "monthly_budget": (
                list(np.random.normal(3000, 500, 120))
                + list(np.random.normal(5000, 800, 50))
                + list(np.random.normal(4000, 600, 20))
                + list(np.random.normal(2000, 300, 10))
            ),
            "score": (
                list(np.random.uniform(0.5, 0.8, 120))
                + list(np.random.uniform(0.6, 0.9, 50))
                + list(np.random.uniform(0.4, 0.7, 20))
                + list(np.random.uniform(0.3, 0.6, 10))
            ),
        }
    )


@pytest.fixture
def balanced_df():
    """DataFrame with roughly equal groups."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(200)],
            "age_group": ["18-25"] * 50
            + ["26-35"] * 50
            + ["36-50"] * 50
            + ["51-65"] * 50,
            "monthly_budget": np.random.normal(4000, 1000, 200),
            "score": np.random.uniform(0.5, 0.9, 200),
        }
    )


# =====================================================================
# Oversampling
# =====================================================================


class TestOversampling:
    def test_oversamples_minority(self, mitigator, imbalanced_df):
        result_df, result = mitigator.resample_oversample(imbalanced_df, "age_group")
        assert isinstance(result, MitigationResult)
        assert result.strategy == "oversample"
        assert result.rows_after >= result.rows_before
        # After: all groups should have at least as many as the majority
        counts = result_df["age_group"].value_counts()
        assert counts.min() >= 100  # close to majority (120)

    def test_reduces_imbalance(self, mitigator, imbalanced_df):
        _, result = mitigator.resample_oversample(imbalanced_df, "age_group")
        assert result.after_imbalance_ratio < result.before_imbalance_ratio

    def test_preserves_all_original_rows(self, mitigator, imbalanced_df):
        result_df, _ = mitigator.resample_oversample(imbalanced_df, "age_group")
        # All original user_ids should still be present
        original_ids = set(imbalanced_df["user_id"])
        assert original_ids.issubset(set(result_df["user_id"]))

    def test_reproducible(self, mitigator, imbalanced_df):
        df1, _ = mitigator.resample_oversample(
            imbalanced_df, "age_group", random_state=42
        )
        df2, _ = mitigator.resample_oversample(
            imbalanced_df, "age_group", random_state=42
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_trade_offs_documented(self, mitigator, imbalanced_df):
        _, result = mitigator.resample_oversample(imbalanced_df, "age_group")
        assert len(result.trade_offs) >= 1

    def test_result_to_dict(self, mitigator, imbalanced_df):
        _, result = mitigator.resample_oversample(imbalanced_df, "age_group")
        d = result.to_dict()
        assert "strategy" in d
        assert "before_distribution" in d
        assert "after_distribution" in d
        assert "improvement" in d
        assert "trade_offs" in d

    def test_custom_target_ratio(self, mitigator, imbalanced_df):
        _, result = mitigator.resample_oversample(
            imbalanced_df, "age_group", target_ratio=0.5
        )
        assert result.rows_after < result.rows_before + 200  # partial oversample


# =====================================================================
# Undersampling
# =====================================================================


class TestUndersampling:
    def test_undersamples_majority(self, mitigator, imbalanced_df):
        result_df, result = mitigator.resample_undersample(imbalanced_df, "age_group")
        assert result.strategy == "undersample"
        assert result.rows_after <= result.rows_before
        # After: all groups should have at most as many as the minority
        counts = result_df["age_group"].value_counts()
        assert counts.max() <= 15  # close to minority (10)

    def test_reduces_imbalance(self, mitigator, imbalanced_df):
        _, result = mitigator.resample_undersample(imbalanced_df, "age_group")
        assert result.after_imbalance_ratio <= result.before_imbalance_ratio

    def test_rows_decrease(self, mitigator, imbalanced_df):
        _, result = mitigator.resample_undersample(imbalanced_df, "age_group")
        assert result.rows_after < result.rows_before

    def test_reproducible(self, mitigator, imbalanced_df):
        df1, _ = mitigator.resample_undersample(
            imbalanced_df, "age_group", random_state=42
        )
        df2, _ = mitigator.resample_undersample(
            imbalanced_df, "age_group", random_state=42
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_trade_offs_documented(self, mitigator, imbalanced_df):
        _, result = mitigator.resample_undersample(imbalanced_df, "age_group")
        assert len(result.trade_offs) >= 1


# =====================================================================
# Sample weights
# =====================================================================


class TestSampleWeights:
    def test_produces_weights(self, mitigator, imbalanced_df):
        weights, result = mitigator.compute_sample_weights(imbalanced_df, "age_group")
        assert result.strategy == "sample_weights"
        assert len(weights) == len(imbalanced_df)
        assert weights.dtype == float

    def test_mean_weight_is_one(self, mitigator, imbalanced_df):
        weights, _ = mitigator.compute_sample_weights(imbalanced_df, "age_group")
        assert abs(weights.mean() - 1.0) < 0.01

    def test_minority_gets_higher_weight(self, mitigator, imbalanced_df):
        weights, _ = mitigator.compute_sample_weights(imbalanced_df, "age_group")
        # 65+ group (10 rows) should have higher avg weight than 18-25 (120 rows)
        mask_minority = imbalanced_df["age_group"] == "65+"
        mask_majority = imbalanced_df["age_group"] == "18-25"
        assert weights[mask_minority].mean() > weights[mask_majority].mean()

    def test_no_data_change(self, mitigator, imbalanced_df):
        _, result = mitigator.compute_sample_weights(imbalanced_df, "age_group")
        assert result.rows_before == result.rows_after

    def test_weight_map_in_metadata(self, mitigator, imbalanced_df):
        _, result = mitigator.compute_sample_weights(imbalanced_df, "age_group")
        assert "weight_map" in result.metadata
        assert "mean_weight" in result.metadata

    def test_trade_offs_documented(self, mitigator, imbalanced_df):
        _, result = mitigator.compute_sample_weights(imbalanced_df, "age_group")
        assert len(result.trade_offs) >= 1


# =====================================================================
# Threshold adjustment
# =====================================================================


class TestThresholdAdjustment:
    def test_produces_per_group_thresholds(self, mitigator, imbalanced_df):
        result = mitigator.recommend_threshold_adjustments(
            imbalanced_df, "age_group", "score"
        )
        assert result.strategy == "threshold_adjustment"
        assert "group_thresholds" in result.metadata
        assert len(result.metadata["group_thresholds"]) == 4

    def test_targets_equal_rates(self, mitigator, imbalanced_df):
        result = mitigator.recommend_threshold_adjustments(
            imbalanced_df, "age_group", "score", target_rate=0.5
        )
        # After distribution should have all groups at ~0.5
        for rate in result.after_distribution.values():
            assert abs(rate - 0.5) < 0.01

    def test_custom_target_rate(self, mitigator, imbalanced_df):
        result = mitigator.recommend_threshold_adjustments(
            imbalanced_df, "age_group", "score", target_rate=0.3
        )
        assert result.metadata["target_rate"] == 0.3

    def test_trade_offs_documented(self, mitigator, imbalanced_df):
        result = mitigator.recommend_threshold_adjustments(
            imbalanced_df, "age_group", "score"
        )
        assert len(result.trade_offs) >= 1


# =====================================================================
# Run all strategies
# =====================================================================


class TestRunAllStrategies:
    def test_runs_three_strategies_without_score(self, mitigator, imbalanced_df):
        results = mitigator.run_all_strategies(imbalanced_df, "age_group")
        assert "oversample" in results
        assert "undersample" in results
        assert "sample_weights" in results
        assert "threshold_adjustment" not in results  # no score_column

    def test_runs_four_strategies_with_score(self, mitigator, imbalanced_df):
        results = mitigator.run_all_strategies(
            imbalanced_df, "age_group", score_column="score"
        )
        assert len(results) == 4
        assert "threshold_adjustment" in results

    def test_all_results_have_trade_offs(self, mitigator, imbalanced_df):
        results = mitigator.run_all_strategies(
            imbalanced_df, "age_group", score_column="score"
        )
        for name, result in results.items():
            assert len(result.trade_offs) >= 1, f"{name} missing trade-offs"

    def test_all_results_have_before_after(self, mitigator, imbalanced_df):
        results = mitigator.run_all_strategies(imbalanced_df, "age_group")
        for name, result in results.items():
            assert len(result.before_distribution) > 0, f"{name} missing before_dist"
            assert len(result.after_distribution) > 0, f"{name} missing after_dist"


# =====================================================================
# MitigationResult
# =====================================================================


class TestMitigationResult:
    def test_improvement_ratio(self):
        r = MitigationResult(
            strategy="test",
            description="test",
            before_distribution={},
            after_distribution={},
            before_imbalance_ratio=10.0,
            after_imbalance_ratio=2.0,
            rows_before=100,
            rows_after=100,
        )
        assert r.improvement == 0.2  # 2.0 / 10.0

    def test_improvement_no_change(self):
        r = MitigationResult(
            strategy="test",
            description="test",
            before_distribution={},
            after_distribution={},
            before_imbalance_ratio=5.0,
            after_imbalance_ratio=5.0,
            rows_before=100,
            rows_after=100,
        )
        assert r.improvement == 1.0


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_single_group(self, mitigator):
        df = pd.DataFrame({"group": ["A"] * 100, "val": range(100)})
        result_df, result = mitigator.resample_oversample(df, "group")
        assert result.rows_after == 100  # nothing to oversample

    def test_two_equal_groups(self, mitigator):
        df = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 50, "val": range(100)})
        result_df, result = mitigator.resample_oversample(df, "group")
        assert result.after_imbalance_ratio <= 1.01

    def test_balanced_data_no_change(self, mitigator, balanced_df):
        _, result = mitigator.resample_oversample(balanced_df, "age_group")
        # Already balanced → minimal change
        assert abs(result.improvement - 1.0) < 0.5
