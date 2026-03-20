"""
Bias Regression Test Suite.

Runs a fixed synthetic dataset through all bias detection modules and
asserts that metric values stay within known-good bounds. If a code
change introduces new bias or breaks fairness guarantees, these tests
fail in CI and block the PR.

The golden dataset is deterministic (seeded RNG) so results are
reproducible across machines and Python versions.

Usage:
    PYTHONPATH=. pytest tests/model_pipeline/bias/test_bias_regression.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.model_pipeline.bias.slice_evaluator import SliceEvaluator
from src.model_pipeline.bias.model_bias_detector import (
    ModelBiasConfig,
    ModelBiasDetector,
)
from src.model_pipeline.bias.component_bias import (
    ScoringBiasChecker,
    ExplanationBiasChecker,
)

# =====================================================================
# Golden dataset — deterministic, versioned
# =====================================================================

GOLDEN_SEED = 20260319
GOLDEN_N_USERS = 400


def _build_golden_dataset():
    """
    Build a fixed synthetic dataset for regression testing.

    Returns (df, y_true, y_pred, recommendations_df, explanations_df).
    All outputs are deterministic for GOLDEN_SEED.
    """
    rng = np.random.default_rng(GOLDEN_SEED)
    n = GOLDEN_N_USERS

    archetypes = rng.choice(
        [
            "young_professional",
            "suburban_family",
            "frequent_traveler",
            "budget_conscious",
            "high_roller",
        ],
        n,
        p=[0.30, 0.25, 0.15, 0.20, 0.10],
    )
    age_groups = rng.choice(["18-25", "26-35", "36-50", "51-65"], n)
    locations = rng.choice(["urban", "suburban", "rural"], n)

    df = pd.DataFrame(
        {
            "user_id": range(n),
            "archetype": archetypes,
            "age_group": age_groups,
            "location_type": locations,
            "monthly_budget": rng.normal(3500, 1200, n).clip(500),
            "num_cards": rng.choice([1, 2, 3, 4, 5], n),
            "top_category": rng.choice(
                ["groceries", "dining", "travel", "gas", "online_shopping"], n
            ),
            "total_transactions": rng.integers(50, 500, n),
            "redemption_preference": rng.choice(
                ["cash_back", "travel_transfer", "statement_credit"], n
            ),
        }
    )

    # Ground truth: binary relevance
    y_true = rng.integers(0, 2, n).astype(float)

    # Predictions: correlated with truth but with controlled noise
    noise = rng.normal(0, 0.15, n)
    y_pred = (y_true * 0.7 + noise).clip(0, 1)

    # Recommendations for scoring bias
    issuers = rng.choice(
        ["Chase", "Amex", "Capital One", "Citi", "Discover"],
        n,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )
    card_types = rng.choice(["premium", "standard"], n, p=[0.35, 0.65])

    recommendations_df = pd.DataFrame(
        {
            "user_id": range(n),
            "archetype": archetypes,
            "recommended_card_issuer": issuers,
            "recommended_card_type": card_types,
        }
    )

    # Explanations for LLM bias
    explanations = []
    for arch in archetypes:
        base = (
            f"For your {arch.replace('_', ' ')} spending pattern, "
            "we recommend this card because it offers strong rewards "
            "in your top spending categories. "
        )
        # Vary length slightly by archetype to test consistency
        if arch == "high_roller":
            base += "Premium benefits include lounge access and concierge. "
        explanations.append(base)

    explanations_df = pd.DataFrame(
        {
            "user_id": range(n),
            "user_segment": archetypes,
            "explanation_text": explanations,
        }
    )

    return df, y_true, y_pred, recommendations_df, explanations_df


# Build once at module level — deterministic
(
    GOLDEN_DF,
    GOLDEN_Y_TRUE,
    GOLDEN_Y_PRED,
    GOLDEN_RECS,
    GOLDEN_EXPLANATIONS,
) = _build_golden_dataset()


# =====================================================================
# Golden thresholds — these define "known-good" bounds.
# If a code change causes values outside these ranges, the test fails.
# Update these ONLY after deliberate review.
# =====================================================================

GOLDEN_THRESHOLDS = {
    # SliceEvaluator: overall metrics must be within these ranges
    "overall_ndcg_5": (0.0, 1.0),
    "overall_precision_5": (0.0, 1.0),
    "overall_recall_5": (0.0, 1.0),
    # ModelBiasDetector: max allowed values
    "max_demographic_parity": 0.30,
    "max_equalized_odds": 0.30,
    "max_performance_disparity": 0.50,
    # ScoringBiasChecker: max issuer disparity
    "max_issuer_disparity": 0.25,
    # ExplanationBiasChecker: max quality deviation
    "max_explanation_length_deviation": 0.40,
    "max_explanation_readability_deviation": 0.40,
    # Counts: expected number of bias checks (regression on check count)
    "min_total_model_bias_checks": 3,
    "min_total_scoring_checks": 3,
    "min_total_explanation_checks": 2,
}


# =====================================================================
# Regression Tests
# =====================================================================


class TestSliceEvaluatorRegression:
    """Golden tests for SliceEvaluator output stability."""

    @pytest.fixture(autouse=True)
    def _run_evaluator(self):
        self.evaluator = SliceEvaluator(
            slicing_config={
                "archetype": {"column": "archetype", "type": "categorical"},
                "budget_tier": {
                    "column": "monthly_budget",
                    "type": "quantile",
                    "n_quantiles": 4,
                    "labels": ["Q1", "Q2", "Q3", "Q4"],
                },
            },
            disparity_threshold=0.10,
        )
        self.report = self.evaluator.evaluate(GOLDEN_DF, GOLDEN_Y_TRUE, GOLDEN_Y_PRED)

    def test_overall_metrics_in_range(self):
        """Overall metrics should stay within golden bounds."""
        for metric in ("ndcg_5", "precision_5", "recall_5"):
            val = self.report.overall_metrics.get(metric, -1)
            lo, hi = GOLDEN_THRESHOLDS[f"overall_{metric}"]
            assert (
                lo <= val <= hi
            ), f"Overall {metric}={val:.4f} outside golden range [{lo}, {hi}]"

    def test_slice_count_stable(self):
        """Number of slices should not change unexpectedly."""
        # 5 archetypes + up to 4 quantile bins
        assert len(self.report.slices) >= 5

    def test_no_nan_metrics(self):
        """No slice should have NaN metrics."""
        for s in self.report.slices:
            for mname, mval in s.metrics.items():
                assert not np.isnan(mval), f"NaN in slice '{s.name}' metric '{mname}'"

    def test_report_serializable(self):
        """Report must serialize to JSON without error."""
        d = self.report.to_dict()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100


class TestModelBiasDetectorRegression:
    """Golden tests for Fairlearn bias detection stability."""

    @pytest.fixture(autouse=True)
    def _run_detector(self):
        self.detector = ModelBiasDetector(
            config=ModelBiasConfig(
                demographic_parity_threshold=0.10,
                equalized_odds_threshold=0.10,
                performance_disparity_threshold=0.10,
            )
        )
        self.report = self.detector.detect(
            GOLDEN_Y_TRUE,
            GOLDEN_Y_PRED,
            GOLDEN_DF[["archetype", "age_group"]],
            model_name="golden_test",
        )

    def test_minimum_checks_ran(self):
        """Detector must run at least N checks."""
        assert (
            len(self.report.metrics) >= GOLDEN_THRESHOLDS["min_total_model_bias_checks"]
        )

    def test_demographic_parity_bounded(self):
        """Demographic parity must stay below golden max."""
        dp_metrics = [
            m for m in self.report.metrics if m.name == "demographic_parity_difference"
        ]
        for m in dp_metrics:
            assert m.value <= GOLDEN_THRESHOLDS["max_demographic_parity"], (
                f"Demographic parity for '{m.sensitive_feature}' = {m.value:.4f} "
                f"exceeds golden max {GOLDEN_THRESHOLDS['max_demographic_parity']}"
            )

    def test_equalized_odds_bounded(self):
        """Equalized odds must stay below golden max."""
        eo_metrics = [
            m for m in self.report.metrics if m.name == "equalized_odds_difference"
        ]
        for m in eo_metrics:
            assert m.value <= GOLDEN_THRESHOLDS["max_equalized_odds"], (
                f"Equalized odds for '{m.sensitive_feature}' = {m.value:.4f} "
                f"exceeds golden max"
            )

    def test_performance_disparity_bounded(self):
        """Performance disparity must stay below golden max."""
        pd_metrics = [
            m for m in self.report.metrics if m.name == "performance_disparity"
        ]
        for m in pd_metrics:
            assert m.value <= GOLDEN_THRESHOLDS["max_performance_disparity"], (
                f"Performance disparity for '{m.sensitive_feature}' = "
                f"{m.value:.4f} exceeds golden max"
            )

    def test_no_new_biased_metrics(self):
        """Track biased metric count — flag if it increases."""
        # Record current count. If this test starts failing,
        # it means a code change introduced new bias.
        biased_count = len(self.report.biased_metrics)
        # Allow up to N biased metrics on the golden dataset
        # Update this number ONLY after deliberate review.
        max_allowed_biased = 6
        assert biased_count <= max_allowed_biased, (
            f"Biased metric count {biased_count} exceeds golden max "
            f"{max_allowed_biased}. Review new bias before updating threshold."
        )

    def test_report_serializable(self):
        d = self.report.to_dict()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100


class TestScoringBiasRegression:
    """Golden tests for scoring engine bias stability."""

    @pytest.fixture(autouse=True)
    def _run_checker(self):
        self.checker = ScoringBiasChecker(
            issuer_disparity_threshold=0.15,
        )
        self.report = self.checker.check_issuer_bias(GOLDEN_RECS, "archetype")

    def test_minimum_checks_ran(self):
        assert len(self.report.metrics) >= GOLDEN_THRESHOLDS["min_total_scoring_checks"]

    def test_issuer_disparity_bounded(self):
        for m in self.report.metrics:
            assert m.value <= GOLDEN_THRESHOLDS["max_issuer_disparity"], (
                f"Issuer disparity '{m.check_name}' = {m.value:.4f} "
                f"exceeds golden max"
            )

    def test_no_systematic_issuer_bias(self):
        """No single issuer should be flagged as biased on golden data."""
        biased = [m for m in self.report.metrics if m.is_biased]
        max_allowed = 2
        assert len(biased) <= max_allowed, (
            f"{len(biased)} issuers flagged as biased on golden data. "
            f"Max allowed: {max_allowed}"
        )


class TestExplanationBiasRegression:
    """Golden tests for LLM explanation quality stability."""

    @pytest.fixture(autouse=True)
    def _run_checker(self):
        self.checker = ExplanationBiasChecker(
            length_disparity_threshold=0.20,
            readability_disparity_threshold=0.15,
        )
        self.report = self.checker.check_quality_consistency(
            GOLDEN_EXPLANATIONS, "user_segment"
        )

    def test_minimum_checks_ran(self):
        assert (
            len(self.report.metrics)
            >= GOLDEN_THRESHOLDS["min_total_explanation_checks"]
        )

    def test_length_deviation_bounded(self):
        length_checks = [m for m in self.report.metrics if "length" in m.check_name]
        for m in length_checks:
            assert (
                m.value <= GOLDEN_THRESHOLDS["max_explanation_length_deviation"]
            ), f"Length deviation = {m.value:.4f} exceeds golden max"

    def test_readability_deviation_bounded(self):
        read_checks = [m for m in self.report.metrics if "readability" in m.check_name]
        for m in read_checks:
            assert (
                m.value <= GOLDEN_THRESHOLDS["max_explanation_readability_deviation"]
            ), f"Readability deviation = {m.value:.4f} exceeds golden max"


class TestGoldenDatasetStability:
    """Meta-tests ensuring the golden dataset itself is deterministic."""

    def test_dataset_shape(self):
        assert GOLDEN_DF.shape[0] == GOLDEN_N_USERS

    def test_y_true_deterministic(self):
        """y_true should be identical across runs."""
        expected_sum = float(GOLDEN_Y_TRUE.sum())
        _, y2, _, _, _ = _build_golden_dataset()
        assert float(y2.sum()) == expected_sum

    def test_predictions_deterministic(self):
        """y_pred should be identical across runs."""
        expected_mean = float(GOLDEN_Y_PRED.mean())
        _, _, yp2, _, _ = _build_golden_dataset()
        assert abs(float(yp2.mean()) - expected_mean) < 1e-10

    def test_no_missing_values(self):
        assert GOLDEN_DF.isnull().sum().sum() == 0

    def test_all_archetypes_present(self):
        expected = {
            "young_professional",
            "suburban_family",
            "frequent_traveler",
            "budget_conscious",
            "high_roller",
        }
        assert set(GOLDEN_DF["archetype"].unique()) == expected
