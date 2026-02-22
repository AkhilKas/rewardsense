"""
Tests for BiasDetector and fairness metrics.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.bias_detection.metrics import (
    FAIRLEARN_AVAILABLE,
    BiasConfig,
    BiasDetector,
    BiasMetric,
    BiasReport,
)


@pytest.fixture
def detector():
    return BiasDetector()


@pytest.fixture
def strict_detector():
    cfg = BiasConfig(
        representation_min_fraction=0.15,
        outcome_disparity_threshold=1.5,
        group_metric_diff_threshold=0.10,
    )
    return BiasDetector(config=cfg)


@pytest.fixture
def balanced_users():
    """Users with roughly equal group sizes."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "age_group": (
                ["18-25"] * 50 + ["26-35"] * 50 + ["36-50"] * 50 + ["51-65"] * 50
            ),
            "location_type": np.random.choice(["urban", "suburban", "rural"], n),
            "monthly_budget": np.random.normal(4000, 500, n),
            "estimated_point_value": np.random.uniform(0.01, 0.02, n),
        }
    )


@pytest.fixture
def imbalanced_users():
    """Users with significant group imbalance."""
    np.random.seed(42)
    n = 200
    age_groups = ["18-25"] * 150 + ["26-35"] * 30 + ["36-50"] * 15 + ["51-65"] * 5
    budgets = [3000.0] * 150 + [5000.0] * 30 + [8000.0] * 15 + [15000.0] * 5
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "age_group": age_groups,
            "monthly_budget": budgets,
            "estimated_point_value": [0.01] * 150
            + [0.015] * 30
            + [0.02] * 15
            + [0.025] * 5,
        }
    )


# =====================================================================
# Representation
# =====================================================================


class TestRepresentation:
    def test_balanced_no_underrepresentation(self, detector, balanced_users):
        metrics = detector.check_representation(balanced_users, "age_group")
        underrep = [m for m in metrics if m.name == "underrepresentation"]
        assert len(underrep) == 0

    def test_imbalanced_detects_underrepresentation(self, detector, imbalanced_users):
        metrics = detector.check_representation(imbalanced_users, "age_group")
        underrep = [m for m in metrics if m.name == "underrepresentation"]
        # 51-65 has 5/200 = 2.5% < 5% threshold
        assert len(underrep) >= 1
        assert underrep[0].is_biased is True

    def test_imbalance_ratio_detected(self, detector, imbalanced_users):
        metrics = detector.check_representation(imbalanced_users, "age_group")
        imb = [m for m in metrics if m.name == "representation_imbalance"]
        assert len(imb) == 1
        # 150/5 = 30.0 >> 2.0 threshold
        assert imb[0].is_biased is True
        assert imb[0].value > 10.0

    def test_missing_column(self, detector, balanced_users):
        metrics = detector.check_representation(balanced_users, "nonexistent")
        assert len(metrics) == 0

    def test_strict_config(self, strict_detector, balanced_users):
        # Even balanced data might fail at 15% threshold (25% per group)
        metrics = strict_detector.check_representation(balanced_users, "age_group")
        underrep = [m for m in metrics if m.name == "underrepresentation"]
        assert len(underrep) == 0  # 25% per group > 15% threshold


# =====================================================================
# Outcome disparity
# =====================================================================


class TestOutcomeDisparity:
    def test_no_disparity_balanced(self, detector, balanced_users):
        metrics = detector.check_outcome_disparity(
            balanced_users, "age_group", "monthly_budget"
        )
        assert len(metrics) == 1
        # Similar budgets across groups → low disparity
        assert metrics[0].is_biased is False

    def test_detects_disparity(self, detector, imbalanced_users):
        metrics = detector.check_outcome_disparity(
            imbalanced_users, "age_group", "monthly_budget"
        )
        assert len(metrics) == 1
        # 15000/3000 = 5.0 >> 2.0 threshold
        assert metrics[0].is_biased is True
        assert metrics[0].value > 2.0

    def test_details_contain_group_means(self, detector, imbalanced_users):
        metrics = detector.check_outcome_disparity(
            imbalanced_users, "age_group", "monthly_budget"
        )
        d = metrics[0].details
        assert "group_means" in d
        assert "best_group" in d
        assert "worst_group" in d

    def test_missing_columns(self, detector, balanced_users):
        metrics = detector.check_outcome_disparity(
            balanced_users, "nonexistent", "monthly_budget"
        )
        assert len(metrics) == 0


# =====================================================================
# Group metric difference
# =====================================================================


class TestGroupMetricDifference:
    def test_balanced_no_difference(self, detector, balanced_users):
        metrics = detector.check_group_metric_difference(
            balanced_users, "age_group", "monthly_budget"
        )
        assert len(metrics) == 1
        assert metrics[0].is_biased is False

    def test_detects_difference(self, detector, imbalanced_users):
        metrics = detector.check_group_metric_difference(
            imbalanced_users, "age_group", "monthly_budget"
        )
        assert len(metrics) == 1
        assert metrics[0].is_biased is True

    def test_custom_metric_fn(self, detector, balanced_users):
        metrics = detector.check_group_metric_difference(
            balanced_users,
            "age_group",
            "monthly_budget",
            metric_fn=lambda s: float(s.median()),
        )
        assert len(metrics) == 1


# =====================================================================
# Fairlearn: demographic parity
# =====================================================================


@pytest.mark.skipif(not FAIRLEARN_AVAILABLE, reason="Fairlearn not installed")
class TestDemographicParity:
    def test_fair_predictions(self, detector):
        np.random.seed(42)
        n = 200
        y_true = np.ones(n)
        y_pred = np.ones(n)  # same prediction for everyone → fair
        sf = np.array(["A"] * 100 + ["B"] * 100)
        metrics = detector.check_demographic_parity(y_true, y_pred, sf)
        assert len(metrics) == 1
        assert metrics[0].is_biased is False
        assert metrics[0].value < 0.01

    def test_biased_predictions(self, detector):
        np.random.seed(42)
        n = 200
        y_true = np.ones(n)
        # Group A always gets 1, group B always gets 0
        y_pred = np.array([1] * 100 + [0] * 100)
        sf = np.array(["A"] * 100 + ["B"] * 100)
        metrics = detector.check_demographic_parity(y_true, y_pred, sf)
        assert len(metrics) == 1
        assert metrics[0].is_biased is True
        assert metrics[0].value > 0.5

    def test_details_contain_group_rates(self, detector):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        sf = np.array(["A", "A", "B", "B"])
        metrics = detector.check_demographic_parity(y_true, y_pred, sf)
        assert "group_rates" in metrics[0].details


# =====================================================================
# Fairlearn: equalized odds
# =====================================================================


@pytest.mark.skipif(not FAIRLEARN_AVAILABLE, reason="Fairlearn not installed")
class TestEqualizedOdds:
    def test_fair_predictions(self, detector):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # perfect for both groups
        sf = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        metrics = detector.check_equalized_odds(y_true, y_pred, sf)
        assert len(metrics) == 1
        assert metrics[0].is_biased is False

    def test_biased_predictions(self, detector):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        # Group A: perfect; Group B: always predicts 1
        y_pred = np.array([1, 0, 1, 0, 1, 1, 1, 1])
        sf = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        metrics = detector.check_equalized_odds(y_true, y_pred, sf)
        assert len(metrics) == 1
        assert metrics[0].is_biased is True


# =====================================================================
# Full bias scan
# =====================================================================


class TestFullBiasScan:
    def test_scan_balanced(self, detector, balanced_users):
        report = detector.run_data_bias_scan(
            balanced_users,
            sensitive_features=["age_group"],
            outcome_columns=["monthly_budget"],
            dataset="users",
        )
        assert isinstance(report, BiasReport)
        assert report.dataset == "users"
        assert len(report.metrics) > 0
        assert report.metadata["row_count"] == 200

    def test_scan_imbalanced(self, detector, imbalanced_users):
        report = detector.run_data_bias_scan(
            imbalanced_users,
            sensitive_features=["age_group"],
            outcome_columns=["monthly_budget", "estimated_point_value"],
            dataset="users",
        )
        assert report.has_bias is True
        assert len(report.biased_metrics) > 0

    def test_scan_multiple_features(self, detector, balanced_users):
        report = detector.run_data_bias_scan(
            balanced_users,
            sensitive_features=["age_group", "location_type"],
            outcome_columns=["monthly_budget"],
            dataset="users",
        )
        features = {m.sensitive_feature for m in report.metrics}
        assert "age_group" in features
        assert "location_type" in features

    def test_report_to_json(self, tmp_path, detector, balanced_users):
        report = detector.run_data_bias_scan(
            balanced_users,
            sensitive_features=["age_group"],
            outcome_columns=["monthly_budget"],
            dataset="users",
        )
        out = tmp_path / "bias_report.json"
        report.to_json(out)

        loaded = json.loads(out.read_text())
        assert loaded["dataset"] == "users"
        assert "metrics" in loaded
        assert "summary" in loaded

    def test_report_summary(self, detector, imbalanced_users):
        report = detector.run_data_bias_scan(
            imbalanced_users,
            sensitive_features=["age_group"],
            outcome_columns=["monthly_budget"],
            dataset="users",
        )
        s = report.summary
        assert "total_metrics" in s
        assert "biased" in s
        assert s["total_metrics"] == s["biased"] + s["unbiased"]


# =====================================================================
# Config
# =====================================================================


class TestBiasConfig:
    def test_defaults(self):
        cfg = BiasConfig()
        assert cfg.demographic_parity_threshold == 0.10
        assert cfg.representation_min_fraction == 0.05

    def test_from_dict(self):
        cfg = BiasConfig.from_dict({"outcome_disparity_threshold": 3.0})
        assert cfg.outcome_disparity_threshold == 3.0

    def test_from_empty_dict(self):
        cfg = BiasConfig.from_dict({})
        assert cfg.demographic_parity_threshold == 0.10


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_empty_dataframe(self, detector):
        df = pd.DataFrame(columns=["age_group", "budget"])
        report = detector.run_data_bias_scan(
            df, sensitive_features=["age_group"], outcome_columns=["budget"]
        )
        assert len(report.metrics) == 0

    def test_single_group(self, detector):
        df = pd.DataFrame({"group": ["A"] * 50, "score": np.random.normal(100, 10, 50)})
        metrics = detector.check_outcome_disparity(df, "group", "score")
        assert len(metrics) == 0  # < 2 groups

    def test_metric_to_dict(self):
        m = BiasMetric(
            name="test",
            value=0.15,
            threshold=0.10,
            is_biased=True,
            sensitive_feature="age",
        )
        d = m.to_dict()
        assert d["name"] == "test"
        assert d["is_biased"] is True
        assert d["value"] == 0.15
