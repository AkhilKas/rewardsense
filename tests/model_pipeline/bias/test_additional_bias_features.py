"""
Tests for additional bias modules:
  - BiasDriftMonitor
  - BiasReportExporter
  - CounterfactualAnalyzer
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.model_pipeline.bias.drift_monitor import (
    BiasDriftMonitor,
    BiasDriftReport,
    TrendPoint,
)
from src.model_pipeline.bias.report_export import BiasReportExporter
from src.model_pipeline.bias.counterfactual import (
    CounterfactualAnalyzer,
    CounterfactualResult,
)
from src.model_pipeline.bias.model_bias_detector import (
    ModelBiasDetector,
)
from src.model_pipeline.bias.component_bias import (
    ScoringBiasChecker,
)
from src.model_pipeline.bias.slice_evaluator import SliceEvaluator


# =====================================================================
# Shared fixtures
# =====================================================================


@pytest.fixture
def sample_model_report():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "archetype": rng.choice(["A", "B"], n),
        }
    )
    yt = rng.integers(0, 2, n).astype(float)
    yp = yt * 0.8 + rng.normal(0, 0.2, n)
    detector = ModelBiasDetector()
    return detector.detect(yt, yp, df["archetype"], model_name="test")


@pytest.fixture
def sample_scoring_report():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "arch": rng.choice(["A", "B"], n),
            "recommended_card_issuer": rng.choice(["Chase", "Amex", "Citi"], n),
        }
    )
    checker = ScoringBiasChecker()
    return checker.check_issuer_bias(df, "arch")


@pytest.fixture
def sample_slice_report():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({"group": rng.choice(["X", "Y"], n)})
    yt = rng.integers(0, 2, n).astype(float)
    yp = yt * 0.7 + rng.normal(0, 0.2, n)
    evaluator = SliceEvaluator(
        slicing_config={"group": {"column": "group", "type": "categorical"}},
    )
    return evaluator.evaluate(df, yt, yp)


# =====================================================================
# BiasDriftMonitor
# =====================================================================


class TestBiasDriftMonitor:
    def test_record_creates_file(self, tmp_path, sample_model_report):
        monitor = BiasDriftMonitor(history_dir=tmp_path / "history")
        path = monitor.record(sample_model_report, "1.0.0")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["model_version"] == "1.0.0"

    def test_list_versions(self, tmp_path, sample_model_report):
        monitor = BiasDriftMonitor(history_dir=tmp_path / "history")
        monitor.record(sample_model_report, "1.0.0")
        monitor.record(sample_model_report, "2.0.0")
        versions = monitor.list_versions()
        assert "1.0.0" in versions
        assert "2.0.0" in versions

    def test_compare_two_versions(self, tmp_path, sample_model_report):
        monitor = BiasDriftMonitor(history_dir=tmp_path / "history")
        monitor.record(sample_model_report, "1.0.0")
        monitor.record(sample_model_report, "2.0.0")
        drift = monitor.compare("1.0.0", "2.0.0")
        assert isinstance(drift, BiasDriftReport)
        assert drift.before_version == "1.0.0"
        assert drift.after_version == "2.0.0"
        assert len(drift.metrics) > 0

    def test_compare_identical_no_regression(self, tmp_path, sample_model_report):
        monitor = BiasDriftMonitor(history_dir=tmp_path / "history")
        monitor.record(sample_model_report, "1.0.0")
        monitor.record(sample_model_report, "1.0.1")
        drift = monitor.compare("1.0.0", "1.0.1")
        # Same report → no drift
        assert not drift.has_regression

    def test_compare_missing_version_raises(self, tmp_path, sample_model_report):
        monitor = BiasDriftMonitor(history_dir=tmp_path / "history")
        monitor.record(sample_model_report, "1.0.0")
        with pytest.raises(FileNotFoundError):
            monitor.compare("1.0.0", "99.0.0")

    def test_trend(self, tmp_path):
        monitor = BiasDriftMonitor(history_dir=tmp_path / "history")

        # Create reports with known metric values
        for ver, val in [("1.0.0", 0.08), ("2.0.0", 0.12), ("3.0.0", 0.06)]:
            report = MagicMock()
            report.to_dict.return_value = {
                "all_metrics": [
                    {
                        "name": "demographic_parity_difference",
                        "sensitive_feature": "arch",
                        "value": val,
                    },
                ],
            }
            monitor.record(report, ver)

        trend = monitor.trend("demographic_parity_difference", "arch")
        assert len(trend) == 3
        assert all(isinstance(p, TrendPoint) for p in trend)

    def test_drift_report_serializable(self, tmp_path, sample_model_report):
        monitor = BiasDriftMonitor(history_dir=tmp_path / "history")
        monitor.record(sample_model_report, "1.0.0")
        monitor.record(sample_model_report, "2.0.0")
        drift = monitor.compare("1.0.0", "2.0.0")
        d = drift.to_dict()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 50

    def test_drift_report_log_to_mlflow(self, tmp_path, sample_model_report):
        monitor = BiasDriftMonitor(history_dir=tmp_path / "history")
        monitor.record(sample_model_report, "1.0.0")
        monitor.record(sample_model_report, "2.0.0")
        drift = monitor.compare("1.0.0", "2.0.0")
        tracker = MagicMock()
        drift.log_to_mlflow(tracker)
        tracker.log_metrics.assert_called_once()
        tracker.log_dict.assert_called_once()


# =====================================================================
# BiasReportExporter
# =====================================================================


class TestBiasReportExporter:
    def test_export_model_report_html(self, tmp_path, sample_model_report):
        exporter = BiasReportExporter()
        path = exporter.export_html(
            sample_model_report,
            output_path=tmp_path / "report.html",
        )
        assert path.exists()
        content = path.read_text()
        assert "<html" in content
        assert "RewardSense" in content
        assert "Model Bias Detection" in content

    def test_export_scoring_report_html(self, tmp_path, sample_scoring_report):
        exporter = BiasReportExporter()
        path = exporter.export_html(
            sample_scoring_report,
            output_path=tmp_path / "scoring.html",
        )
        assert path.exists()
        content = path.read_text()
        assert "Scoring Engine" in content

    def test_export_slice_report_html(self, tmp_path, sample_slice_report):
        exporter = BiasReportExporter()
        path = exporter.export_html(
            sample_slice_report,
            output_path=tmp_path / "slices.html",
        )
        assert path.exists()
        content = path.read_text()
        assert "Per-Slice" in content

    def test_export_full_report(
        self, tmp_path, sample_model_report, sample_scoring_report, sample_slice_report
    ):
        exporter = BiasReportExporter()
        path = exporter.export_full_report(
            model_report=sample_model_report,
            scoring_report=sample_scoring_report,
            slice_report=sample_slice_report,
            output_path=tmp_path / "full.html",
        )
        assert path.exists()
        content = path.read_text()
        assert "Model Bias Detection" in content
        assert "Scoring Engine" in content
        assert "Per-Slice" in content

    def test_html_contains_charts(self, tmp_path, sample_model_report):
        exporter = BiasReportExporter()
        path = exporter.export_html(
            sample_model_report,
            output_path=tmp_path / "with_charts.html",
        )
        content = path.read_text()
        # Charts are embedded as base64 images
        assert "data:image/png;base64" in content

    def test_export_creates_parent_dirs(self, tmp_path, sample_model_report):
        exporter = BiasReportExporter()
        path = exporter.export_html(
            sample_model_report,
            output_path=tmp_path / "nested" / "deep" / "report.html",
        )
        assert path.exists()

    def test_export_full_report_partial(self, tmp_path, sample_model_report):
        """Full report should work with only some components provided."""
        exporter = BiasReportExporter()
        path = exporter.export_full_report(
            model_report=sample_model_report,
            output_path=tmp_path / "partial.html",
        )
        assert path.exists()


# =====================================================================
# CounterfactualAnalyzer
# =====================================================================


class TestCounterfactualResult:
    def test_max_change(self):
        r = CounterfactualResult(
            user_index=0,
            original_prediction=0.8,
            flips=[
                {"prediction_change": 0.02},
                {"prediction_change": -0.15},
                {"prediction_change": 0.05},
            ],
        )
        assert r.max_change == pytest.approx(0.15)

    def test_is_sensitive(self):
        r = CounterfactualResult(
            user_index=0,
            original_prediction=0.8,
            flips=[{"prediction_change": 0.10}],
        )
        assert r.is_sensitive is True

    def test_not_sensitive(self):
        r = CounterfactualResult(
            user_index=0,
            original_prediction=0.8,
            flips=[{"prediction_change": 0.01}],
        )
        assert r.is_sensitive is False

    def test_empty_flips(self):
        r = CounterfactualResult(user_index=0, original_prediction=0.5)
        assert r.max_change == 0.0
        assert r.is_sensitive is False


class TestCounterfactualAnalyzer:
    @pytest.fixture
    def feature_df(self):
        rng = np.random.default_rng(42)
        n = 50
        return pd.DataFrame(
            {
                "age_group": rng.choice(["18-25", "26-35", "36-50"], n),
                "location": rng.choice(["urban", "suburban", "rural"], n),
                "monthly_budget": rng.normal(3000, 1000, n).clip(500),
                "num_cards": rng.choice([1, 2, 3, 4], n),
            }
        )

    @pytest.fixture
    def fair_model(self):
        """Model that ignores sensitive features (fair)."""

        def predict(X):
            if isinstance(X, pd.DataFrame):
                return X["monthly_budget"].values / 10000
            return np.zeros(len(X))

        return predict

    @pytest.fixture
    def unfair_model(self):
        """Model that heavily depends on age_group (unfair)."""

        def predict(X):
            if isinstance(X, pd.DataFrame):
                base = X["monthly_budget"].values / 10000
                # age_group strongly affects prediction
                age_effect = (
                    X["age_group"]
                    .map({"18-25": -0.3, "26-35": 0.0, "36-50": 0.3})
                    .fillna(0)
                    .values
                )
                return (base + age_effect).clip(0, 1)
            return np.zeros(len(X))

        return predict

    def test_analyze_user(self, feature_df, fair_model):
        analyzer = CounterfactualAnalyzer(predict_fn=fair_model)
        result = analyzer.analyze_user(
            feature_df,
            user_index=0,
            sensitive_columns=["age_group"],
        )
        assert isinstance(result, CounterfactualResult)
        assert result.user_index == 0
        assert len(result.flips) > 0

    def test_fair_model_low_sensitivity(self, feature_df, fair_model):
        """Fair model should show low counterfactual sensitivity."""
        analyzer = CounterfactualAnalyzer(predict_fn=fair_model)
        report = analyzer.analyze_batch(
            feature_df,
            sensitive_columns=["age_group", "location"],
        )
        # Fair model doesn't use age_group → low sensitivity
        assert report.sensitivity_rate < 0.3

    def test_unfair_model_high_sensitivity(self, feature_df, unfair_model):
        """Unfair model should show high counterfactual sensitivity."""
        analyzer = CounterfactualAnalyzer(predict_fn=unfair_model)
        report = analyzer.analyze_batch(
            feature_df,
            sensitive_columns=["age_group"],
        )
        assert report.sensitivity_rate > 0.5

    def test_per_feature_sensitivity(self, feature_df, unfair_model):
        analyzer = CounterfactualAnalyzer(predict_fn=unfair_model)
        report = analyzer.analyze_batch(
            feature_df,
            sensitive_columns=["age_group", "location"],
        )
        per_feat = report.per_feature_sensitivity
        # age_group should be more sensitive than location
        assert per_feat["age_group"] > per_feat["location"]

    def test_sample_size(self, feature_df, fair_model):
        analyzer = CounterfactualAnalyzer(predict_fn=fair_model)
        report = analyzer.analyze_batch(
            feature_df,
            sensitive_columns=["age_group"],
            sample_size=10,
        )
        assert report.n_users == 10

    def test_report_serializable(self, feature_df, fair_model):
        analyzer = CounterfactualAnalyzer(predict_fn=fair_model)
        report = analyzer.analyze_batch(
            feature_df,
            sensitive_columns=["age_group"],
        )
        d = report.to_dict()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 50

    def test_report_summary(self, feature_df, fair_model):
        analyzer = CounterfactualAnalyzer(predict_fn=fair_model)
        report = analyzer.analyze_batch(
            feature_df,
            sensitive_columns=["age_group"],
        )
        s = report.summary
        assert "n_users" in s
        assert "sensitivity_rate" in s
        assert "per_feature_sensitivity" in s

    def test_log_to_mlflow(self, feature_df, fair_model):
        analyzer = CounterfactualAnalyzer(predict_fn=fair_model)
        report = analyzer.analyze_batch(
            feature_df,
            sensitive_columns=["age_group"],
        )
        tracker = MagicMock()
        report.log_to_mlflow(tracker)
        tracker.log_metrics.assert_called()
        tracker.log_dict.assert_called_once()

    def test_missing_column_graceful(self, feature_df, fair_model):
        analyzer = CounterfactualAnalyzer(predict_fn=fair_model)
        result = analyzer.analyze_user(
            feature_df,
            user_index=0,
            sensitive_columns=["nonexistent_col"],
        )
        assert len(result.flips) == 0

    def test_model_param(self, feature_df):
        """Test passing model object instead of predict_fn."""
        mock_model = MagicMock(spec=["predict"])
        mock_model.predict.return_value = np.array([0.5])
        analyzer = CounterfactualAnalyzer(model=mock_model)
        analyzer.analyze_user(
            feature_df,
            user_index=0,
            sensitive_columns=["age_group"],
        )
        assert mock_model.predict.called

    def test_no_model_or_fn_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            CounterfactualAnalyzer()
