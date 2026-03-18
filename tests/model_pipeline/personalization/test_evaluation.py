"""Tests for model_pipeline.personalization.evaluation."""

import numpy as np
import pandas as pd

from model_pipeline.personalization.evaluation import (
    EvaluationReport,
    RegressionMetrics,
    check_overfitting,
    compute_regression_metrics,
    compute_segment_metrics,
    evaluate,
)


class TestComputeRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_regression_metrics(y, y)
        assert m.rmse == 0.0
        assert m.mae == 0.0
        assert m.r2 == 1.0

    def test_constant_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        m = compute_regression_metrics(y_true, y_pred)
        assert m.rmse > 0
        assert m.mae > 0
        assert m.r2 < 1.0

    def test_to_dict(self):
        m = RegressionMetrics(rmse=0.1, mae=0.05, r2=0.9)
        d = m.to_dict()
        assert set(d.keys()) == {"rmse", "mae", "r2"}


class TestComputeSegmentMetrics:
    def test_returns_per_segment(self):
        y_true = pd.Series([0.01, 0.02, 0.015, 0.025, 0.01, 0.03])
        y_pred = np.array([0.011, 0.019, 0.016, 0.024, 0.012, 0.028])
        meta = pd.DataFrame(
            {"archetype": ["A", "A", "B", "B", "A", "B"]},
            index=y_true.index,
        )
        result = compute_segment_metrics(y_true, y_pred, meta, ["archetype"])
        assert "archetype" in result
        assert "A" in result["archetype"]
        assert "B" in result["archetype"]

    def test_skips_single_element_groups(self):
        y_true = pd.Series([0.01, 0.02])
        y_pred = np.array([0.01, 0.02])
        meta = pd.DataFrame({"seg": ["A", "B"]}, index=y_true.index)
        result = compute_segment_metrics(y_true, y_pred, meta, ["seg"])
        assert len(result.get("seg", {})) == 0


class TestCheckOverfitting:
    def test_no_overfit(self):
        train = RegressionMetrics(rmse=0.05, mae=0.03, r2=0.9)
        val = RegressionMetrics(rmse=0.06, mae=0.04, r2=0.85)
        result = check_overfitting(train, val)
        assert result["is_overfit"] is False

    def test_overfit_detected(self):
        train = RegressionMetrics(rmse=0.01, mae=0.005, r2=0.99)
        val = RegressionMetrics(rmse=0.50, mae=0.30, r2=0.30)
        result = check_overfitting(train, val)
        assert result["is_overfit"] is True


class TestEvaluate:
    def test_returns_evaluation_report(self):
        y_true = pd.Series([0.01, 0.02, 0.015, 0.025])
        y_pred = np.array([0.011, 0.019, 0.016, 0.024])
        report = evaluate(y_true, y_pred)
        assert isinstance(report, EvaluationReport)
        assert report.overall.rmse >= 0

    def test_to_dict_serializable(self):
        y_true = pd.Series([0.01, 0.02, 0.015])
        y_pred = np.array([0.011, 0.019, 0.016])
        report = evaluate(y_true, y_pred)
        d = report.to_dict()
        assert "overall" in d
