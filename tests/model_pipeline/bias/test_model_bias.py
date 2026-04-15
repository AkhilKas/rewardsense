"""
Unit tests for Model Bias Detection & Mitigation.

Covers:
  — SliceEvaluator (slicing, per-slice metrics, disparity flags)
  — ModelBiasDetector (Fairlearn integration on predictions)
  — ScoringBiasChecker + ExplanationBiasChecker
  — ModelBiasMitigator (ExponentiatedGradient, ThresholdOptimizer, sample weights, scoring/prompt adjustment recommendations)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.model_pipeline.bias.slice_evaluator import (
    SliceEvaluator,
    SliceEvaluationReport,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recommendation_diversity,
)
from src.model_pipeline.bias.model_bias_detector import (
    ModelBiasConfig,
    ModelBiasDetector,
    ModelBiasReport,
)
from src.model_pipeline.bias.component_bias import (
    ScoringBiasChecker,
    ExplanationBiasChecker,
    ComponentBiasReport,
)
from src.model_pipeline.bias.model_bias_mitigator import (
    ModelBiasMitigator,
    MitigationResult,
)
from src.model_pipeline.bias.visualizations import (
    plot_slice_metrics,
    plot_disparity_heatmap,
    plot_fairness_metrics,
    plot_bias_summary,
    plot_issuer_distribution,
    plot_explanation_quality,
    plot_mitigation_comparison,
    plot_group_metric_comparison,
)

# =====================================================================
# Shared fixtures
# =====================================================================


@pytest.fixture
def sample_data():
    """Balanced test data with 2 groups."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "user_id": range(n),
            "archetype": (["young_professional"] * 100) + (["suburban_family"] * 100),
            "monthly_budget": rng.normal(3000, 1000, n).clip(500),
            "num_cards": rng.choice([1, 2, 3, 4, 5], n),
            "top_category": rng.choice(["groceries", "dining", "travel", "gas"], n),
            "age_group": rng.choice(["18-25", "26-35", "36-50"], n),
            "location_type": rng.choice(["urban", "suburban", "rural"], n),
        }
    )
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = y_true * 0.8 + rng.normal(0, 0.2, n)  # correlated predictions
    return df, y_true, y_pred


@pytest.fixture
def biased_data():
    """Data where group A gets much better predictions than group B."""
    n = 200
    groups = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.ones(n)
    y_pred = np.concatenate(
        [
            np.ones(100) * 0.9,  # group A: good predictions
            np.ones(100) * 0.3,  # group B: poor predictions
        ]
    )
    df = pd.DataFrame({"group": groups, "val": np.arange(n)})
    return df, y_true, y_pred, groups


# =====================================================================
# SliceEvaluator
# =====================================================================


class TestNDCGAtK:
    def test_perfect_ranking(self):
        y_true = np.array([3, 2, 1, 0, 0])
        y_pred = np.array([5, 4, 3, 2, 1])
        assert ndcg_at_k(y_true, y_pred, k=5) == pytest.approx(1.0)

    def test_worst_ranking(self):
        y_true = np.array([0, 0, 0, 3, 2])
        y_pred = np.array([5, 4, 3, 2, 1])
        assert ndcg_at_k(y_true, y_pred, k=5) < 0.5

    def test_all_zeros(self):
        assert ndcg_at_k(np.zeros(5), np.ones(5), k=5) == 0.0

    def test_single_element(self):
        assert ndcg_at_k(np.array([1.0]), np.array([1.0]), k=1) == 1.0


class TestPrecisionRecallAtK:
    def test_precision_perfect(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([5, 4, 3, 2, 1])
        assert precision_at_k(y_true, y_pred, k=3) == 1.0

    def test_recall_partial(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([5, 4, 3, 2, 1])
        # top-2: indices 0,1 → 1 relevant out of 3 total
        assert recall_at_k(y_true, y_pred, k=2) == pytest.approx(1 / 3)


class TestRecommendationDiversity:
    def test_all_unique(self):
        assert recommendation_diversity(["a", "b", "c"]) == 1.0

    def test_all_same(self):
        assert recommendation_diversity(["a", "a", "a"]) == pytest.approx(1 / 3)

    def test_empty(self):
        assert recommendation_diversity([]) == 0.0


class TestSliceEvaluator:
    def test_categorical_slicing(self, sample_data):
        df, yt, yp = sample_data
        evaluator = SliceEvaluator(
            slicing_config={
                "archetype": {"column": "archetype", "type": "categorical"},
            },
            disparity_threshold=0.10,
        )
        report = evaluator.evaluate(df, yt, yp)
        assert isinstance(report, SliceEvaluationReport)
        assert len(report.slices) == 2  # young_professional, suburban_family
        assert "ndcg_5" in report.overall_metrics

    def test_quantile_slicing(self, sample_data):
        df, yt, yp = sample_data
        evaluator = SliceEvaluator(
            slicing_config={
                "budget_tier": {
                    "column": "monthly_budget",
                    "type": "quantile",
                    "n_quantiles": 4,
                    "labels": ["Q1", "Q2", "Q3", "Q4"],
                },
            },
        )
        report = evaluator.evaluate(df, yt, yp)
        assert len(report.slices) >= 2  # might be fewer if quantiles collapse

    def test_disparity_flagging(self, biased_data):
        df, yt, yp, groups = biased_data
        evaluator = SliceEvaluator(
            slicing_config={
                "group": {"column": "group", "type": "categorical"},
            },
            disparity_threshold=0.05,
        )
        report = evaluator.evaluate(df, yt, yp)
        # With very different predictions, expect disparities
        assert len(report.disparities) > 0

    def test_missing_column_graceful(self, sample_data):
        df, yt, yp = sample_data
        evaluator = SliceEvaluator(
            slicing_config={
                "nonexistent": {"column": "does_not_exist", "type": "categorical"},
            },
        )
        report = evaluator.evaluate(df, yt, yp)
        assert len(report.slices) == 0  # graceful, no crash

    def test_summary(self, sample_data):
        df, yt, yp = sample_data
        evaluator = SliceEvaluator(
            slicing_config={
                "archetype": {"column": "archetype", "type": "categorical"},
            },
        )
        report = evaluator.evaluate(df, yt, yp)
        s = report.summary
        assert "total_slices" in s
        assert "disparities_found" in s

    def test_to_dict_serializable(self, sample_data):
        df, yt, yp = sample_data
        evaluator = SliceEvaluator(
            slicing_config={
                "archetype": {"column": "archetype", "type": "categorical"},
            },
        )
        report = evaluator.evaluate(df, yt, yp)
        d = report.to_dict()
        assert "slices" in d
        assert "overall_metrics" in d

    def test_from_yaml(self, tmp_path):
        cfg = tmp_path / "slices.yaml"
        cfg.write_text(
            "disparity_threshold: 0.15\n"
            "slicing_dimensions:\n"
            "  arch:\n"
            "    column: archetype\n"
            "    type: categorical\n"
        )
        evaluator = SliceEvaluator.from_yaml(cfg)
        assert evaluator.disparity_threshold == 0.15
        assert "arch" in evaluator.slicing_config


# =====================================================================
# ModelBiasDetector
# =====================================================================


class TestModelBiasDetector:
    def test_detect_returns_report(self, sample_data):
        df, yt, yp = sample_data
        detector = ModelBiasDetector()
        report = detector.detect(yt, yp, df["archetype"], model_name="test")
        assert isinstance(report, ModelBiasReport)
        assert report.model_name == "test"
        assert len(report.metrics) > 0

    def test_detect_flags_biased_data(self, biased_data):
        df, yt, yp, groups = biased_data
        detector = ModelBiasDetector(
            config=ModelBiasConfig(performance_disparity_threshold=0.05)
        )
        report = detector.detect(yt, yp, groups, model_name="biased")
        assert len(report.biased_metrics) > 0

    def test_detect_with_dataframe_features(self, sample_data):
        df, yt, yp = sample_data
        detector = ModelBiasDetector()
        sf = df[["archetype", "age_group"]]
        report = detector.detect(yt, yp, sf, model_name="multi")
        # Should have metrics for both features
        features_checked = {m.sensitive_feature for m in report.metrics}
        assert "archetype" in features_checked
        assert "age_group" in features_checked

    def test_summary(self, sample_data):
        df, yt, yp = sample_data
        detector = ModelBiasDetector()
        report = detector.detect(yt, yp, df["archetype"])
        s = report.summary
        assert s["total_metrics"] == len(report.metrics)
        assert s["biased"] == len(report.biased_metrics)

    def test_to_dict_serializable(self, sample_data):
        df, yt, yp = sample_data
        detector = ModelBiasDetector()
        report = detector.detect(yt, yp, df["archetype"])
        d = report.to_dict()
        assert "summary" in d
        assert "all_metrics" in d

    def test_log_to_mlflow(self, sample_data):
        df, yt, yp = sample_data
        detector = ModelBiasDetector()
        report = detector.detect(yt, yp, df["archetype"])
        mock_tracker = MagicMock()
        report.log_to_mlflow(mock_tracker)
        mock_tracker.log_metrics.assert_called_once()
        mock_tracker.log_dict.assert_called_once()

    def test_custom_metrics(self, sample_data):
        df, yt, yp = sample_data
        detector = ModelBiasDetector(
            custom_metrics={
                "mean_error": lambda yt, yp: float(np.mean(np.abs(yt - yp)))
            }
        )
        report = detector.detect(yt, yp, df["archetype"])
        custom = [m for m in report.metrics if "custom" in m.name]
        assert len(custom) > 0

    def test_small_group_skipped(self):
        yt = np.array([1, 0, 1, 0, 1, 0])
        yp = np.array([1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "A", "B"])  # B has only 1
        detector = ModelBiasDetector(config=ModelBiasConfig(min_slice_size=5))
        report = detector.detect(yt, yp, groups)
        # performance_disparity should skip group B (size=1 < min=5)
        # The check should still complete without error
        assert isinstance(report, ModelBiasReport)


# =====================================================================
# ScoringBiasChecker + ExplanationBiasChecker
# =====================================================================


class TestScoringBiasChecker:
    @pytest.fixture
    def recommendations_df(self):
        rng = np.random.default_rng(42)
        n = 300
        return pd.DataFrame(
            {
                "user_id": range(n),
                "spending_archetype": (
                    ["young_professional"] * 150 + ["suburban_family"] * 150
                ),
                "recommended_card_issuer": rng.choice(
                    ["Chase", "Amex", "Capital One"], n, p=[0.5, 0.3, 0.2]
                ),
                "recommended_card_type": rng.choice(
                    ["premium", "standard"], n, p=[0.4, 0.6]
                ),
            }
        )

    def test_issuer_bias_check(self, recommendations_df):
        checker = ScoringBiasChecker()
        report = checker.check_issuer_bias(recommendations_df, "spending_archetype")
        assert isinstance(report, ComponentBiasReport)
        assert report.component == "scoring_engine"
        assert len(report.metrics) == 3  # one per issuer

    def test_card_type_bias_check(self, recommendations_df):
        checker = ScoringBiasChecker()
        report = checker.check_card_type_bias(recommendations_df, "spending_archetype")
        assert len(report.metrics) == 1  # premium disparity

    def test_missing_column_graceful(self, recommendations_df):
        checker = ScoringBiasChecker()
        report = checker.check_issuer_bias(
            recommendations_df,
            "spending_archetype",
            issuer_col="nonexistent",
        )
        assert len(report.metrics) == 0

    def test_biased_issuer_flagged(self):
        """Construct data where one issuer is heavily recommended for one group."""
        df = pd.DataFrame(
            {
                "group": ["A"] * 100 + ["B"] * 100,
                "recommended_card_issuer": (
                    ["Chase"] * 95
                    + ["Amex"] * 5  # A overwhelmingly gets Chase
                    + ["Amex"] * 90
                    + ["Chase"] * 10  # B overwhelmingly gets Amex
                ),
            }
        )
        checker = ScoringBiasChecker(issuer_disparity_threshold=0.10)
        report = checker.check_issuer_bias(df, "group")
        assert len(report.biased_metrics) > 0

    def test_log_to_mlflow(self, recommendations_df):
        checker = ScoringBiasChecker()
        report = checker.check_issuer_bias(recommendations_df, "spending_archetype")
        mock_tracker = MagicMock()
        report.log_to_mlflow(mock_tracker)
        mock_tracker.log_metrics.assert_called_once()


class TestExplanationBiasChecker:
    @pytest.fixture
    def explanations_df(self):
        groups = ["young_professional"] * 100 + ["suburban_family"] * 100
        texts = []
        for g in groups:
            if g == "young_professional":
                texts.append(
                    "Use your Chase Sapphire for dining to earn 3x points. " * 2
                )
            else:
                texts.append("Consider Amex Gold. " * 1)
        return pd.DataFrame(
            {
                "user_segment": groups,
                "explanation_text": texts,
            }
        )

    def test_quality_consistency(self, explanations_df):
        checker = ExplanationBiasChecker()
        report = checker.check_quality_consistency(explanations_df, "user_segment")
        assert isinstance(report, ComponentBiasReport)
        assert report.component == "llm_explainability"
        assert len(report.metrics) >= 2  # length + readability + detail

    def test_length_disparity_detected(self, explanations_df):
        """Explanations differ significantly in length between groups."""
        checker = ExplanationBiasChecker(length_disparity_threshold=0.05)
        report = checker.check_quality_consistency(explanations_df, "user_segment")
        length_checks = [m for m in report.metrics if "length" in m.check_name]
        assert len(length_checks) > 0
        # The fixture has 2x length for young_professional
        assert any(m.is_biased for m in length_checks)

    def test_missing_text_col_graceful(self, explanations_df):
        checker = ExplanationBiasChecker()
        report = checker.check_quality_consistency(
            explanations_df, "user_segment", text_col="nonexistent"
        )
        assert len(report.metrics) == 0


# =====================================================================
# ModelBiasMitigator
# =====================================================================


class TestModelBiasMitigator:
    def test_sample_weights(self):
        groups = pd.Series(["A"] * 80 + ["B"] * 20)
        mitigator = ModelBiasMitigator()
        weights = mitigator.compute_sample_weights(groups)
        assert len(weights) == 100
        # B group should have higher weights (minority)
        assert np.mean(weights[80:]) > np.mean(weights[:80])

    def test_sample_weights_balanced(self):
        groups = pd.Series(["A"] * 50 + ["B"] * 50)
        mitigator = ModelBiasMitigator()
        weights = mitigator.compute_sample_weights(groups)
        # Balanced → equal weights
        assert np.allclose(weights, 1.0)

    def test_scoring_adjustments_no_bias(self):
        report = ComponentBiasReport(component="scoring_engine")
        mitigator = ModelBiasMitigator()
        adj = mitigator.recommend_scoring_adjustments(report)
        assert adj["apply_diversity_penalty"] is False

    def test_scoring_adjustments_with_bias(self):
        from src.model_pipeline.bias.component_bias import ComponentBiasMetric

        report = ComponentBiasReport(component="scoring_engine")
        report.metrics.append(
            ComponentBiasMetric(
                component="scoring_engine",
                check_name="issuer_disparity_Chase",
                sensitive_feature="archetype",
                value=0.25,
                threshold=0.15,
                is_biased=True,
                details={
                    "issuer": "Chase",
                    "per_group_rates": {"A": 0.80, "B": 0.55},
                    "max_rate": 0.80,
                },
            )
        )
        mitigator = ModelBiasMitigator()
        adj = mitigator.recommend_scoring_adjustments(report)
        assert adj["apply_diversity_penalty"] is True
        assert "Chase" in adj["issuer_caps"]

    def test_prompt_adjustments_no_bias(self):
        report = ComponentBiasReport(component="llm_explainability")
        mitigator = ModelBiasMitigator()
        adj = mitigator.recommend_prompt_adjustments(report)
        assert adj["modify_prompts"] is False

    def test_prompt_adjustments_with_length_bias(self):
        from src.model_pipeline.bias.component_bias import ComponentBiasMetric

        report = ComponentBiasReport(component="llm_explainability")
        report.metrics.append(
            ComponentBiasMetric(
                component="llm_explainability",
                check_name="explanation_length_disparity",
                sensitive_feature="archetype",
                value=0.30,
                threshold=0.20,
                is_biased=True,
            )
        )
        mitigator = ModelBiasMitigator()
        adj = mitigator.recommend_prompt_adjustments(report)
        assert adj["modify_prompts"] is True
        assert any("length" in s.lower() for s in adj["suggestions"])

    def test_log_comparison(self):
        before = MagicMock()
        before.to_dict.return_value = {"before": True}
        before.biased_metrics = [1, 2, 3]
        after = MagicMock()
        after.to_dict.return_value = {"after": True}
        after.biased_metrics = [1]

        tracker = MagicMock()
        ModelBiasMitigator.log_comparison(
            tracker, before, after, strategy_name="exp_grad"
        )
        tracker.log_metrics.assert_called_once()
        tracker.log_dict.assert_called_once()

    def test_mitigation_result_improvement(self):
        r = MitigationResult(
            strategy="test",
            component="personalization",
            before_metrics={"bias_flagged": 5, "accuracy": 0.85},
            after_metrics={"bias_flagged": 2, "accuracy": 0.82},
        )
        imp = r.improvement
        assert imp["bias_flagged"] == -3
        assert imp["accuracy"] == pytest.approx(-0.03)

    def test_exponentiated_gradient_without_fairlearn(self):
        """Should return error result when Fairlearn reductions unavailable."""
        mitigator = ModelBiasMitigator()
        # Mock the availability flag
        import src.model_pipeline.bias.model_bias_mitigator as mod

        original = mod.FAIRLEARN_REDUCTIONS_AVAILABLE
        mod.FAIRLEARN_REDUCTIONS_AVAILABLE = False
        try:
            result = mitigator.mitigate_with_exponentiated_gradient(
                estimator=MagicMock(),
                X_train=np.zeros((10, 3)),
                y_train=np.zeros(10),
                sensitive_features=np.array(["A"] * 5 + ["B"] * 5),
            )
            assert "error" in result.trade_offs
        finally:
            mod.FAIRLEARN_REDUCTIONS_AVAILABLE = original


# =====================================================================
# Visualizations
# =====================================================================

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestVisualizationsSlice:
    """Test Story 6.1 visualizations."""

    def test_plot_slice_metrics_returns_figure(self):
        slices = [
            {"name": "Q1", "metrics": {"ndcg_5": 0.7, "precision_5": 0.6}},
            {"name": "Q2", "metrics": {"ndcg_5": 0.8, "precision_5": 0.75}},
            {"name": "Q3", "metrics": {"ndcg_5": 0.85, "precision_5": 0.8}},
        ]
        fig = plot_slice_metrics(slices, metrics=("ndcg_5", "precision_5"))
        assert fig is not None
        assert hasattr(fig, "savefig")
        _plt.close(fig)

    def test_plot_slice_metrics_with_overall(self):
        slices = [
            {"name": "A", "metrics": {"ndcg_5": 0.7}},
            {"name": "B", "metrics": {"ndcg_5": 0.9}},
        ]
        fig = plot_slice_metrics(
            slices,
            metrics=("ndcg_5",),
            overall={"ndcg_5": 0.8},
        )
        assert fig is not None
        _plt.close(fig)

    def test_plot_disparity_heatmap(self):
        slices = [
            {"name": "low", "metrics": {"ndcg_5": 0.6, "precision_5": 0.5}},
            {"name": "high", "metrics": {"ndcg_5": 0.9, "precision_5": 0.85}},
        ]
        overall = {"ndcg_5": 0.75, "precision_5": 0.7}
        fig = plot_disparity_heatmap(slices, overall)
        assert fig is not None
        _plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestVisualizationsFairness:
    """Test Story 6.2 visualizations."""

    def test_plot_fairness_metrics(self):
        per_group = {
            "archetype": {"young_prof": 0.82, "family": 0.78, "traveler": 0.85},
        }
        fig = plot_fairness_metrics(per_group)
        assert fig is not None
        _plt.close(fig)

    def test_plot_bias_summary(self):
        metrics = [
            {
                "name": "dem_parity",
                "sensitive_feature": "arch",
                "value": 0.12,
                "threshold": 0.10,
                "is_biased": True,
            },
            {
                "name": "eq_odds",
                "sensitive_feature": "arch",
                "value": 0.05,
                "threshold": 0.10,
                "is_biased": False,
            },
        ]
        fig = plot_bias_summary(metrics)
        assert fig is not None
        _plt.close(fig)

    def test_plot_bias_summary_empty(self):
        fig = plot_bias_summary([])
        assert fig is not None
        _plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestVisualizationsComponent:
    """Test Story 6.3 visualizations."""

    def test_plot_issuer_distribution(self):
        metrics = [
            {
                "check": "issuer_Chase",
                "details": {
                    "issuer": "Chase",
                    "per_group_rates": {"young_prof": 0.6, "family": 0.4},
                },
            },
            {
                "check": "issuer_Amex",
                "details": {
                    "issuer": "Amex",
                    "per_group_rates": {"young_prof": 0.3, "family": 0.5},
                },
            },
        ]
        fig = plot_issuer_distribution(metrics)
        assert fig is not None
        _plt.close(fig)

    def test_plot_issuer_distribution_empty(self):
        fig = plot_issuer_distribution([])
        assert fig is not None
        _plt.close(fig)

    def test_plot_explanation_quality(self):
        metrics = [
            {
                "check_name": "explanation_length",
                "details": {
                    "overall_mean_length": 120,
                    "per_group_mean_length": {"young_prof": 150, "family": 90},
                },
            },
        ]
        fig = plot_explanation_quality(metrics)
        assert fig is not None
        _plt.close(fig)

    def test_plot_explanation_quality_empty(self):
        fig = plot_explanation_quality([])
        assert fig is not None
        _plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestVisualizationsMitigation:
    """Test Story 6.4 visualizations."""

    def test_plot_mitigation_comparison(self):
        before = {"dem_parity": 0.15, "eq_odds": 0.12, "accuracy": 0.85}
        after = {"dem_parity": 0.08, "eq_odds": 0.06, "accuracy": 0.83}
        fig = plot_mitigation_comparison(before, after)
        assert fig is not None
        _plt.close(fig)

    def test_plot_group_metric_comparison(self):
        before = {"A": 0.90, "B": 0.70}
        after = {"A": 0.85, "B": 0.82}
        fig = plot_group_metric_comparison(before, after, metric_name="accuracy")
        assert fig is not None
        _plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestLogToMlflowWithViz:
    """Test that log_to_mlflow calls produce figures."""

    def test_model_bias_report_logs_figures(self, sample_data):
        df, yt, yp = sample_data
        detector = ModelBiasDetector()
        report = detector.detect(yt, yp, df["archetype"])
        mock_tracker = MagicMock()
        report.log_to_mlflow(mock_tracker)
        # Should have at least: log_metrics, log_dict, log_figure calls
        assert mock_tracker.log_metrics.called
        assert mock_tracker.log_dict.called
        assert mock_tracker.log_figure.called

    def test_slice_report_logs_figures(self, sample_data):
        df, yt, yp = sample_data
        evaluator = SliceEvaluator(
            slicing_config={
                "archetype": {"column": "archetype", "type": "categorical"},
            },
        )
        report = evaluator.evaluate(df, yt, yp)
        mock_tracker = MagicMock()
        report.log_to_mlflow(mock_tracker)
        assert mock_tracker.log_figure.called

    def test_component_report_logs_figures(self):
        from src.model_pipeline.bias.component_bias import ComponentBiasMetric

        report = ComponentBiasReport(component="scoring_engine")
        report.metrics.append(
            ComponentBiasMetric(
                component="scoring_engine",
                check_name="issuer_disparity_Chase",
                sensitive_feature="archetype",
                value=0.20,
                threshold=0.15,
                is_biased=True,
                details={
                    "issuer": "Chase",
                    "per_group_rates": {"A": 0.7, "B": 0.5},
                },
            )
        )
        mock_tracker = MagicMock()
        report.log_to_mlflow(mock_tracker)
        assert mock_tracker.log_figure.called
