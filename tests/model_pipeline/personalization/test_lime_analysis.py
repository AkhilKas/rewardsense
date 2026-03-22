"""Tests for LIME-based local explanation analysis (Story 5.1)."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.model_pipeline.personalization.sensitivity.lime_analysis import LIMEAnalyzer


@pytest.fixture()
def trained_rf(xy_pair):
    X, y = xy_pair
    rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    rf.fit(X, y)
    return rf


@pytest.fixture()
def lime_analyzer(trained_rf, xy_pair):
    X, _ = xy_pair
    return LIMEAnalyzer(model=trained_rf, X_train=X)


class TestLIMEAnalyzer:
    def test_explain_instance_keys(self, lime_analyzer, xy_pair):
        X, _ = xy_pair
        exp = lime_analyzer.explain_instance(X.iloc[0].values, num_features=5)
        assert "feature_weights" in exp
        assert isinstance(exp["feature_weights"], dict)

    def test_explain_instance_num_features(self, lime_analyzer, xy_pair):
        X, _ = xy_pair
        exp = lime_analyzer.explain_instance(X.iloc[0].values, num_features=3)
        assert len(exp["feature_weights"]) <= 5  # LIME may return fewer

    def test_explain_batch_returns_list(self, lime_analyzer, xy_pair):
        X, _ = xy_pair
        explanations = lime_analyzer.explain_batch(X, num_features=3, max_samples=5)
        assert isinstance(explanations, list)
        assert len(explanations) == 5

    def test_aggregate_importance_columns(self, lime_analyzer, xy_pair):
        X, _ = xy_pair
        explanations = lime_analyzer.explain_batch(X, num_features=5, max_samples=10)
        agg = lime_analyzer.aggregate_importance(explanations)
        assert "feature" in agg.columns
        assert "mean_abs_weight" in agg.columns
        assert "rank" in agg.columns

    def test_aggregate_importance_sorted(self, lime_analyzer, xy_pair):
        X, _ = xy_pair
        explanations = lime_analyzer.explain_batch(X, num_features=5, max_samples=10)
        agg = lime_analyzer.aggregate_importance(explanations)
        vals = agg["mean_abs_weight"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_compare_with_shap_keys(self, lime_analyzer):
        shap_imp = pd.DataFrame({"feature": ["a", "b", "c"], "rank": [1, 2, 3]})
        lime_imp = pd.DataFrame({"feature": ["a", "b", "c"], "rank": [2, 1, 3]})
        result = lime_analyzer.compare_with_shap(shap_imp, lime_imp)
        assert "spearman_rho" in result
        assert "p_value" in result
        assert "top_5_overlap_pct" in result

    def test_compare_with_shap_perfect_correlation(self, lime_analyzer):
        imp = pd.DataFrame(
            {"feature": ["a", "b", "c", "d", "e"], "rank": [1, 2, 3, 4, 5]}
        )
        result = lime_analyzer.compare_with_shap(imp, imp)
        assert result["spearman_rho"] == pytest.approx(1.0)
        assert result["top_5_overlap_pct"] == 100.0

    def test_compare_with_shap_no_overlap(self, lime_analyzer):
        shap_imp = pd.DataFrame({"feature": ["a", "b"], "rank": [1, 2]})
        lime_imp = pd.DataFrame({"feature": ["c", "d"], "rank": [1, 2]})
        result = lime_analyzer.compare_with_shap(shap_imp, lime_imp)
        assert result["spearman_rho"] is None

    def test_generate_importance_plot(self, lime_analyzer, xy_pair):
        import matplotlib.pyplot as plt

        X, _ = xy_pair
        explanations = lime_analyzer.explain_batch(X, num_features=3, max_samples=5)
        agg = lime_analyzer.aggregate_importance(explanations)
        fig = lime_analyzer.generate_importance_plot(agg, max_display=5)
        assert fig is not None
        plt.close("all")
