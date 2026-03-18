"""Tests for SHAP-based feature importance analysis (Story 5.1)."""

import pytest
from sklearn.ensemble import RandomForestRegressor

from model_pipeline.personalization.sensitivity.shap_analysis import (
    SHAPAnalyzer,
    _is_tree_model,
)


@pytest.fixture()
def trained_rf(xy_pair):
    """Fit a small RandomForest on the synthetic data."""
    X, y = xy_pair
    rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    rf.fit(X, y)
    return rf


@pytest.fixture()
def analyzer(trained_rf, xy_pair):
    X, _ = xy_pair
    return SHAPAnalyzer(model=trained_rf, X=X)


class TestIsTreeModel:
    def test_random_forest_detected(self, trained_rf):
        assert _is_tree_model(trained_rf) is True

    def test_non_tree_not_detected(self):
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        assert _is_tree_model(lr) is False


class TestSHAPAnalyzer:
    def test_compute_shap_values_shape(self, analyzer, xy_pair):
        X, _ = xy_pair
        sv = analyzer.compute_shap_values()
        assert sv.shape == X.shape

    def test_shap_values_cached(self, analyzer):
        sv1 = analyzer.compute_shap_values()
        sv2 = analyzer.compute_shap_values()
        assert sv1 is sv2

    def test_global_feature_importance_columns(self, analyzer):
        imp = analyzer.global_feature_importance()
        assert "feature" in imp.columns
        assert "mean_abs_shap" in imp.columns
        assert "rank" in imp.columns

    def test_importance_sorted_descending(self, analyzer):
        imp = analyzer.global_feature_importance()
        vals = imp["mean_abs_shap"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_top_features_returns_correct_count(self, analyzer):
        top = analyzer.top_features(n=3)
        assert len(top) == 3
        assert all(isinstance(f, str) for f in top)

    def test_force_plot_data_structure(self, analyzer):
        data = analyzer.generate_force_plot_data(idx=0)
        assert "base_value" in data
        assert "shap_values" in data
        assert "feature_values" in data
        assert "predicted_value" in data
        assert isinstance(data["base_value"], float)

    def test_generate_summary_plot(self, analyzer):
        import matplotlib.pyplot as plt

        fig = analyzer.generate_summary_plot(max_display=5)
        assert fig is not None
        plt.close("all")

    def test_generate_bar_plot(self, analyzer):
        import matplotlib.pyplot as plt

        fig = analyzer.generate_bar_plot(max_display=5)
        assert fig is not None
        plt.close("all")

    def test_generate_dependence_plot(self, analyzer, xy_pair):
        import matplotlib.pyplot as plt

        X, _ = xy_pair
        feat = X.columns[0]
        fig = analyzer.generate_dependence_plot(feat)
        assert fig is not None
        plt.close("all")
