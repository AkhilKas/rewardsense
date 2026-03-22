"""Tests for hyperparameter sensitivity analysis (Story 5.2)."""

import numpy as np
import pandas as pd
import pytest

from src.model_pipeline.personalization.sensitivity.hyperparameter_sensitivity import (
    HyperparameterAnalyzer,
    HPSensitivityResult,
    ParameterRange,
)


@pytest.fixture()
def mock_trials_df():
    """Simulate an Optuna trials DataFrame with 30 trials."""
    rng = np.random.RandomState(42)
    n = 30
    max_depth = rng.randint(3, 11, n)
    learning_rate = rng.uniform(0.01, 0.3, n)
    n_estimators = rng.choice([50, 100, 150, 200, 250, 300], n)
    subsample = rng.uniform(0.6, 1.0, n)
    colsample = rng.uniform(0.6, 1.0, n)

    # Synthetic metric correlated with learning_rate and max_depth
    noise = rng.normal(0, 0.02, n)
    value = 0.5 - 0.3 * learning_rate + 0.02 * max_depth + noise

    return pd.DataFrame(
        {
            "number": range(n),
            "value": value.round(4),
            "params_max_depth": max_depth,
            "params_learning_rate": learning_rate.round(4),
            "params_n_estimators": n_estimators,
            "params_subsample": subsample.round(4),
            "params_colsample_bytree": colsample.round(4),
        }
    )


@pytest.fixture()
def hp_analyzer(mock_trials_df):
    return HyperparameterAnalyzer(
        trials_df=mock_trials_df,
        metric_column="value",
        metric_direction="minimize",
    )


class TestHyperparameterAnalyzer:
    def test_param_names_detected(self, hp_analyzer):
        names = hp_analyzer.param_names
        assert "max_depth" in names
        assert "learning_rate" in names
        assert len(names) == 5

    def test_compute_importance_columns(self, hp_analyzer):
        imp = hp_analyzer.compute_parameter_importance()
        assert "param" in imp.columns
        assert "abs_correlation" in imp.columns
        assert "rank" in imp.columns

    def test_importance_sorted_descending(self, hp_analyzer):
        imp = hp_analyzer.compute_parameter_importance()
        vals = imp["abs_correlation"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_top_params_include_key_hps(self, hp_analyzer):
        """learning_rate and max_depth should both appear in top 3."""
        imp = hp_analyzer.compute_parameter_importance()
        top3 = imp.head(3)["param"].tolist()
        assert "learning_rate" in top3 or "max_depth" in top3

    def test_identify_safe_ranges(self, hp_analyzer):
        imp = hp_analyzer.compute_parameter_importance()
        ranges = hp_analyzer.identify_safe_ranges(imp)
        assert len(ranges) > 0
        assert all(isinstance(r, ParameterRange) for r in ranges)

    def test_safe_range_bounds(self, hp_analyzer):
        imp = hp_analyzer.compute_parameter_importance()
        ranges = hp_analyzer.identify_safe_ranges(imp)
        for r in ranges:
            assert r.lower <= r.upper

    def test_compute_interactions(self, hp_analyzer):
        interactions = hp_analyzer.compute_interactions(top_n=3)
        assert "param_1" in interactions.columns
        assert "param_2" in interactions.columns
        assert "interaction_strength" in interactions.columns

    def test_run_returns_result(self, hp_analyzer):
        result = hp_analyzer.run()
        assert isinstance(result, HPSensitivityResult)
        assert not result.importance_ranking.empty
        assert len(result.safe_ranges) > 0

    def test_result_to_dict(self, hp_analyzer):
        result = hp_analyzer.run()
        d = result.to_dict()
        assert "importance_ranking" in d
        assert "safe_ranges" in d
        assert "interactions" in d

    def test_param_vs_metric_plot(self, hp_analyzer):
        import matplotlib.pyplot as plt

        fig = hp_analyzer.generate_param_vs_metric_plot("learning_rate")
        assert fig is not None
        plt.close("all")

    def test_interaction_plot(self, hp_analyzer):
        import matplotlib.pyplot as plt

        fig = hp_analyzer.generate_interaction_plot("learning_rate", "max_depth")
        assert fig is not None
        plt.close("all")

    def test_importance_bar_plot(self, hp_analyzer):
        import matplotlib.pyplot as plt

        imp = hp_analyzer.compute_parameter_importance()
        fig = hp_analyzer.generate_importance_bar_plot(imp)
        assert fig is not None
        plt.close("all")
