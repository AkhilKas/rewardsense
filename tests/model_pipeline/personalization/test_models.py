"""Tests for model_pipeline.personalization.models."""

import numpy as np
import pytest

from model_pipeline.personalization.models import (
    MeanBaselineRegressor,
    create_model,
    list_models,
)


class TestMeanBaseline:
    def test_fit_and_predict(self):
        X = np.array([[1], [2], [3]])
        y = np.array([10.0, 20.0, 30.0])
        model = MeanBaselineRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        assert np.allclose(preds, 20.0)

    def test_predict_before_fit_raises(self):
        model = MeanBaselineRegressor()
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1]]))


class TestCreateModel:
    def test_create_mean(self):
        model = create_model("mean")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_random_forest(self):
        model = create_model("random_forest", {"n_estimators": 10})
        assert hasattr(model, "fit")

    def test_create_xgboost(self):
        model = create_model("xgboost", {"n_estimators": 10})
        assert hasattr(model, "fit")

    def test_create_mlp(self):
        model = create_model("mlp", {"max_iter": 10})
        assert hasattr(model, "fit")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")


class TestListModels:
    def test_returns_expected_names(self):
        names = list_models()
        assert "mean" in names
        assert "random_forest" in names
        assert "xgboost" in names
        assert "mlp" in names


class TestEndToEnd:
    def test_all_models_fit_and_predict(self, xy_pair):
        X, y = xy_pair
        param_overrides = {
            "mean": {},
            "random_forest": {"n_estimators": 5},
            "xgboost": {"n_estimators": 5},
            "mlp": {"max_iter": 10},
        }
        for name in list_models():
            model = create_model(name, param_overrides.get(name, {}))
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y)
            assert not np.any(np.isnan(preds))
