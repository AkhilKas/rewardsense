"""Tests for model_pipeline.personalization.tuning."""

import pytest

from model_pipeline.personalization.tuning import HyperparameterTuner


class TestHyperparameterTuner:
    def test_tune_returns_best_params(self, xy_pair):
        X, y = xy_pair
        tuner = HyperparameterTuner(
            model_name="xgboost",
            X_train=X,
            y_train=y,
            n_trials=3,
            cv_folds=2,
            use_mlflow=False,
        )
        best = tuner.tune()
        assert isinstance(best, dict)
        assert "max_depth" in best
        assert "learning_rate" in best
        assert tuner.best_score is not None
        assert tuner.best_score > 0

    def test_retrain_best(self, xy_pair):
        X, y = xy_pair
        tuner = HyperparameterTuner(
            model_name="xgboost",
            X_train=X,
            y_train=y,
            n_trials=3,
            cv_folds=2,
            use_mlflow=False,
        )
        tuner.tune()
        model = tuner.retrain_best()
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_retrain_before_tune_raises(self, xy_pair):
        X, y = xy_pair
        tuner = HyperparameterTuner(
            model_name="xgboost",
            X_train=X,
            y_train=y,
            use_mlflow=False,
        )
        with pytest.raises(RuntimeError, match="tune"):
            tuner.retrain_best()

    def test_trials_dataframe(self, xy_pair):
        X, y = xy_pair
        tuner = HyperparameterTuner(
            model_name="xgboost",
            X_train=X,
            y_train=y,
            n_trials=3,
            cv_folds=2,
            use_mlflow=False,
        )
        tuner.tune()
        df = tuner.get_trials_dataframe()
        assert len(df) == 3
