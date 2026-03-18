"""Tests for model_pipeline.personalization.trainer."""

import pytest

from model_pipeline.personalization.splits import split_data
from model_pipeline.personalization.trainer import Trainer, TrainingResult


class TestTrainer:
    @pytest.fixture()
    def split_result(self, xy_pair, joined_df):
        X, y = xy_pair
        meta = joined_df[["user_id", "archetype", "age_group"]].loc[X.index]
        return split_data(X, y, meta=meta)

    def test_train_all_returns_result(self, split_result, tmp_path):
        trainer = Trainer(
            split=split_result,
            model_names=["mean", "random_forest"],
            model_params={"random_forest": {"n_estimators": 5}},
            use_mlflow=False,
            artifact_dir=str(tmp_path),
        )
        result = trainer.train_all()
        assert isinstance(result, TrainingResult)
        assert len(result.records) == 2
        assert result.best_model_name is not None

    def test_best_model_has_lowest_val_rmse(self, split_result, tmp_path):
        trainer = Trainer(
            split=split_result,
            model_names=["mean", "random_forest"],
            model_params={"random_forest": {"n_estimators": 5}},
            use_mlflow=False,
            artifact_dir=str(tmp_path),
        )
        result = trainer.train_all()
        best_rmse = result.best_record.val_metrics.rmse
        for r in result.records:
            assert r.val_metrics.rmse >= best_rmse

    def test_leaderboard_shape(self, split_result, tmp_path):
        trainer = Trainer(
            split=split_result,
            model_names=["mean"],
            use_mlflow=False,
            artifact_dir=str(tmp_path),
        )
        result = trainer.train_all()
        lb = result.leaderboard
        assert len(lb) == 1
        assert "val_rmse" in lb.columns

    def test_model_artifact_saved(self, split_result, tmp_path):
        trainer = Trainer(
            split=split_result,
            model_names=["mean"],
            use_mlflow=False,
            artifact_dir=str(tmp_path),
        )
        trainer.train_all()
        artifacts = list(tmp_path.glob("best_model_*.joblib"))
        assert len(artifacts) == 1
