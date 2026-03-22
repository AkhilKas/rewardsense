"""Tests for model_pipeline.personalization.validation."""

import pytest

from src.model_pipeline.personalization.models import create_model
from src.model_pipeline.personalization.splits import split_data
from src.model_pipeline.personalization.validation import (
    HoldoutValidator,
    ValidationVerdict,
)


class TestHoldoutValidator:
    @pytest.fixture()
    def trained_setup(self, xy_pair, joined_df):
        X, y = xy_pair
        meta = joined_df[["user_id", "archetype", "age_group"]].loc[X.index]
        split = split_data(X, y, meta=meta)

        model = create_model("random_forest", {"n_estimators": 10, "random_state": 42})
        model.fit(split.X_train, split.y_train)

        val_pred = model.predict(split.X_val)
        from src.model_pipeline.personalization.evaluation import (
            compute_regression_metrics,
        )

        val_metrics = compute_regression_metrics(split.y_val.values, val_pred)

        return model, split, val_metrics

    def test_validate_returns_verdict(self, trained_setup, tmp_path):
        model, split, val_metrics = trained_setup
        validator = HoldoutValidator(
            model=model,
            split=split,
            val_metrics=val_metrics,
            rmse_threshold=1.0,
            r2_threshold=-10.0,
            artifact_dir=str(tmp_path),
        )
        verdict = validator.validate()
        assert isinstance(verdict, ValidationVerdict)
        assert isinstance(verdict.passed, bool)

    def test_strict_threshold_fails(self, trained_setup, tmp_path):
        model, split, val_metrics = trained_setup
        validator = HoldoutValidator(
            model=model,
            split=split,
            val_metrics=val_metrics,
            rmse_threshold=0.0001,
            r2_threshold=0.999,
            artifact_dir=str(tmp_path),
        )
        verdict = validator.validate()
        assert verdict.passed is False

    def test_report_saved(self, trained_setup, tmp_path):
        model, split, val_metrics = trained_setup
        validator = HoldoutValidator(
            model=model,
            split=split,
            val_metrics=val_metrics,
            rmse_threshold=1.0,
            r2_threshold=-10.0,
            artifact_dir=str(tmp_path),
        )
        validator.validate()
        report_file = tmp_path / "holdout_validation_report.json"
        assert report_file.exists()

    def test_verdict_to_dict(self, trained_setup, tmp_path):
        model, split, val_metrics = trained_setup
        validator = HoldoutValidator(
            model=model,
            split=split,
            val_metrics=val_metrics,
            rmse_threshold=1.0,
            r2_threshold=-10.0,
            artifact_dir=str(tmp_path),
        )
        verdict = validator.validate()
        d = verdict.to_dict()
        assert "passed" in d
        assert "test_metrics" in d
        assert "timestamp" in d

    def test_validation_plots_generated(self, trained_setup, tmp_path):
        model, split, val_metrics = trained_setup
        validator = HoldoutValidator(
            model=model,
            split=split,
            val_metrics=val_metrics,
            rmse_threshold=1.0,
            r2_threshold=-10.0,
            artifact_dir=str(tmp_path),
        )
        plots = validator.generate_validation_plots()
        assert "pred_vs_actual" in plots
        assert "residual_histogram" in plots
        assert plots["pred_vs_actual"].exists()
        assert plots["residual_histogram"].exists()

    def test_validate_also_creates_plots(self, trained_setup, tmp_path):
        model, split, val_metrics = trained_setup
        validator = HoldoutValidator(
            model=model,
            split=split,
            val_metrics=val_metrics,
            rmse_threshold=1.0,
            r2_threshold=-10.0,
            artifact_dir=str(tmp_path),
        )
        validator.validate()
        assert (tmp_path / "pred_vs_actual.png").exists()
        assert (tmp_path / "residual_histogram.png").exists()
