"""
Training orchestrator for the personalization model.

Responsibilities:
- Train each candidate model from the model factory
- Evaluate on the validation set
- Log parameters, metrics, and artifacts to MLflow
- Select the best model by RMSE
- Persist the best model artifact
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from loguru import logger

from src.model_pipeline.personalization.evaluation import (
    EvaluationReport,
    RegressionMetrics,
    compute_regression_metrics,
    evaluate,
)
from src.model_pipeline.personalization.models import create_model
from src.model_pipeline.personalization.splits import SplitResult


@dataclass
class TrainedModelRecord:
    """Metadata for one trained model run."""

    model_name: str
    model: Any
    train_metrics: RegressionMetrics
    val_metrics: RegressionMetrics
    val_report: EvaluationReport
    params: Dict[str, Any]
    train_time_s: float
    mlflow_run_id: Optional[str] = None


@dataclass
class TrainingResult:
    """Result of the full training pipeline across all candidates."""

    records: List[TrainedModelRecord] = field(default_factory=list)
    best_model_name: Optional[str] = None
    best_model: Any = None
    best_record: Optional[TrainedModelRecord] = None

    @property
    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append(
                {
                    "model": r.model_name,
                    "train_rmse": r.train_metrics.rmse,
                    "val_rmse": r.val_metrics.rmse,
                    "val_mae": r.val_metrics.mae,
                    "val_r2": r.val_metrics.r2,
                    "train_time_s": r.train_time_s,
                }
            )
        return pd.DataFrame(rows).sort_values("val_rmse")


class Trainer:
    """Orchestrate training, evaluation, and selection of candidate models.

    Parameters
    ----------
    split : SplitResult
        Pre-split data.
    model_names : list of str or None
        Which models to train. Defaults to ``["mean", "random_forest", "xgboost"]``.
    model_params : dict or None
        Optional per-model hyperparameters: ``{"xgboost": {"max_depth": 4}, ...}``.
    experiment_name : str
        MLflow experiment name.
    artifact_dir : str or Path
        Where to save model artifacts locally.
    use_mlflow : bool
        Whether to log to MLflow (can be disabled for testing).
    """

    def __init__(
        self,
        split: SplitResult,
        model_names: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        experiment_name: str = "personalization-point-valuation",
        artifact_dir: str = "models/personalization",
        use_mlflow: bool = True,
    ) -> None:
        self.split = split
        self.model_names = model_names or ["mean", "random_forest", "xgboost"]
        self.model_params = model_params or {}
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow

    def train_all(self) -> TrainingResult:
        """Train all candidate models and return the full result."""
        result = TrainingResult()

        for name in self.model_names:
            logger.info("Training model: {}", name)
            record = self._train_single(name)
            result.records.append(record)

        if result.records:
            best = min(result.records, key=lambda r: r.val_metrics.rmse)
            result.best_model_name = best.model_name
            result.best_model = best.model
            result.best_record = best

            logger.info(
                "Best model: {} (val RMSE={:.6f})",
                best.model_name,
                best.val_metrics.rmse,
            )

            self._save_model(best)

        logger.info("\n{}", result.leaderboard.to_string(index=False))
        return result

    def _train_single(self, model_name: str) -> TrainedModelRecord:
        """Train a single model, evaluate, and optionally log to MLflow."""
        params = self.model_params.get(model_name, {})
        model = create_model(model_name, params)

        t0 = time.time()
        model.fit(self.split.X_train, self.split.y_train)
        train_time = time.time() - t0

        train_pred = model.predict(self.split.X_train)
        val_pred = model.predict(self.split.X_val)

        train_metrics = compute_regression_metrics(
            self.split.y_train.values, train_pred
        )
        val_report = evaluate(
            self.split.y_val,
            val_pred,
            meta=self.split.meta_val,
            train_metrics=train_metrics,
        )

        actual_params = model.get_params() if hasattr(model, "get_params") else params

        record = TrainedModelRecord(
            model_name=model_name,
            model=model,
            train_metrics=train_metrics,
            val_metrics=val_report.overall,
            val_report=val_report,
            params=actual_params,
            train_time_s=round(train_time, 3),
        )

        if self.use_mlflow:
            record.mlflow_run_id = self._log_to_mlflow(record)

        return record

    def _log_to_mlflow(self, record: TrainedModelRecord) -> Optional[str]:
        """Log a training run to MLflow. Returns the run_id or None."""
        try:
            import mlflow

            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(run_name=record.model_name) as run:
                safe_params = {
                    k: v
                    for k, v in record.params.items()
                    if isinstance(v, (int, float, str, bool))
                }
                mlflow.log_params(safe_params)
                mlflow.log_metrics(
                    {
                        "train_rmse": record.train_metrics.rmse,
                        "train_mae": record.train_metrics.mae,
                        "train_r2": record.train_metrics.r2,
                        "val_rmse": record.val_metrics.rmse,
                        "val_mae": record.val_metrics.mae,
                        "val_r2": record.val_metrics.r2,
                        "train_time_s": record.train_time_s,
                    }
                )

                report_path = self.artifact_dir / f"{record.model_name}_report.json"
                report_path.write_text(
                    json.dumps(record.val_report.to_dict(), indent=2)
                )
                mlflow.log_artifact(str(report_path))

                return run.info.run_id
        except Exception as exc:
            logger.warning("MLflow logging failed for {}: {}", record.model_name, exc)
            return None

    def _save_model(self, record: TrainedModelRecord) -> Path:
        """Persist the best model to disk via joblib."""
        path = self.artifact_dir / f"best_model_{record.model_name}.joblib"
        joblib.dump(record.model, path)
        logger.info("Saved best model to {}", path)
        return path
