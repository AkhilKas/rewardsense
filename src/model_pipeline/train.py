"""
Model training entrypoint.

Executes the ML personalization pipeline:
1. Data combination
2. Splitting
3. Hyperparameter sweeps
4. Bias Detection
5. Artifacts emission
"""

import sys
import json
import shutil
from pathlib import Path

from loguru import logger
import joblib
import numpy as np

from model_pipeline.personalization.dataset_builder import DatasetBuilder
from model_pipeline.personalization.splits import split_data
from model_pipeline.personalization.trainer import Trainer
from model_pipeline.bias.model_bias_detector import ModelBiasDetector
from model_pipeline.personalization.evaluation import compute_ranking_metrics


def main():
    logger.info("Starting model training pipeline")

    # 1. Load data via DataPipelineLoader inside DatasetBuilder
    try:
        builder = DatasetBuilder()
        X, y, df = builder.build()
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        sys.exit(1)

    # 2. Split data
    split = split_data(X, y, meta=df)

    # 3. Train model
    trainer = Trainer(split=split)
    result = trainer.train_all()

    if not result.best_record:
        logger.error("No models trained successfully.")
        sys.exit(1)

    best = result.best_record
    logger.info(f"Selected best model: {best.model_name}")

    # 4. Evaluate and get real metrics
    val_pred = best.model.predict(split.X_val)
    ranking_10 = compute_ranking_metrics(split.y_val.values, val_pred, k=10)

    metrics = {
        "rmse": best.val_metrics.rmse,
        "mae": best.val_metrics.mae,
        "r2": best.val_metrics.r2,
        "ndcg@10": ranking_10.ndcg_at_k,
        "run_id": best.mlflow_run_id or "local_test",
    }

    if best.val_report.ranking:
        metrics.update(best.val_report.ranking.to_dict())

    # 5. Run bias detection and get real report
    detector = ModelBiasDetector()
    sensitive_cols = [
        c
        for c in ["age_group", "budget_quartile", "archetype"]
        if c in split.meta_val.columns
    ]
    sensitive_feats = (
        split.meta_val[sensitive_cols] if sensitive_cols else split.meta_val
    )

    bias_report_obj = detector.detect(
        y_true=split.y_val.values,
        y_pred=val_pred,
        sensitive_features=sensitive_feats,
        model_name=best.model_name,
    )
    bias_report = bias_report_obj.to_dict()

    # 6. Save outputs to /tmp/model_pipeline/ for the DAG quality gates
    out_dir = Path("/tmp/model_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _numpy_safe(obj):
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, default=_numpy_safe)

    with open(out_dir / "bias_report.json", "w") as f:
        json.dump(bias_report, f, default=_numpy_safe)

    artifact_dir = out_dir / "model_artifact"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    # 7. Write model artifact locally
    joblib.dump(best.model, artifact_dir / "model.joblib")

    logger.info(f"Integration artifacts written to {out_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
