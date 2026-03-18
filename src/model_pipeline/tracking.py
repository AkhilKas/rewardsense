"""
MLflow Experiment Tracking Wrapper.

Shared MLflow client with helper functions for logging
params, metrics, artifacts, and managing experiment namespaces.

Usage:
    from src.model_pipeline.tracking import RewardSenseTracker

    tracker = RewardSenseTracker(experiment="personalization-model")
    with tracker.start_run(run_name="xgboost-v1") as run:
        tracker.log_params({"lr": 0.01, "max_depth": 6})
        tracker.log_metrics({"ndcg_5": 0.82, "map_5": 0.78})
        tracker.log_artifact("model.pkl", artifact_path="models")
        tracker.log_dict({"feature_importance": {...}}, "feature_importance.json")
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MLflow import — graceful fallback when mlflow is not installed
# ---------------------------------------------------------------------------
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment,misc]
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Experiment namespaces
# ---------------------------------------------------------------------------
EXPERIMENT_NAMESPACES: Dict[str, str] = {
    "reward-scoring": "Deterministic reward scoring engine experiments",
    "personalization-model": "ML personalization model training & tuning",
    "llm-explainability": "LLM explanation generation experiments",
}

# ---------------------------------------------------------------------------
# Default tracking URI (override via env var or constructor)
# ---------------------------------------------------------------------------
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


class RewardSenseTracker:
    """
    Shared MLflow tracking client for RewardSense model pipeline.

    Wraps MLflow's API with project-specific conventions:
    - Auto-creates experiment namespaces on first use
    - Provides typed helpers for common logging patterns
    - Supports context-manager runs (`with tracker.start_run(...)`)
    - Gracefully degrades if MLflow is unavailable (logs warnings)

    Parameters
    ----------
    experiment : str
        One of the predefined experiment namespaces, or a custom name.
    tracking_uri : str, optional
        MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env
        var or ``http://localhost:5000``.
    """

    def __init__(
        self,
        experiment: str = "personalization-model",
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        self._active_run: Optional[Any] = None
        self._client: Optional[Any] = None

        if not MLFLOW_AVAILABLE:
            logger.warning(
                "mlflow not installed — tracking calls will be no-ops. "
                "Install with: pip install mlflow"
            )
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        self._client = MlflowClient(self.tracking_uri)
        self._ensure_experiment_exists(self.experiment_name)

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def _ensure_experiment_exists(self, name: str) -> str:
        """Create experiment if it doesn't exist. Return experiment ID."""
        if not MLFLOW_AVAILABLE:
            return "0"

        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            desc = EXPERIMENT_NAMESPACES.get(name, f"RewardSense experiment: {name}")
            exp_id = mlflow.create_experiment(name, tags={"description": desc})
            logger.info("Created MLflow experiment '%s' (id=%s)", name, exp_id)
            return exp_id
        return exp.experiment_id

    def create_all_namespaces(self) -> Dict[str, str]:
        """Create all predefined experiment namespaces. Returns {name: id}."""
        results = {}
        for name in EXPERIMENT_NAMESPACES:
            results[name] = self._ensure_experiment_exists(name)
        return results

    def list_experiments(self) -> List[Dict[str, str]]:
        """List all experiments on the tracking server."""
        if not MLFLOW_AVAILABLE or self._client is None:
            return []
        exps = self._client.search_experiments()
        return [
            {"name": e.name, "id": e.experiment_id, "lifecycle": e.lifecycle_stage}
            for e in exps
        ]

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """Context manager for an MLflow run.

        Yields the active run object (or None if MLflow unavailable).
        Automatically ends the run on exit.
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow unavailable — run '%s' is a no-op", run_name)
            yield None
            return

        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
        self._active_run = run
        try:
            yield run
        finally:
            mlflow.end_run()
            self._active_run = None

    @property
    def active_run_id(self) -> Optional[str]:
        """Return the active run ID, or None."""
        if self._active_run is not None:
            return self._active_run.info.run_id
        return None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dict of parameters to the active run."""
        if not MLFLOW_AVAILABLE:
            return
        # MLflow params must be strings ≤ 500 chars
        sanitized = {k: str(v)[:500] for k, v in params.items()}
        mlflow.log_params(sanitized)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log a dict of metrics to the active run."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_metric(key, value, step=step)

    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log a local file as an artifact."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_artifacts(
        self, local_dir: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log all files in a directory as artifacts."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """Serialize a dict to JSON and log as an artifact."""
        if not MLFLOW_AVAILABLE:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            tmp_path = f.name
        try:
            mlflow.log_artifact(tmp_path)
        finally:
            os.unlink(tmp_path)

    def log_figure(self, fig: Any, filename: str) -> None:
        """Log a matplotlib figure as an artifact."""
        if not MLFLOW_AVAILABLE:
            return
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, dpi=150, bbox_inches="tight")
            tmp_path = f.name
        try:
            mlflow.log_artifact(tmp_path)
        finally:
            os.unlink(tmp_path)

    def log_model(self, model: Any, artifact_path: str = "model", **kwargs) -> None:
        """Log a model artifact. Auto-detects sklearn/xgboost/pytorch."""
        if not MLFLOW_AVAILABLE:
            return

        model_type = type(model).__module__.split(".")[0]

        if model_type in ("sklearn", "xgboost", "lightgbm"):
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        elif model_type == "torch":
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
        else:
            # Fallback: pickle and log as artifact
            import pickle

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(model, f)
                tmp_path = f.name
            try:
                mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
            finally:
                os.unlink(tmp_path)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the active run."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.set_tags(tags)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_best_run(
        self,
        metric: str = "ndcg_5",
        order: str = "DESC",
        max_results: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Get the best run from the current experiment by a metric."""
        if not MLFLOW_AVAILABLE or self._client is None:
            return None

        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            return None

        runs = self._client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=max_results,
        )
        if not runs:
            return None

        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "params": dict(best.data.params),
            "metrics": dict(best.data.metrics),
            "tags": dict(best.data.tags),
            "status": best.info.status,
        }

    def compare_runs(
        self,
        metric: str = "ndcg_5",
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top runs for comparison."""
        if not MLFLOW_AVAILABLE or self._client is None:
            return []

        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            return []

        runs = self._client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=max_results,
        )
        return [
            {
                "run_id": r.info.run_id,
                "run_name": r.data.tags.get("mlflow.runName", ""),
                "params": dict(r.data.params),
                "metrics": dict(r.data.metrics),
            }
            for r in runs
        ]
