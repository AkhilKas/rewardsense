"""
Model loader for the RewardSense serving service (Story 1.3).

On container startup (call ``load_model()`` once):
  1. Connects to the MLflow tracking server at MLFLOW_TRACKING_URI.
  2. Queries the Model Registry for the latest version in MODEL_STAGE
     (default: "Production") of REGISTERED_MODEL_NAME (default: "personalization").
  3. Downloads and loads the sklearn model artifact into memory.
  4. Wraps it in a PersonalizedScorer and caches as a module-level singleton.

Fail-fast contract:
  - If MLflow is unreachable, the process exits with code 1.
  - If no model exists in the requested stage, the process exits with code 1.
  - If the artifact cannot be loaded, the process exits with code 1.

Public API:
  load_model()         — call once at startup; exits on failure.
  get_model()          — returns the cached PersonalizedScorer; raises if not loaded.
  get_model_version()  — returns the cached version string (e.g. "3"), or None.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Lazy top-level imports — kept at module scope so tests can patch them
# ---------------------------------------------------------------------------
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False

try:
    from model_pipeline.personalization.personalized_scorer import PersonalizedScorer

    SCORER_AVAILABLE = True
except ImportError:  # pragma: no cover
    PersonalizedScorer = None  # type: ignore[assignment]
    SCORER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration (all overrideable via environment variables)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME: str = os.getenv("REGISTERED_MODEL_NAME", "personalization")
MODEL_STAGE: str = os.getenv("MODEL_STAGE", "Production")
MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache"))

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_scorer: Optional[Any] = None
_model_version: Optional[str] = None


def load_model() -> None:
    """Load the Production model from MLflow at container startup.

    Populates the module-level ``_scorer`` and ``_model_version`` singletons.
    Calls ``sys.exit(1)`` with a descriptive log on any failure so Cloud Run
    treats the revision as unhealthy and rolls back.
    """
    global _scorer, _model_version

    # ---- Check MLflow is available ----
    if not MLFLOW_AVAILABLE:
        logger.error("mlflow is not installed. Add it to requirements-serving.txt.")
        sys.exit(1)

    logger.info(
        "Connecting to MLflow at '{}' — loading model '{}' (stage={}).",
        MLFLOW_TRACKING_URI,
        REGISTERED_MODEL_NAME,
        MODEL_STAGE,
    )

    # ---- Connect ----
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(MLFLOW_TRACKING_URI)
    except Exception as exc:
        logger.error(
            "Failed to connect to MLflow tracking server at '{}': {}",
            MLFLOW_TRACKING_URI,
            exc,
        )
        sys.exit(1)

    # ---- Find latest Production version ----
    try:
        versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME, stages=[MODEL_STAGE]
        )
    except Exception as exc:
        logger.error(
            "MLflow registry query failed for model '{}': {}. "
            "Check that the MLflow server is reachable and the model is registered.",
            REGISTERED_MODEL_NAME,
            exc,
        )
        sys.exit(1)

    if not versions:
        logger.error(
            "No '{}' model found in stage '{}'. "
            "Run the model pipeline DAG to train and promote a Production model.",
            REGISTERED_MODEL_NAME,
            MODEL_STAGE,
        )
        sys.exit(1)

    version_info = versions[0]
    version_number = version_info.version
    run_id = version_info.run_id
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"

    logger.info(
        "Found {} model '{}' version {} (run_id={}).",
        MODEL_STAGE,
        REGISTERED_MODEL_NAME,
        version_number,
        run_id,
    )

    # ---- Download and load artifact ----
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        raw_model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        logger.error(
            "Failed to load model artifact from MLflow (uri='{}'): {}",
            model_uri,
            exc,
        )
        sys.exit(1)

    # ---- Wrap in PersonalizedScorer ----
    try:
        scorer = PersonalizedScorer(model=raw_model)
    except Exception as exc:
        logger.error(
            "Failed to initialise PersonalizedScorer with loaded model: {}", exc
        )
        sys.exit(1)

    _scorer = scorer
    _model_version = str(version_number)

    logger.info(
        "Model '{}' version {} loaded and ready.",
        REGISTERED_MODEL_NAME,
        _model_version,
    )


def get_model() -> Any:
    """Return the cached PersonalizedScorer.

    Raises
    ------
    RuntimeError
        If ``load_model()`` has not been called yet.
    """
    if _scorer is None:
        raise RuntimeError(
            "Model has not been loaded. Call load_model() at container startup."
        )
    return _scorer


def get_model_version() -> Optional[str]:
    """Return the cached MLflow model version string, or None if not loaded."""
    return _model_version
