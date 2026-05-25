"""
Model loader for the RewardSense serving service (Story 1.3).

Loading priority:
  1. If MODEL_LOCAL_PATH is set and the file exists, load directly from that
     .joblib file — no MLflow connection required.
  2. Otherwise connect to the MLflow tracking server at MLFLOW_TRACKING_URI,
     query the Model Registry for the latest version in MODEL_STAGE, and
     download the artifact.

Failure behaviour:
  - If MODEL_LOAD_REQUIRED=true (default), any failure calls sys.exit(1).
  - If MODEL_LOAD_REQUIRED=false, failures are logged as warnings and the
    service starts in deterministic-only mode (no personalisation).

Public API:
  load_model()         — call once at startup; exits on failure unless
                         MODEL_LOAD_REQUIRED=false.
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

# Local file path — set this to skip MLflow entirely (e.g. for portfolio hosting).
MODEL_LOCAL_PATH: Optional[str] = os.getenv("MODEL_LOCAL_PATH")

# Set to "false" to start in deterministic-only mode when no model is available.
MODEL_LOAD_REQUIRED: bool = os.getenv("MODEL_LOAD_REQUIRED", "true").lower() not in (
    "0",
    "false",
    "no",
)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_scorer: Optional[Any] = None
_model_version: Optional[str] = None


def _fail(message: str) -> None:
    """Exit or warn depending on MODEL_LOAD_REQUIRED."""
    if MODEL_LOAD_REQUIRED:
        logger.error(message)
        sys.exit(1)
    else:
        logger.warning(
            "{} — starting in deterministic-only mode (MODEL_LOAD_REQUIRED=false).",
            message,
        )


def _load_from_local(path: Path) -> None:
    """Load a .joblib model artifact directly from the local filesystem."""
    global _scorer, _model_version

    try:
        import joblib
    except ImportError:
        _fail("joblib is not installed. Add it to requirements-serving.txt.")
        return

    if not path.exists():
        _fail(f"MODEL_LOCAL_PATH '{path}' does not exist.")
        return

    logger.info("Loading model from local path '{}'.", path)

    try:
        raw_model = joblib.load(path)
    except Exception as exc:
        _fail(f"Failed to load model artifact from '{path}': {exc}")
        return

    try:
        scorer = PersonalizedScorer(model=raw_model)
    except Exception as exc:
        _fail(f"Failed to initialise PersonalizedScorer with loaded model: {exc}")
        return

    _scorer = scorer
    _model_version = "local"
    logger.info("Model loaded from local path '{}' and ready.", path)


def _load_from_mlflow() -> None:
    """Load the Production model from the MLflow registry."""
    global _scorer, _model_version

    if not MLFLOW_AVAILABLE:
        _fail("mlflow is not installed. Add it to requirements-serving.txt.")
        return

    logger.info(
        "Connecting to MLflow at '{}' — loading model '{}' (stage={}).",
        MLFLOW_TRACKING_URI,
        REGISTERED_MODEL_NAME,
        MODEL_STAGE,
    )

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(MLFLOW_TRACKING_URI)
    except Exception as exc:
        _fail(
            f"Failed to connect to MLflow tracking server at '{MLFLOW_TRACKING_URI}': {exc}"
        )
        return

    try:
        versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME, stages=[MODEL_STAGE]
        )
    except Exception as exc:
        _fail(
            f"MLflow registry query failed for model '{REGISTERED_MODEL_NAME}': {exc}"
        )
        return

    if not versions:
        _fail(
            f"No '{REGISTERED_MODEL_NAME}' model found in stage '{MODEL_STAGE}'. "
            "Run the model pipeline DAG to train and promote a Production model."
        )
        return

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

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        raw_model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        _fail(f"Failed to load model artifact from MLflow (uri='{model_uri}'): {exc}")
        return

    try:
        scorer = PersonalizedScorer(model=raw_model)
    except Exception as exc:
        _fail(f"Failed to initialise PersonalizedScorer with loaded model: {exc}")
        return

    _scorer = scorer
    _model_version = str(version_number)
    logger.info(
        "Model '{}' version {} loaded and ready.",
        REGISTERED_MODEL_NAME,
        _model_version,
    )


def load_model() -> None:
    """Load the model at container startup.

    Tries MODEL_LOCAL_PATH first, then falls back to MLflow.
    Behaviour on failure is controlled by MODEL_LOAD_REQUIRED.
    """
    if MODEL_LOCAL_PATH:
        _load_from_local(Path(MODEL_LOCAL_PATH))
    else:
        _load_from_mlflow()


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
