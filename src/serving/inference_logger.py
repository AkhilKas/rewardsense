"""
Inference logging for monitoring.

Asynchronously writes JSON log records to a GCS bucket on every
``/predict`` call.  Records are date-partitioned under
``gs://<bucket>/YYYY/MM/DD/<request_id>.json`` so the monitoring
pipeline can query them by date range.

The logger is designed to be used as a **FastAPI BackgroundTask** so
that it adds negligible latency to the response path.

When GCS is unavailable (e.g. local development), the logger falls
back to writing records into a local directory (``LOCAL_LOG_DIR``
environment variable, default ``/tmp/rewardsense-inference-logs``).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GCS_INFERENCE_LOG_BUCKET: str = os.getenv(
    "INFERENCE_LOG_BUCKET", "rewardsense-inference-logs"
)
LOCAL_LOG_DIR: str = os.getenv(
    "LOCAL_INFERENCE_LOG_DIR", "/tmp/rewardsense-inference-logs"
)

# ---------------------------------------------------------------------------
# Lazy GCS import
# ---------------------------------------------------------------------------
try:
    from google.cloud import storage as gcs_storage

    GCS_AVAILABLE = True
except ImportError:
    gcs_storage = None  # type: ignore[assignment]
    GCS_AVAILABLE = False

# Re-usable GCS client (module-level singleton to avoid per-request overhead)
_gcs_client: Optional[Any] = None


def _get_gcs_client() -> Any:
    """Return a cached ``google.cloud.storage.Client``, or *None*."""
    global _gcs_client
    if _gcs_client is not None:
        return _gcs_client
    if not GCS_AVAILABLE:
        return None
    try:
        _gcs_client = gcs_storage.Client()
        return _gcs_client
    except Exception as exc:
        logger.warning("Failed to initialise GCS client: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Log record builder
# ---------------------------------------------------------------------------


def build_log_record(
    *,
    request_id: str,
    user_hash: str,
    input_features: Dict[str, Any],
    scores: List[Dict[str, Any]],
    top_card: str,
    model_version: str,
    latency_breakdown: Dict[str, float],
    is_personalized: bool,
    explanation_latency_ms: Optional[float] = None,
    # --- Story 5.1 extensions (all optional for backward compat) ---
    recommendation_flow: str = "predict",
    request_status: str = "success",
    llm_telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a structured JSON log record for one inference request.

    All personally-identifiable information should already be hashed/anonymised
    by the caller (``user_hash`` is a truncated SHA-256 of ``user_id``).

    Story 5.1 additions
    --------------------
    recommendation_flow : str
        Which endpoint produced this record.  One of ``"predict"``,
        ``"portfolio"``, ``"transaction"``.
    request_status : str
        ``"success"`` or ``"error"``.
    llm_telemetry : dict, optional
        LLM-specific metrics for business reporting::

            {
                "llm_calls": int,
                "llm_successes": int,
                "llm_fallbacks": int,
                "llm_model_name": str | None,
                "llm_token_estimate": int | None,
                "llm_cost_estimate_usd": float | None,
                "llm_prompt_version": str | None,
            }
    """
    now = datetime.now(timezone.utc)
    record: Dict[str, Any] = {
        "timestamp": now.isoformat(),
        "request_id": request_id,
        "user_hash": user_hash,
        "input_features": input_features,
        "predicted_scores": scores,
        "top_card": top_card,
        "model_version": model_version,
        "latency_breakdown_ms": latency_breakdown,
        "is_personalized": is_personalized,
        # Story 5.1 fields
        "recommendation_flow": recommendation_flow,
        "request_status": request_status,
    }
    if explanation_latency_ms is not None:
        record["explanation_latency_ms"] = round(explanation_latency_ms, 3)
    if llm_telemetry is not None:
        record["llm_telemetry"] = llm_telemetry
    return record


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _gcs_date_prefix() -> str:
    """Return ``YYYY/MM/DD`` for the current UTC date."""
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}/{now.month:02d}/{now.day:02d}"


def _write_to_gcs(record: Dict[str, Any]) -> bool:
    """Upload *record* as a JSON blob to GCS.  Returns ``True`` on success."""
    client = _get_gcs_client()
    if client is None:
        return False

    prefix = _gcs_date_prefix()
    blob_name = f"{prefix}/{record['request_id']}.json"

    try:
        bucket = client.bucket(GCS_INFERENCE_LOG_BUCKET)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(record, default=str),
            content_type="application/json",
        )
        logger.debug(
            "Inference log written to gs://%s/%s", GCS_INFERENCE_LOG_BUCKET, blob_name
        )
        return True
    except Exception as exc:
        logger.warning(
            "GCS inference log write failed (gs://%s/%s): %s",
            GCS_INFERENCE_LOG_BUCKET,
            blob_name,
            exc,
        )
        return False


def _write_to_local(record: Dict[str, Any]) -> bool:
    """Write *record* to local filesystem as fallback.  Returns ``True`` on success."""
    prefix = _gcs_date_prefix()
    directory = Path(LOCAL_LOG_DIR) / prefix
    try:
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / f"{record['request_id']}.json"
        filepath.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
        logger.debug("Inference log written to %s", filepath)
        return True
    except Exception as exc:
        logger.warning("Local inference log write failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API -- intended to be called inside BackgroundTasks
# ---------------------------------------------------------------------------


def log_inference(record: Dict[str, Any]) -> None:
    """Persist *record* to GCS (preferred) or local filesystem (fallback).

    This function is **synchronous** and intended to be dispatched via
    ``fastapi.BackgroundTasks.add_task(log_inference, record)`` so it
    runs after the HTTP response has been sent.
    """
    if _write_to_gcs(record):
        return
    _write_to_local(record)
