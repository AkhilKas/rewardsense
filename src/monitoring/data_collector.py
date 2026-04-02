"""
Monitoring Data Collector

Reads inference log records from GCS (or local fallback) and aggregates them into structured DataFrames for the monitoring pipeline.

Log path convention (set by inference_logger.py):
    gs://rewardsense-inference-logs/YYYY/MM/DD/<request_id>.json

Usage:
    collector = InferenceDataCollector()
    summary = collector.collect(days=7)
    print(summary.input_features_df.shape)
    print(summary.predictions_df.shape)
    print(summary.latency_df.describe())
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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


# =====================================================================
# Collection result
# =====================================================================


@dataclass
class InferenceDataSummary:
    """Aggregated inference data for a time window."""

    start_date: datetime
    end_date: datetime
    total_records: int = 0
    model_versions: List[str] = field(default_factory=list)

    # Raw records
    records: List[Dict[str, Any]] = field(default_factory=list)

    # Structured DataFrames (built by .to_dataframes())
    input_features_df: Optional[pd.DataFrame] = None
    predictions_df: Optional[pd.DataFrame] = None
    latency_df: Optional[pd.DataFrame] = None
    metadata_df: Optional[pd.DataFrame] = None

    def to_dataframes(self) -> "InferenceDataSummary":
        """Parse raw records into structured DataFrames."""
        if not self.records:
            empty = pd.DataFrame()
            self.input_features_df = empty
            self.predictions_df = empty
            self.latency_df = empty
            self.metadata_df = empty
            return self

        # --- Input features ---
        input_rows = []
        for r in self.records:
            features = r.get("input_features", {})
            row = {
                "request_id": r.get("request_id"),
                "timestamp": r.get("timestamp"),
                "user_hash": r.get("user_hash"),
                "monthly_spend": features.get("monthly_spend"),
                "transaction_history_count": features.get(
                    "transaction_history_count", 0
                ),
            }
            # Flatten spending_categories
            categories = features.get("spending_categories", {})
            for cat, amount in categories.items():
                row[f"spend_{cat}"] = amount
            # Flatten preferred_rewards as binary flags
            for reward in features.get("preferred_rewards", []):
                row[f"pref_{reward}"] = 1
            input_rows.append(row)
        self.input_features_df = pd.DataFrame(input_rows).fillna(0)

        # --- Predictions ---
        pred_rows = []
        for r in self.records:
            scores = r.get("predicted_scores", [])
            for score in scores:
                pred_rows.append(
                    {
                        "request_id": r.get("request_id"),
                        "timestamp": r.get("timestamp"),
                        "card_name": score.get("card_name"),
                        "rank": score.get("rank"),
                        "deterministic_score": score.get("deterministic_score", 0),
                        "personalization_score": score.get("personalization_score", 0),
                        "blended_score": score.get("blended_score", 0),
                    }
                )
        self.predictions_df = pd.DataFrame(pred_rows)

        # --- Latency ---
        latency_rows = []
        for r in self.records:
            breakdown = r.get("latency_breakdown_ms", {})
            latency_rows.append(
                {
                    "request_id": r.get("request_id"),
                    "timestamp": r.get("timestamp"),
                    "normalize_ms": breakdown.get("normalize", 0),
                    "deterministic_ms": breakdown.get("deterministic", 0),
                    "personalization_ms": breakdown.get("personalization", 0),
                    "rank_ms": breakdown.get("rank", 0),
                    "llm_explanation_ms": breakdown.get("llm_explanation", 0),
                    "total_ms": breakdown.get("total", 0),
                }
            )
        self.latency_df = pd.DataFrame(latency_rows)

        # --- Metadata ---
        meta_rows = []
        for r in self.records:
            meta_rows.append(
                {
                    "request_id": r.get("request_id"),
                    "timestamp": r.get("timestamp"),
                    "model_version": r.get("model_version"),
                    "top_card": r.get("top_card"),
                    "is_personalized": r.get("is_personalized", False),
                    "explanation_latency_ms": r.get("explanation_latency_ms"),
                }
            )
        self.metadata_df = pd.DataFrame(meta_rows)

        self.total_records = len(self.records)
        versions = self.metadata_df["model_version"].dropna().unique().tolist()
        self.model_versions = [str(v) for v in versions]

        return self

    @property
    def summary_stats(self) -> Dict[str, Any]:
        """Quick summary for logging and notifications."""
        stats: Dict[str, Any] = {
            "total_records": self.total_records,
            "model_versions": self.model_versions,
            "date_range": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
        }
        if self.latency_df is not None and not self.latency_df.empty:
            stats["latency"] = {
                "p50_ms": round(self.latency_df["total_ms"].median(), 2),
                "p95_ms": round(self.latency_df["total_ms"].quantile(0.95), 2),
                "p99_ms": round(self.latency_df["total_ms"].quantile(0.99), 2),
                "mean_ms": round(self.latency_df["total_ms"].mean(), 2),
            }
        if self.metadata_df is not None and not self.metadata_df.empty:
            top_card_counts = (
                self.metadata_df["top_card"].value_counts().head(5).to_dict()
            )
            stats["top_card_frequency"] = top_card_counts
            stats["personalization_rate"] = round(
                self.metadata_df["is_personalized"].mean(), 4
            )
        return stats


# =====================================================================
# Collector
# =====================================================================


class InferenceDataCollector:
    """Reads and aggregates inference logs from GCS or local filesystem.

    Parameters
    ----------
    bucket : str
        GCS bucket name for inference logs.
    local_dir : str
        Local fallback directory.
    """

    def __init__(
        self,
        bucket: str = GCS_INFERENCE_LOG_BUCKET,
        local_dir: str = LOCAL_LOG_DIR,
    ) -> None:
        self.bucket_name = bucket
        self.local_dir = Path(local_dir)
        self._gcs_client: Optional[Any] = None

        if GCS_AVAILABLE:
            try:
                self._gcs_client = gcs_storage.Client()
            except Exception as e:
                logger.warning("GCS client init failed: %s", e)

    def collect(
        self,
        days: int = 7,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> InferenceDataSummary:
        """Collect inference logs for a time window.

        Parameters
        ----------
        days : int
            Number of days to look back (ignored if start/end provided).
        start_date, end_date : datetime, optional
            Explicit date range. Defaults to last ``days`` days.

        Returns
        -------
        InferenceDataSummary
            Aggregated data with structured DataFrames.
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        summary = InferenceDataSummary(start_date=start_date, end_date=end_date)

        # Generate date prefixes to scan
        date_prefixes = self._generate_date_prefixes(start_date, end_date)

        # Try GCS first, fall back to local
        if self._gcs_client is not None:
            records = self._read_from_gcs(date_prefixes)
        else:
            records = self._read_from_local(date_prefixes)

        summary.records = records
        summary.to_dataframes()

        logger.info(
            "Collected %d inference records (%s to %s)",
            summary.total_records,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        return summary

    def _generate_date_prefixes(self, start: datetime, end: datetime) -> List[str]:
        """Generate YYYY/MM/DD prefixes for each day in the range."""
        prefixes = []
        current = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = end.replace(hour=23, minute=59, second=59)

        while current <= end_day:
            prefixes.append(f"{current.year:04d}/{current.month:02d}/{current.day:02d}")
            current += timedelta(days=1)

        return prefixes

    def _read_from_gcs(self, date_prefixes: List[str]) -> List[Dict[str, Any]]:
        """Read JSON log files from GCS for the given date prefixes."""
        records: List[Dict[str, Any]] = []
        bucket = self._gcs_client.bucket(self.bucket_name)

        for prefix in date_prefixes:
            try:
                blobs = bucket.list_blobs(prefix=f"{prefix}/")
                for blob in blobs:
                    if not blob.name.endswith(".json"):
                        continue
                    try:
                        content = blob.download_as_text()
                        record = json.loads(content)
                        records.append(record)
                    except Exception as e:
                        logger.warning("Failed to parse %s: %s", blob.name, e)
            except Exception as e:
                logger.warning("Failed to list blobs for prefix %s: %s", prefix, e)

        return records

    def _read_from_local(self, date_prefixes: List[str]) -> List[Dict[str, Any]]:
        """Read JSON log files from local filesystem."""
        records: List[Dict[str, Any]] = []

        for prefix in date_prefixes:
            directory = self.local_dir / prefix
            if not directory.exists():
                continue
            for filepath in directory.glob("*.json"):
                try:
                    content = filepath.read_text(encoding="utf-8")
                    record = json.loads(content)
                    records.append(record)
                except Exception as e:
                    logger.warning("Failed to parse %s: %s", filepath, e)

        return records
