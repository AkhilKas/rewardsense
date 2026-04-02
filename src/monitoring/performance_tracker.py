"""
Performance Metrics Tracker

- Tracks model performance and serving health metrics over time.
- Computes latency percentiles, error rates, prediction confidence, and score variance from inference logs.

Stores daily JSON snapshots to GCS for trend analysis.

Usage:
    tracker = PerformanceTracker()
    snapshot = tracker.compute(summary=collector_summary)
    print(snapshot.to_dict())
    tracker.save_snapshot(snapshot)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy GCS
# ---------------------------------------------------------------------------
try:
    from google.cloud import storage as gcs_storage

    GCS_AVAILABLE = True
except ImportError:
    gcs_storage = None  # type: ignore[assignment]
    GCS_AVAILABLE = False

GCS_MONITORING_BUCKET: str = os.getenv("MONITORING_BUCKET", "rewardsense-monitoring")
PERFORMANCE_GCS_PREFIX: str = os.getenv("PERFORMANCE_PREFIX", "performance-snapshots")
LOCAL_SNAPSHOT_DIR: str = os.getenv("LOCAL_SNAPSHOT_DIR", "data/monitoring/performance")

# Alerting thresholds
LATENCY_P95_THRESHOLD_MS: float = float(os.getenv("LATENCY_P95_THRESHOLD_MS", "10000"))
ERROR_RATE_THRESHOLD: float = float(os.getenv("ERROR_RATE_THRESHOLD", "0.05"))


# =====================================================================
# Snapshot
# =====================================================================


@dataclass
class PerformanceSnapshot:
    """Daily performance metrics snapshot."""

    timestamp: str
    date: str
    total_requests: int = 0
    model_versions: List[str] = field(default_factory=list)

    # Latency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0

    # Per-stage latency
    stage_latency: Dict[str, float] = field(default_factory=dict)

    # Prediction metrics
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    top_card_entropy: float = 0.0  # diversity of recommendations

    # Personalization
    personalization_rate: float = 0.0

    # Top card frequency
    top_card_distribution: Dict[str, int] = field(default_factory=dict)

    # Alerts
    alerts: List[str] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "date": self.date,
            "total_requests": self.total_requests,
            "model_versions": self.model_versions,
            "latency": {
                "p50_ms": round(self.latency_p50_ms, 2),
                "p95_ms": round(self.latency_p95_ms, 2),
                "p99_ms": round(self.latency_p99_ms, 2),
                "mean_ms": round(self.latency_mean_ms, 2),
                "std_ms": round(self.latency_std_ms, 2),
            },
            "stage_latency": {k: round(v, 2) for k, v in self.stage_latency.items()},
            "prediction_metrics": {
                "score_mean": round(self.score_mean, 4),
                "score_std": round(self.score_std, 4),
                "score_min": round(self.score_min, 4),
                "score_max": round(self.score_max, 4),
                "top_card_entropy": round(self.top_card_entropy, 4),
            },
            "personalization_rate": round(self.personalization_rate, 4),
            "top_card_distribution": self.top_card_distribution,
            "alerts": self.alerts,
        }


# =====================================================================
# Tracker
# =====================================================================


class PerformanceTracker:
    """Compute and store daily performance snapshots.

    Parameters
    ----------
    output_dir : str or Path
        Local directory for snapshot storage.
    latency_threshold_ms : float
        p95 latency threshold for alerting.
    error_rate_threshold : float
        Error rate threshold for alerting.
    """

    def __init__(
        self,
        output_dir: str | Path = LOCAL_SNAPSHOT_DIR,
        latency_threshold_ms: float = LATENCY_P95_THRESHOLD_MS,
        error_rate_threshold: float = ERROR_RATE_THRESHOLD,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latency_threshold = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold

    def compute(self, summary: Any) -> PerformanceSnapshot:
        """Compute performance snapshot from an InferenceDataSummary.

        Parameters
        ----------
        summary : InferenceDataSummary
            Output from InferenceDataCollector.collect().

        Returns
        -------
        PerformanceSnapshot
        """
        ts = datetime.now(timezone.utc)
        snapshot = PerformanceSnapshot(
            timestamp=ts.isoformat(),
            date=ts.strftime("%Y-%m-%d"),
            total_requests=summary.total_records,
            model_versions=summary.model_versions,
        )

        # --- Latency metrics ---
        if summary.latency_df is not None and not summary.latency_df.empty:
            total_ms = summary.latency_df["total_ms"]
            snapshot.latency_p50_ms = float(total_ms.median())
            snapshot.latency_p95_ms = float(total_ms.quantile(0.95))
            snapshot.latency_p99_ms = float(total_ms.quantile(0.99))
            snapshot.latency_mean_ms = float(total_ms.mean())
            snapshot.latency_std_ms = float(total_ms.std())

            # Per-stage averages
            stage_cols = [
                c
                for c in summary.latency_df.columns
                if c.endswith("_ms") and c != "total_ms"
            ]
            for col in stage_cols:
                stage_name = col.replace("_ms", "")
                snapshot.stage_latency[stage_name] = float(
                    summary.latency_df[col].mean()
                )

            # Latency alert
            if snapshot.latency_p95_ms > self.latency_threshold:
                snapshot.alerts.append(
                    f"LATENCY: p95={snapshot.latency_p95_ms:.0f}ms "
                    f"exceeds threshold {self.latency_threshold:.0f}ms"
                )

        # --- Prediction metrics ---
        if summary.predictions_df is not None and not summary.predictions_df.empty:
            scores = summary.predictions_df["blended_score"]
            snapshot.score_mean = float(scores.mean())
            snapshot.score_std = float(scores.std())
            snapshot.score_min = float(scores.min())
            snapshot.score_max = float(scores.max())

        # --- Metadata metrics ---
        if summary.metadata_df is not None and not summary.metadata_df.empty:
            snapshot.personalization_rate = float(
                summary.metadata_df["is_personalized"].mean()
            )

            # Top card distribution and entropy
            top_cards = summary.metadata_df["top_card"].value_counts()
            snapshot.top_card_distribution = top_cards.to_dict()

            # Shannon entropy of top card distribution
            if len(top_cards) > 0:
                total = top_cards.sum()
                probs = top_cards.values / total
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                snapshot.top_card_entropy = float(entropy)

        logger.info(
            "Performance snapshot: %d requests, p95=%.0fms, %d alerts",
            snapshot.total_requests,
            snapshot.latency_p95_ms,
            len(snapshot.alerts),
        )

        return snapshot

    def save_snapshot(self, snapshot: PerformanceSnapshot) -> Path:
        """Save snapshot to local filesystem and GCS."""
        filename = f"performance_{snapshot.date}.json"
        local_path = self.output_dir / filename
        local_path.write_text(json.dumps(snapshot.to_dict(), indent=2, default=str))
        logger.info("Saved performance snapshot: %s", local_path)

        # Upload to GCS
        if GCS_AVAILABLE:
            try:
                client = gcs_storage.Client()
                bucket = client.bucket(GCS_MONITORING_BUCKET)
                blob = bucket.blob(f"{PERFORMANCE_GCS_PREFIX}/{filename}")
                blob.upload_from_filename(str(local_path))
                logger.info("Uploaded performance snapshot to GCS")
            except Exception as e:
                logger.warning("GCS upload failed: %s", e)

        return local_path

    def load_history(self, days: int = 30) -> List[PerformanceSnapshot]:
        """Load historical snapshots from local directory."""
        snapshots = []
        for filepath in sorted(self.output_dir.glob("performance_*.json")):
            try:
                data = json.loads(filepath.read_text())
                snapshots.append(
                    PerformanceSnapshot(
                        timestamp=data.get("timestamp", ""),
                        date=data.get("date", ""),
                        total_requests=data.get("total_requests", 0),
                        model_versions=data.get("model_versions", []),
                        latency_p50_ms=data.get("latency", {}).get("p50_ms", 0),
                        latency_p95_ms=data.get("latency", {}).get("p95_ms", 0),
                        latency_p99_ms=data.get("latency", {}).get("p99_ms", 0),
                        latency_mean_ms=data.get("latency", {}).get("mean_ms", 0),
                        personalization_rate=data.get("personalization_rate", 0),
                        alerts=data.get("alerts", []),
                    )
                )
            except Exception as e:
                logger.warning("Failed to load snapshot %s: %s", filepath, e)
        return snapshots[-days:]
