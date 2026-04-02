"""
AI Drift Detection

Compares recent inference data against a reference (training) distribution. Produces HTML reports for the dashboard and JSON for programmatic threshold checking.

Reference profile is stored during model training at:
    gs://rewardsense-monitoring/reference/training_reference.csv

Usage:
    detector = DriftDetector(reference_path="data/reference/training_reference.csv")
    result = detector.detect(current_df=collector.input_features_df)
    print(result.drift_detected)
    print(result.drifted_features)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Evidently import
# ---------------------------------------------------------------------------
try:
    from evidently import ColumnMapping
    from evidently import Report
    from evidently.presets import DataDriftPreset

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning(
        "evidently not installed — drift detection disabled. "
        "Install with: pip install evidently"
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GCS_MONITORING_BUCKET: str = os.getenv("MONITORING_BUCKET", "rewardsense-monitoring")
REFERENCE_GCS_PATH: str = os.getenv(
    "REFERENCE_PROFILE_PATH",
    "reference/training_reference.csv",
)
DRIFT_REPORT_GCS_PREFIX: str = os.getenv("DRIFT_REPORT_PREFIX", "drift-reports")
DEFAULT_FEATURE_DRIFT_THRESHOLD: float = float(
    os.getenv("FEATURE_DRIFT_THRESHOLD", "0.3")
)
DEFAULT_PREDICTION_DRIFT_THRESHOLD: float = float(
    os.getenv("PREDICTION_DRIFT_THRESHOLD", "0.1")
)


# =====================================================================
# Result
# =====================================================================


@dataclass
class DriftResult:
    """Results of a drift detection run."""

    timestamp: str
    drift_detected: bool = False
    dataset_drift_share: float = 0.0
    feature_drift_threshold: float = DEFAULT_FEATURE_DRIFT_THRESHOLD
    prediction_drift_threshold: float = DEFAULT_PREDICTION_DRIFT_THRESHOLD
    drifted_features: List[str] = field(default_factory=list)
    per_feature_drift: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    prediction_drift: Optional[Dict[str, Any]] = None
    n_reference: int = 0
    n_current: int = 0
    html_report_path: Optional[str] = None
    json_report_path: Optional[str] = None

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "drift_detected": self.drift_detected,
            "dataset_drift_share": round(self.dataset_drift_share, 4),
            "n_drifted_features": len(self.drifted_features),
            "drifted_features": self.drifted_features,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "thresholds": {
                "feature": self.feature_drift_threshold,
                "prediction": self.prediction_drift_threshold,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "per_feature_drift": self.per_feature_drift,
            "prediction_drift": self.prediction_drift,
            "html_report_path": self.html_report_path,
            "json_report_path": self.json_report_path,
        }


# =====================================================================
# DriftDetector
# =====================================================================


class DriftDetector:
    """Detect data and prediction drift using Evidently AI.

    Parameters
    ----------
    reference_path : str or Path, optional
        Local path to the reference (training) dataset CSV.
        If not found locally, attempts to download from GCS.
    feature_drift_threshold : float
        If more than this fraction of features drift, flag it.
    prediction_drift_threshold : float
        KL divergence threshold for prediction drift.
    output_dir : str or Path
        Directory to save drift reports.
    """

    def __init__(
        self,
        reference_path: Optional[str | Path] = None,
        feature_drift_threshold: float = DEFAULT_FEATURE_DRIFT_THRESHOLD,
        prediction_drift_threshold: float = DEFAULT_PREDICTION_DRIFT_THRESHOLD,
        output_dir: str | Path = "data/monitoring/drift-reports",
    ) -> None:
        self.reference_path = Path(reference_path) if reference_path else None
        self.feature_drift_threshold = feature_drift_threshold
        self.prediction_drift_threshold = prediction_drift_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reference_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Reference data loading
    # ------------------------------------------------------------------

    def load_reference(self, path: Optional[str | Path] = None) -> pd.DataFrame:
        """Load the reference dataset from local file or GCS."""
        load_path = Path(path) if path else self.reference_path

        # Try local first
        if load_path and load_path.exists():
            self._reference_df = pd.read_csv(load_path)
            logger.info(
                "Loaded reference data from %s (%d rows)",
                load_path,
                len(self._reference_df),
            )
            return self._reference_df

        # Try GCS
        if GCS_AVAILABLE:
            try:
                client = gcs_storage.Client()
                bucket = client.bucket(GCS_MONITORING_BUCKET)
                blob = bucket.blob(REFERENCE_GCS_PATH)

                if load_path is None:
                    load_path = self.output_dir / "training_reference.csv"
                load_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(load_path))

                self._reference_df = pd.read_csv(load_path)
                logger.info(
                    "Downloaded reference data from GCS (%d rows)",
                    len(self._reference_df),
                )
                return self._reference_df
            except Exception as e:
                logger.warning("Failed to load reference from GCS: %s", e)

        raise FileNotFoundError(
            "Reference dataset not found locally or in GCS. "
            "Run the model training pipeline to generate it."
        )

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        prediction_column: Optional[str] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> DriftResult:
        """Run drift detection comparing current data to reference.

        Parameters
        ----------
        current_df : DataFrame
            Recent inference data (from InferenceDataCollector).
        reference_df : DataFrame, optional
            Reference data. Uses cached if not provided.
        prediction_column : str, optional
            Column name for prediction drift. If None, only data drift.
        numerical_features, categorical_features : list[str], optional
            Explicit feature lists. Auto-detected if not provided.

        Returns
        -------
        DriftResult
        """
        if not EVIDENTLY_AVAILABLE:
            logger.error("Evidently not installed, cannot run drift detection")
            return DriftResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                drift_detected=False,
            )

        if reference_df is None:
            if self._reference_df is None:
                self.load_reference()
            reference_df = self._reference_df

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        result = DriftResult(
            timestamp=ts,
            feature_drift_threshold=self.feature_drift_threshold,
            prediction_drift_threshold=self.prediction_drift_threshold,
            n_reference=len(reference_df),
            n_current=len(current_df),
        )

        # Align columns
        common_cols = list(set(reference_df.columns) & set(current_df.columns))
        # Exclude non-feature columns
        exclude = {"request_id", "timestamp", "user_hash"}
        feature_cols = [c for c in common_cols if c not in exclude]

        if not feature_cols:
            logger.warning("No common feature columns between reference and current")
            return result

        ref_aligned = reference_df[feature_cols].copy()
        cur_aligned = current_df[feature_cols].copy()

        # Auto-detect feature types
        if numerical_features is None:
            numerical_features = list(
                ref_aligned.select_dtypes(include=[np.number]).columns
            )
        if categorical_features is None:
            categorical_features = list(
                ref_aligned.select_dtypes(include=["object", "category"]).columns
            )

        column_mapping = ColumnMapping(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target=prediction_column,
        )

        # --- Data Drift Report ---
        try:
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(
                reference_data=ref_aligned,
                current_data=cur_aligned,
                column_mapping=column_mapping,
            )

            # Extract results
            report_dict = data_drift_report.as_dict()
            metrics = report_dict.get("metrics", [])

            for metric in metrics:
                metric_id = metric.get("metric", "")
                metric_result = metric.get("result", {})

                if "DatasetDriftMetric" in metric_id:
                    result.dataset_drift_share = metric_result.get(
                        "share_of_drifted_columns", 0.0
                    )
                    result.drift_detected = metric_result.get("dataset_drift", False)

                if "DataDriftTable" in metric_id:
                    drift_by_columns = metric_result.get("drift_by_columns", {})
                    for col_name, col_data in drift_by_columns.items():
                        is_drifted = col_data.get("drift_detected", False)
                        result.per_feature_drift[col_name] = {
                            "drift_detected": is_drifted,
                            "drift_score": col_data.get("drift_score", 0),
                            "stattest_name": col_data.get("stattest_name", ""),
                            "threshold": col_data.get("stattest_threshold", 0),
                        }
                        if is_drifted:
                            result.drifted_features.append(col_name)

            # Check threshold
            if result.dataset_drift_share > self.feature_drift_threshold:
                result.drift_detected = True

            # Save HTML report
            html_path = self.output_dir / f"data_drift_{ts}.html"
            data_drift_report.save_html(str(html_path))
            result.html_report_path = str(html_path)

            # Save JSON
            json_path = self.output_dir / f"data_drift_{ts}.json"
            json_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
            result.json_report_path = str(json_path)

            logger.info(
                "Drift detection complete: drift=%s, share=%.2f%%, drifted=%d/%d features",
                result.drift_detected,
                result.dataset_drift_share * 100,
                len(result.drifted_features),
                len(feature_cols),
            )

        except Exception as e:
            logger.error("Evidently drift detection failed: %s", e)
            result.drift_detected = False

        # --- Upload reports to GCS ---
        self._upload_reports_to_gcs(result, ts)

        return result

    # ------------------------------------------------------------------
    # GCS upload
    # ------------------------------------------------------------------

    def _upload_reports_to_gcs(self, result: DriftResult, ts: str) -> None:
        """Upload drift reports to GCS monitoring bucket."""
        if not GCS_AVAILABLE:
            return

        try:
            client = gcs_storage.Client()
            bucket = client.bucket(GCS_MONITORING_BUCKET)

            if result.html_report_path and Path(result.html_report_path).exists():
                blob = bucket.blob(f"{DRIFT_REPORT_GCS_PREFIX}/data_drift_{ts}.html")
                blob.upload_from_filename(result.html_report_path)
                logger.info("Drift HTML report uploaded to GCS")

            if result.json_report_path and Path(result.json_report_path).exists():
                blob = bucket.blob(f"{DRIFT_REPORT_GCS_PREFIX}/data_drift_{ts}.json")
                blob.upload_from_filename(result.json_report_path)
                logger.info("Drift JSON report uploaded to GCS")

        except Exception as e:
            logger.warning("Failed to upload drift reports to GCS: %s", e)
