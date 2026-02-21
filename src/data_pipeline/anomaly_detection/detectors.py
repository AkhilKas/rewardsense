"""
Anomaly Detection Engine

Implement detection for:
  - Missing value ratios exceeding thresholds
  - Numerical outliers via IQR and Z-score methods
  - Schema violations (missing columns, type mismatches)
  - Distribution drift between reference and current data
  - Configurable thresholds loaded from YAML
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# =========================================================================
# Data structures
# =========================================================================


class AnomalySeverity(IntEnum):
    """Severity level for detected anomalies."""

    INFO = 0
    WARNING = 1
    CRITICAL = 2


@dataclass
class Anomaly:
    """A single detected anomaly."""

    check_name: str
    severity: AnomalySeverity
    message: str
    dataset: str = ""
    column: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d


@dataclass
class AnomalyReport:
    """Aggregated results of all anomaly checks on a dataset."""

    dataset: str
    total_checks: int = 0
    anomalies: List[Anomaly] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical(self) -> bool:
        return any(a.severity == AnomalySeverity.CRITICAL for a in self.anomalies)

    @property
    def has_warnings(self) -> bool:
        return any(a.severity >= AnomalySeverity.WARNING for a in self.anomalies)

    @property
    def max_severity(self) -> AnomalySeverity:
        if not self.anomalies:
            return AnomalySeverity.INFO
        return AnomalySeverity(max(a.severity for a in self.anomalies))

    @property
    def summary(self) -> Dict[str, int]:
        counts = {s.name: 0 for s in AnomalySeverity}
        for a in self.anomalies:
            counts[a.severity.name] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "total_checks": self.total_checks,
            "anomaly_count": len(self.anomalies),
            "summary": self.summary,
            "has_critical": self.has_critical,
            "max_severity": self.max_severity.name,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "metadata": self.metadata,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)
            + "\n",
            encoding="utf-8",
        )


# =========================================================================
# Configuration
# =========================================================================


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection thresholds."""

    # Missing value thresholds (fraction)
    missing_warning_threshold: float = 0.05
    missing_critical_threshold: float = 0.20

    # Outlier detection
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    outlier_ratio_warning: float = 0.05
    outlier_ratio_critical: float = 0.15

    # Drift detection (KS test p-value)
    drift_warning_pvalue: float = 0.05
    drift_critical_pvalue: float = 0.001

    # Schema
    required_columns: Dict[str, List[str]] = field(default_factory=dict)
    expected_dtypes: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnomalyConfig":
        return cls(
            missing_warning_threshold=float(d.get("missing_warning_threshold", 0.05)),
            missing_critical_threshold=float(d.get("missing_critical_threshold", 0.20)),
            iqr_multiplier=float(d.get("iqr_multiplier", 1.5)),
            zscore_threshold=float(d.get("zscore_threshold", 3.0)),
            outlier_ratio_warning=float(d.get("outlier_ratio_warning", 0.05)),
            outlier_ratio_critical=float(d.get("outlier_ratio_critical", 0.15)),
            drift_warning_pvalue=float(d.get("drift_warning_pvalue", 0.05)),
            drift_critical_pvalue=float(d.get("drift_critical_pvalue", 0.001)),
            required_columns=dict(d.get("required_columns", {})),
            expected_dtypes=dict(d.get("expected_dtypes", {})),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "AnomalyConfig":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(cfg.get("anomaly_detection", cfg))


# =========================================================================
# Detector
# =========================================================================


class AnomalyDetector:
    """Runs anomaly checks against a DataFrame.

    Parameters
    ----------
    config : AnomalyConfig
        Detection thresholds and rules.
    """

    def __init__(self, config: Optional[AnomalyConfig] = None) -> None:
        self.config = config or AnomalyConfig()
        logger.info("AnomalyDetector initialized")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnomalyDetector":
        cfg = AnomalyConfig.from_yaml(Path(path))
        return cls(config=cfg)

    # ------------------------------------------------------------------
    # Missing value detection
    # ------------------------------------------------------------------

    def detect_missing_values(
        self,
        df: pd.DataFrame,
        dataset: str = "",
    ) -> List[Anomaly]:
        """Flag columns whose missing-value ratio exceeds thresholds."""
        if df.empty:
            return []

        anomalies: List[Anomaly] = []
        n = len(df)

        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            ratio = n_missing / n

            if ratio >= self.config.missing_critical_threshold:
                anomalies.append(
                    Anomaly(
                        check_name="missing_values",
                        severity=AnomalySeverity.CRITICAL,
                        message=(
                            f"Column '{col}' has {ratio:.1%} missing values "
                            f"({n_missing}/{n})"
                        ),
                        dataset=dataset,
                        column=col,
                        details={
                            "missing_count": n_missing,
                            "missing_ratio": round(ratio, 4),
                        },
                    )
                )
            elif ratio >= self.config.missing_warning_threshold:
                anomalies.append(
                    Anomaly(
                        check_name="missing_values",
                        severity=AnomalySeverity.WARNING,
                        message=(
                            f"Column '{col}' has {ratio:.1%} missing values "
                            f"({n_missing}/{n})"
                        ),
                        dataset=dataset,
                        column=col,
                        details={
                            "missing_count": n_missing,
                            "missing_ratio": round(ratio, 4),
                        },
                    )
                )

        if anomalies:
            logger.warning(
                "[%s] Missing value anomalies: %d columns flagged",
                dataset,
                len(anomalies),
            )
        else:
            logger.info("[%s] Missing value check passed", dataset)

        return anomalies

    # ------------------------------------------------------------------
    # Outlier detection — IQR
    # ------------------------------------------------------------------

    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        dataset: str = "",
    ) -> List[Anomaly]:
        """Detect outliers using the IQR method on numeric columns."""
        if df.empty:
            return []

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        anomalies: List[Anomaly] = []
        n = len(df)

        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 4:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - self.config.iqr_multiplier * iqr
            upper = q3 + self.config.iqr_multiplier * iqr
            n_outliers = int(((series < lower) | (series > upper)).sum())
            ratio = n_outliers / n

            severity = None
            if ratio >= self.config.outlier_ratio_critical:
                severity = AnomalySeverity.CRITICAL
            elif ratio >= self.config.outlier_ratio_warning:
                severity = AnomalySeverity.WARNING

            if severity is not None:
                anomalies.append(
                    Anomaly(
                        check_name="outlier_iqr",
                        severity=severity,
                        message=(
                            f"Column '{col}' has {ratio:.1%} outliers "
                            f"({n_outliers}/{n}) outside [{lower:.2f}, {upper:.2f}]"
                        ),
                        dataset=dataset,
                        column=col,
                        details={
                            "outlier_count": n_outliers,
                            "outlier_ratio": round(ratio, 4),
                            "q1": round(q1, 4),
                            "q3": round(q3, 4),
                            "iqr": round(iqr, 4),
                            "lower_bound": round(lower, 4),
                            "upper_bound": round(upper, 4),
                        },
                    )
                )

        if anomalies:
            logger.warning(
                "[%s] IQR outlier anomalies: %d columns", dataset, len(anomalies)
            )

        return anomalies

    # ------------------------------------------------------------------
    # Outlier detection — Z-score
    # ------------------------------------------------------------------

    def detect_outliers_zscore(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        dataset: str = "",
    ) -> List[Anomaly]:
        """Detect outliers using Z-score on numeric columns."""
        if df.empty:
            return []

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        anomalies: List[Anomaly] = []
        n = len(df)

        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 2:
                continue

            mean = float(series.mean())
            std = float(series.std())
            if std == 0:
                continue

            z_scores = ((series - mean) / std).abs()
            n_outliers = int((z_scores > self.config.zscore_threshold).sum())
            ratio = n_outliers / n

            severity = None
            if ratio >= self.config.outlier_ratio_critical:
                severity = AnomalySeverity.CRITICAL
            elif ratio >= self.config.outlier_ratio_warning:
                severity = AnomalySeverity.WARNING

            if severity is not None:
                anomalies.append(
                    Anomaly(
                        check_name="outlier_zscore",
                        severity=severity,
                        message=(
                            f"Column '{col}' has {ratio:.1%} outliers "
                            f"({n_outliers}/{n}) with |z| > {self.config.zscore_threshold}"
                        ),
                        dataset=dataset,
                        column=col,
                        details={
                            "outlier_count": n_outliers,
                            "outlier_ratio": round(ratio, 4),
                            "mean": round(mean, 4),
                            "std": round(std, 4),
                            "threshold": self.config.zscore_threshold,
                        },
                    )
                )

        if anomalies:
            logger.warning(
                "[%s] Z-score outlier anomalies: %d columns", dataset, len(anomalies)
            )

        return anomalies

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def detect_schema_violations(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        expected_dtypes: Optional[Dict[str, str]] = None,
        dataset: str = "",
    ) -> List[Anomaly]:
        """Check for missing required columns and dtype mismatches."""
        anomalies: List[Anomaly] = []

        # resolve from config if not passed
        if required_columns is None:
            required_columns = self.config.required_columns.get(dataset, [])
        if expected_dtypes is None:
            expected_dtypes = self.config.expected_dtypes.get(dataset, {})

        # missing columns
        actual_cols: Set[str] = set(df.columns)
        for col in required_columns:
            if col not in actual_cols:
                anomalies.append(
                    Anomaly(
                        check_name="schema_missing_column",
                        severity=AnomalySeverity.CRITICAL,
                        message=f"Required column '{col}' is missing",
                        dataset=dataset,
                        column=col,
                    )
                )

        # dtype mismatches
        dtype_map = {
            "int": "int",
            "float": "float",
            "str": "object",
            "string": "object",
            "object": "object",
            "bool": "bool",
            "datetime": "datetime",
        }
        for col, expected_type in expected_dtypes.items():
            if col not in actual_cols:
                continue
            actual_kind = df[col].dtype.kind
            expected_kind = dtype_map.get(expected_type.lower(), expected_type)

            # simple kind check: i=int, f=float, O=object, b=bool, M=datetime
            kind_map = {
                "int": "i",
                "float": "f",
                "object": "O",
                "bool": "b",
                "datetime": "M",
            }
            expected_kind_code = kind_map.get(expected_kind, "")

            if expected_kind_code and actual_kind != expected_kind_code:
                anomalies.append(
                    Anomaly(
                        check_name="schema_dtype_mismatch",
                        severity=AnomalySeverity.WARNING,
                        message=(
                            f"Column '{col}' expected dtype '{expected_type}' "
                            f"but got '{df[col].dtype}'"
                        ),
                        dataset=dataset,
                        column=col,
                        details={
                            "expected": expected_type,
                            "actual": str(df[col].dtype),
                        },
                    )
                )

        if anomalies:
            logger.warning("[%s] Schema violations: %d issues", dataset, len(anomalies))
        else:
            logger.info("[%s] Schema check passed", dataset)

        return anomalies

    # ------------------------------------------------------------------
    # Distribution drift
    # ------------------------------------------------------------------

    def detect_distribution_drift(
        self,
        df_reference: pd.DataFrame,
        df_current: pd.DataFrame,
        columns: Optional[List[str]] = None,
        dataset: str = "",
    ) -> List[Anomaly]:
        """Detect distribution drift using the Kolmogorov-Smirnov test.

        Compares each numeric column between reference and current DataFrames.
        """
        from scipy import stats as sp_stats

        if df_reference.empty or df_current.empty:
            return []

        if columns is None:
            ref_num = set(df_reference.select_dtypes(include=[np.number]).columns)
            cur_num = set(df_current.select_dtypes(include=[np.number]).columns)
            columns = sorted(ref_num & cur_num)

        anomalies: List[Anomaly] = []

        for col in columns:
            if col not in df_reference.columns or col not in df_current.columns:
                continue

            ref_vals = df_reference[col].dropna().to_numpy()
            cur_vals = df_current[col].dropna().to_numpy()

            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue

            ks_stat, p_value = sp_stats.ks_2samp(ref_vals, cur_vals)

            severity = None
            if p_value <= self.config.drift_critical_pvalue:
                severity = AnomalySeverity.CRITICAL
            elif p_value <= self.config.drift_warning_pvalue:
                severity = AnomalySeverity.WARNING

            if severity is not None:
                anomalies.append(
                    Anomaly(
                        check_name="distribution_drift",
                        severity=severity,
                        message=(
                            f"Column '{col}' shows distribution drift "
                            f"(KS stat={ks_stat:.4f}, p={p_value:.6f})"
                        ),
                        dataset=dataset,
                        column=col,
                        details={
                            "ks_statistic": round(ks_stat, 6),
                            "p_value": round(p_value, 6),
                            "ref_mean": round(float(np.mean(ref_vals)), 4),
                            "cur_mean": round(float(np.mean(cur_vals)), 4),
                            "ref_std": round(float(np.std(ref_vals)), 4),
                            "cur_std": round(float(np.std(cur_vals)), 4),
                            "ref_count": len(ref_vals),
                            "cur_count": len(cur_vals),
                        },
                    )
                )

        if anomalies:
            logger.warning(
                "[%s] Distribution drift: %d columns drifted", dataset, len(anomalies)
            )
        else:
            logger.info("[%s] No distribution drift detected", dataset)

        return anomalies

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------

    def run_all_checks(
        self,
        df: pd.DataFrame,
        dataset: str = "",
        reference_df: Optional[pd.DataFrame] = None,
        numeric_columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        expected_dtypes: Optional[Dict[str, str]] = None,
    ) -> AnomalyReport:
        """Run all anomaly detection checks and return an aggregated report."""
        logger.info("[%s] Running all anomaly checks on %d rows", dataset, len(df))

        report = AnomalyReport(dataset=dataset)
        all_anomalies: List[Anomaly] = []
        checks_run = 0

        # 1. Missing values
        checks_run += 1
        all_anomalies.extend(self.detect_missing_values(df, dataset))

        # 2. Outliers (IQR)
        checks_run += 1
        all_anomalies.extend(self.detect_outliers_iqr(df, numeric_columns, dataset))

        # 3. Outliers (Z-score)
        checks_run += 1
        all_anomalies.extend(self.detect_outliers_zscore(df, numeric_columns, dataset))

        # 4. Schema violations
        checks_run += 1
        all_anomalies.extend(
            self.detect_schema_violations(
                df, required_columns, expected_dtypes, dataset
            )
        )

        # 5. Distribution drift (if reference provided)
        if reference_df is not None:
            checks_run += 1
            all_anomalies.extend(
                self.detect_distribution_drift(
                    reference_df, df, numeric_columns, dataset
                )
            )

        report.total_checks = checks_run
        report.anomalies = all_anomalies
        report.metadata = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        logger.info(
            "[%s] Anomaly detection complete: %d checks, %d anomalies (%s)",
            dataset,
            checks_run,
            len(all_anomalies),
            report.summary,
        )
        return report
