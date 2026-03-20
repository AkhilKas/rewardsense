"""
Bias Drift Monitor.

Compares bias reports across model versions to detect if fairness
degrades over retraining cycles. Stores historical reports as JSON
and computes drift metrics between consecutive versions.

Usage:
    monitor = BiasDriftMonitor(history_dir="data/bias_history")

    # After each training run, record the report
    monitor.record(report, model_version="1.2.0")

    # Compare two versions
    drift = monitor.compare("1.1.0", "1.2.0")
    print(drift.summary)

    # Check if drift exceeds alerting thresholds
    if drift.has_regression:
        print("Fairness regression detected!")

    # Get full history trend
    trend = monitor.trend(metric="demographic_parity_difference")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass
class DriftMetric:
    """Single metric drift between two versions."""

    name: str
    sensitive_feature: str
    before_value: float
    after_value: float
    absolute_change: float
    relative_change: float  # (after - before) / |before| if before != 0
    regression: bool  # True if fairness got worse (value increased)


@dataclass
class BiasDriftReport:
    """Drift comparison between two model versions."""

    before_version: str
    after_version: str
    timestamp: str
    metrics: List[DriftMetric] = field(default_factory=list)
    regression_threshold: float = 0.05  # 5% relative increase = regression

    @property
    def has_regression(self) -> bool:
        return any(m.regression for m in self.metrics)

    @property
    def regressions(self) -> List[DriftMetric]:
        return [m for m in self.metrics if m.regression]

    @property
    def improvements(self) -> List[DriftMetric]:
        return [m for m in self.metrics if m.absolute_change < 0]

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "before_version": self.before_version,
            "after_version": self.after_version,
            "total_metrics": len(self.metrics),
            "regressions": len(self.regressions),
            "improvements": len(self.improvements),
            "stable": len(self.metrics)
            - len(self.regressions)
            - len(self.improvements),
            "has_regression": self.has_regression,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "timestamp": self.timestamp,
            "regression_threshold": self.regression_threshold,
            "metrics": [
                {
                    "name": m.name,
                    "sensitive_feature": m.sensitive_feature,
                    "before": m.before_value,
                    "after": m.after_value,
                    "absolute_change": m.absolute_change,
                    "relative_change": m.relative_change,
                    "regression": m.regression,
                }
                for m in self.metrics
            ],
        }

    def log_to_mlflow(self, tracker: Any) -> None:
        """Log drift report to MLflow."""
        if tracker is None:
            return
        tracker.log_metrics(
            {
                "drift_regressions": len(self.regressions),
                "drift_improvements": len(self.improvements),
                "drift_has_regression": int(self.has_regression),
            }
        )
        tracker.log_dict(
            self.to_dict(),
            f"bias_drift_{self.before_version}_to_{self.after_version}.json",
        )

        # Visualization
        try:
            from src.model_pipeline.bias.visualizations import (
                plot_mitigation_comparison,
            )
            import matplotlib.pyplot as _plt

            before = {
                f"{m.name}_{m.sensitive_feature}": m.before_value for m in self.metrics
            }
            after = {
                f"{m.name}_{m.sensitive_feature}": m.after_value for m in self.metrics
            }
            if before or after:
                fig = plot_mitigation_comparison(
                    before,
                    after,
                    title=(
                        f"Bias Drift: v{self.before_version} → "
                        f"v{self.after_version}"
                    ),
                )
                tracker.log_figure(
                    fig,
                    f"bias_drift_{self.before_version}_to_" f"{self.after_version}.png",
                )
                _plt.close(fig)
        except ImportError:
            pass


@dataclass
class TrendPoint:
    """Single point in a bias metric trend."""

    version: str
    timestamp: str
    value: float


# =====================================================================
# BiasDriftMonitor
# =====================================================================


class BiasDriftMonitor:
    """Track and compare bias reports across model versions.

    Stores historical bias reports as JSON files and computes
    drift metrics between versions.

    Parameters
    ----------
    history_dir : str or Path
        Directory to store/load historical bias reports.
    regression_threshold : float
        A relative increase above this threshold flags a regression.
        Default 0.05 (5%).
    """

    def __init__(
        self,
        history_dir: str | Path = "data/bias_history",
        regression_threshold: float = 0.05,
    ) -> None:
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.regression_threshold = regression_threshold

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(
        self,
        report: Any,
        model_version: str,
        model_name: str = "personalization",
    ) -> Path:
        """Save a bias report for a model version.

        Parameters
        ----------
        report : ModelBiasReport or ComponentBiasReport
            The bias report to record.
        model_version : str
            Semantic version of the model.
        model_name : str
            Logical model name.

        Returns
        -------
        Path
            Path to the saved report file.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"{model_name}_v{model_version}_{ts}.json"
        filepath = self.history_dir / filename

        record_data = {
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": ts,
            "report": report.to_dict() if hasattr(report, "to_dict") else {},
        }

        filepath.write_text(json.dumps(record_data, indent=2, default=str))
        logger.info("Recorded bias report: %s", filepath)
        return filepath

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_report(
        self, model_name: str, model_version: str
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent report for a model version."""
        pattern = f"{model_name}_v{model_version}_*.json"
        matches = sorted(self.history_dir.glob(pattern), reverse=True)
        if not matches:
            return None
        return json.loads(matches[0].read_text())

    def list_versions(self, model_name: str = "personalization") -> List[str]:
        """List all recorded versions for a model, newest first."""
        pattern = f"{model_name}_v*.json"
        versions = set()
        for f in self.history_dir.glob(pattern):
            # Filename: model_v1.0.0_20260319T120000.json
            parts = f.stem.split("_v", 1)
            if len(parts) == 2:
                ver_ts = parts[1]
                # Split version from timestamp
                ver_parts = ver_ts.rsplit("_", 1)
                if ver_parts:
                    versions.add(ver_parts[0])
        return sorted(versions, reverse=True)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    def compare(
        self,
        before_version: str,
        after_version: str,
        model_name: str = "personalization",
    ) -> BiasDriftReport:
        """Compare bias metrics between two model versions.

        Parameters
        ----------
        before_version, after_version : str
            Version strings to compare.
        model_name : str
            Logical model name.

        Returns
        -------
        BiasDriftReport
        """
        before_data = self._load_report(model_name, before_version)
        after_data = self._load_report(model_name, after_version)

        if before_data is None:
            raise FileNotFoundError(
                f"No bias report found for {model_name} v{before_version}"
            )
        if after_data is None:
            raise FileNotFoundError(
                f"No bias report found for {model_name} v{after_version}"
            )

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        drift_report = BiasDriftReport(
            before_version=before_version,
            after_version=after_version,
            timestamp=ts,
            regression_threshold=self.regression_threshold,
        )

        # Extract metrics from both reports
        before_metrics = self._extract_metrics(before_data)
        after_metrics = self._extract_metrics(after_data)

        # Match metrics by (name, sensitive_feature) key
        all_keys = set(before_metrics.keys()) | set(after_metrics.keys())

        for key in all_keys:
            name, feat = key
            bv = before_metrics.get(key, 0.0)
            av = after_metrics.get(key, 0.0)

            abs_change = av - bv
            rel_change = (
                abs_change / abs(bv) if bv != 0 else (float("inf") if av != 0 else 0.0)
            )

            # Regression = bias metric increased by more than threshold
            regression = rel_change > self.regression_threshold

            drift_report.metrics.append(
                DriftMetric(
                    name=name,
                    sensitive_feature=feat,
                    before_value=bv,
                    after_value=av,
                    absolute_change=abs_change,
                    relative_change=rel_change,
                    regression=regression,
                )
            )

        return drift_report

    def _extract_metrics(self, record: Dict[str, Any]) -> Dict[tuple, float]:
        """Extract (name, feature) → value from a stored report."""
        metrics: Dict[tuple, float] = {}
        report = record.get("report", {})

        # Handle ModelBiasReport format
        all_metrics = report.get("all_metrics", [])
        for m in all_metrics:
            key = (m.get("name", ""), m.get("sensitive_feature", ""))
            metrics[key] = m.get("value", 0.0)

        # Handle ComponentBiasReport format
        component_metrics = report.get("metrics", [])
        for m in component_metrics:
            key = (m.get("check", m.get("name", "")), m.get("feature", ""))
            if key not in metrics:
                metrics[key] = m.get("value", 0.0)

        return metrics

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    def trend(
        self,
        metric_name: str,
        sensitive_feature: str = "",
        model_name: str = "personalization",
    ) -> List[TrendPoint]:
        """
        Get historical trend for a specific bias metric.

        Parameters
        ----------
        metric_name : str
            Name of the bias metric.
        sensitive_feature : str
            Sensitive feature to filter by (empty = first match).
        model_name : str
            Logical model name.

        Returns
        -------
        List[TrendPoint]
            Chronologically ordered trend points.
        """
        points: List[TrendPoint] = []
        pattern = f"{model_name}_v*.json"

        for filepath in sorted(self.history_dir.glob(pattern)):
            data = json.loads(filepath.read_text())
            version = data.get("model_version", "")
            ts = data.get("timestamp", "")
            metrics = self._extract_metrics(data)

            for (mname, feat), value in metrics.items():
                if mname == metric_name:
                    if not sensitive_feature or feat == sensitive_feature:
                        points.append(
                            TrendPoint(
                                version=version,
                                timestamp=ts,
                                value=value,
                            )
                        )
                        break  # one point per version

        return points

    def plot_trend(
        self,
        metric_name: str,
        sensitive_feature: str = "",
        model_name: str = "personalization",
        threshold: Optional[float] = None,
    ) -> Any:
        """
        Plot bias metric trend across versions.

        Returns a matplotlib Figure.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required: pip install matplotlib")

        points = self.trend(metric_name, sensitive_feature, model_name)

        if not points:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No trend data available", ha="center", va="center")
            ax.set_axis_off()
            return fig

        versions = [p.version for p in points]
        values = [p.value for p in points]

        from src.model_pipeline.bias.visualizations import COLORS

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(versions, values, marker="o", color=COLORS[0], linewidth=2)
        ax.fill_between(
            range(len(versions)),
            values,
            alpha=0.15,
            color=COLORS[0],
        )

        if threshold is not None:
            ax.axhline(
                y=threshold,
                color="#D55E00",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold ({threshold})",
            )
            ax.legend(fontsize=9)

        ax.set_xlabel("Model Version")
        ax.set_ylabel(metric_name)
        ax.set_title(
            f"Bias Drift Trend: {metric_name}"
            + (f" ({sensitive_feature})" if sensitive_feature else "")
        )
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions, rotation=45, ha="right")
        fig.tight_layout()
        return fig
