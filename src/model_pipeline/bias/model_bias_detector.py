"""
Model-Level Bias Detection with Fairlearn.

- Extends BiasDetector for model predictions.
- Computes per-slice fairness metrics using Fairlearn MetricFrame, flags disparities, and generates bias reports with visualizations.

Reuses: BiasConfig thresholds and Fairlearn patterns.
Adds: Model-prediction-aware metrics (NDCG per group, recommendation
rate parity, equalized recommendation quality).

Usage:
    detector = ModelBiasDetector()
    report = detector.detect(
        y_true=y_test,
        y_pred=model.predict(X_test),
        sensitive_features=test_df["spending_archetype"],
    )
    print(report.summary)
    report.log_to_mlflow(tracker)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plt_close(fig: Any) -> None:
    """Close a matplotlib figure, handling missing import."""
    try:
        import matplotlib.pyplot as _plt

        _plt.close(fig)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Lazy Fairlearn import
# ---------------------------------------------------------------------------
try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
    )

    FAIRLEARN_AVAILABLE = True
except ImportError:
    MetricFrame = None  # type: ignore[assignment,misc]
    FAIRLEARN_AVAILABLE = False
    logger.warning("fairlearn not installed — model bias detection limited")


# =====================================================================
# Config
# =====================================================================


@dataclass
class ModelBiasConfig:
    """Thresholds for model-level bias detection."""

    demographic_parity_threshold: float = 0.10
    equalized_odds_threshold: float = 0.10
    performance_disparity_threshold: float = 0.10  # per-slice metric deviation
    min_slice_size: int = 10  # skip slices smaller than this

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelBiasConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =====================================================================
# Report
# =====================================================================


@dataclass
class BiasMetricResult:
    """Single bias metric evaluation."""

    name: str
    sensitive_feature: str
    value: float
    threshold: float
    is_biased: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBiasReport:
    """Aggregated model bias report."""

    model_name: str
    metrics: List[BiasMetricResult] = field(default_factory=list)
    per_group_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metric_frame_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def biased_metrics(self) -> List[BiasMetricResult]:
        return [m for m in self.metrics if m.is_biased]

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_metrics": len(self.metrics),
            "biased": len(self.biased_metrics),
            "unbiased": len(self.metrics) - len(self.biased_metrics),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "biased_metrics": [
                {
                    "name": m.name,
                    "feature": m.sensitive_feature,
                    "value": m.value,
                    "threshold": m.threshold,
                    "details": m.details,
                }
                for m in self.biased_metrics
            ],
            "all_metrics": [
                {
                    "name": m.name,
                    "feature": m.sensitive_feature,
                    "value": m.value,
                    "is_biased": m.is_biased,
                }
                for m in self.metrics
            ],
            "per_group_metrics": self.per_group_metrics,
        }

    def log_to_mlflow(self, tracker: Any) -> None:
        """
        Log bias report to MLflow via RewardSenseTracker.

        Logs JSON report + matplotlib/seaborn visualizations:
          - Bias summary chart (metric values vs thresholds)
          - Per-group fairness breakdown (if Fairlearn ran)
        """
        if tracker is None:
            return
        # Log summary metrics
        tracker.log_metrics(
            {
                "bias_total_checks": len(self.metrics),
                "bias_flagged": len(self.biased_metrics),
            }
        )
        # Log per-metric values
        for m in self.metrics:
            safe_name = f"bias_{m.name}_{m.sensitive_feature}".replace(" ", "_")
            tracker.log_metric(safe_name, m.value)
        # Log full report as artifact
        tracker.log_dict(self.to_dict(), f"bias_report_{self.model_name}.json")

        # --- Visualizations ---
        try:
            from src.model_pipeline.bias.visualizations import (
                plot_bias_summary,
                plot_fairness_metrics,
            )

            # Bias summary: all metrics vs thresholds
            all_metrics_dicts = [
                {
                    "name": m.name,
                    "sensitive_feature": m.sensitive_feature,
                    "value": m.value,
                    "threshold": m.threshold,
                    "is_biased": m.is_biased,
                }
                for m in self.metrics
            ]
            if all_metrics_dicts:
                fig = plot_bias_summary(all_metrics_dicts)
                tracker.log_figure(fig, f"bias_summary_{self.model_name}.png")
                plt_close(fig)

            # Per-group fairness breakdown
            if self.per_group_metrics:
                fig = plot_fairness_metrics(self.per_group_metrics)
                tracker.log_figure(fig, f"fairness_groups_{self.model_name}.png")
                plt_close(fig)
        except ImportError:
            pass  # matplotlib not installed — skip visualizations


# =====================================================================
# ModelBiasDetector
# =====================================================================


class ModelBiasDetector:
    """
    Detect bias in model predictions using Fairlearn + custom metrics.

    Parameters
    ----------
    config : ModelBiasConfig
        Thresholds for bias determination.
    custom_metrics : dict[str, Callable], optional
        Additional metrics beyond the built-in ones.
        Each callable: (y_true, y_pred) → float.
    """

    def __init__(
        self,
        config: Optional[ModelBiasConfig] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self.config = config or ModelBiasConfig()
        self.custom_metrics = custom_metrics or {}
        logger.info("ModelBiasDetector initialized (fairlearn=%s)", FAIRLEARN_AVAILABLE)

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series, pd.DataFrame],
        model_name: str = "personalization",
    ) -> ModelBiasReport:
        """
        Run full bias detection on model predictions.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels/relevance scores.
        y_pred : array-like
            Model predictions/scores.
        sensitive_features : array-like or DataFrame
            Sensitive feature(s) to slice by. If DataFrame, each column
            is treated as a separate sensitive feature.
        model_name : str
            Name for report identification.

        Returns
        -------
        ModelBiasReport
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        report = ModelBiasReport(model_name=model_name)

        # Handle single vs multiple sensitive features
        if isinstance(sensitive_features, pd.DataFrame):
            feature_dict = {
                col: sensitive_features[col].values
                for col in sensitive_features.columns
            }
        elif isinstance(sensitive_features, pd.Series):
            feature_dict = {
                sensitive_features.name or "group": sensitive_features.values
            }
        else:
            feature_dict = {"group": np.asarray(sensitive_features)}

        for feat_name, feat_values in feature_dict.items():
            # --- Fairlearn metrics ---
            if FAIRLEARN_AVAILABLE:
                self._run_fairlearn_checks(
                    y_true, y_pred, feat_values, feat_name, report
                )

            # --- Per-group performance disparity ---
            self._check_performance_disparity(
                y_true, y_pred, feat_values, feat_name, report
            )

            # --- Custom metrics ---
            for metric_name, metric_fn in self.custom_metrics.items():
                self._check_custom_metric(
                    y_true,
                    y_pred,
                    feat_values,
                    feat_name,
                    metric_name,
                    metric_fn,
                    report,
                )

        return report

    # ------------------------------------------------------------------
    # Fairlearn checks
    # ------------------------------------------------------------------

    def _run_fairlearn_checks(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
        feat_name: str,
        report: ModelBiasReport,
    ) -> None:
        """Run demographic parity and equalized odds via Fairlearn."""
        # Binarize predictions for Fairlearn classification metrics
        # (threshold at median for continuous predictions)
        if np.issubdtype(y_pred.dtype, np.floating):
            y_pred_binary = (y_pred >= np.median(y_pred)).astype(int)
        else:
            y_pred_binary = y_pred

        y_true_binary = (
            (y_true > 0).astype(int)
            if np.issubdtype(y_true.dtype, np.floating)
            else y_true
        )

        # Demographic parity
        try:
            dp_diff = demographic_parity_difference(
                y_true_binary, y_pred_binary, sensitive_features=sensitive
            )
            report.metrics.append(
                BiasMetricResult(
                    name="demographic_parity_difference",
                    sensitive_feature=feat_name,
                    value=abs(dp_diff),
                    threshold=self.config.demographic_parity_threshold,
                    is_biased=abs(dp_diff) > self.config.demographic_parity_threshold,
                    details={"raw_value": float(dp_diff)},
                )
            )
        except Exception as e:
            logger.warning("Demographic parity failed for '%s': %s", feat_name, e)

        # Equalized odds
        try:
            eo_diff = equalized_odds_difference(
                y_true_binary, y_pred_binary, sensitive_features=sensitive
            )
            report.metrics.append(
                BiasMetricResult(
                    name="equalized_odds_difference",
                    sensitive_feature=feat_name,
                    value=abs(eo_diff),
                    threshold=self.config.equalized_odds_threshold,
                    is_biased=abs(eo_diff) > self.config.equalized_odds_threshold,
                    details={"raw_value": float(eo_diff)},
                )
            )
        except Exception as e:
            logger.warning("Equalized odds failed for '%s': %s", feat_name, e)

        # MetricFrame for per-group breakdown
        try:

            def accuracy(yt, yp):
                return float(np.mean(yt == yp))

            mf = MetricFrame(
                metrics={"accuracy": accuracy},
                y_true=y_true_binary,
                y_pred=y_pred_binary,
                sensitive_features=sensitive,
            )
            report.per_group_metrics[feat_name] = {
                str(k): float(v) for k, v in mf.by_group["accuracy"].items()
            }
            report.metric_frame_results[feat_name] = {
                "overall": float(mf.overall["accuracy"]),
                "by_group": report.per_group_metrics[feat_name],
                "difference": float(mf.difference()["accuracy"]),
                "ratio": float(mf.ratio()["accuracy"]),
            }
        except Exception as e:
            logger.warning("MetricFrame failed for '%s': %s", feat_name, e)

    # ------------------------------------------------------------------
    # Performance disparity
    # ------------------------------------------------------------------

    def _check_performance_disparity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
        feat_name: str,
        report: ModelBiasReport,
    ) -> None:
        """Check if model performance varies significantly across groups."""
        groups = pd.Series(sensitive)
        unique_groups = groups.unique()

        # Compute per-group mean absolute error
        overall_mae = float(np.mean(np.abs(y_true - y_pred)))
        group_maes = {}

        for g in unique_groups:
            mask = groups == g
            if mask.sum() < self.config.min_slice_size:
                continue
            g_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
            group_maes[str(g)] = g_mae

        if not group_maes or overall_mae == 0:
            return

        max_mae = max(group_maes.values())
        min_mae = min(group_maes.values())

        # Ratio-based disparity
        if min_mae > 0:
            disparity_ratio = max_mae / min_mae
        else:
            disparity_ratio = float("inf") if max_mae > 0 else 1.0

        # Deviation-based disparity
        max_deviation = max(
            abs(v - overall_mae) / overall_mae for v in group_maes.values()
        )

        report.metrics.append(
            BiasMetricResult(
                name="performance_disparity",
                sensitive_feature=feat_name,
                value=max_deviation,
                threshold=self.config.performance_disparity_threshold,
                is_biased=max_deviation > self.config.performance_disparity_threshold,
                details={
                    "overall_mae": overall_mae,
                    "group_maes": group_maes,
                    "disparity_ratio": disparity_ratio,
                    "max_deviation": max_deviation,
                },
            )
        )

    # ------------------------------------------------------------------
    # Custom metrics
    # ------------------------------------------------------------------

    def _check_custom_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
        feat_name: str,
        metric_name: str,
        metric_fn: Callable,
        report: ModelBiasReport,
    ) -> None:
        """Evaluate a custom metric per group and check for disparity."""
        groups = pd.Series(sensitive)

        overall_val = metric_fn(y_true, y_pred)
        group_vals = {}

        for g in groups.unique():
            mask = (groups == g).values
            if mask.sum() < self.config.min_slice_size:
                continue
            group_vals[str(g)] = metric_fn(y_true[mask], y_pred[mask])

        if not group_vals or overall_val == 0:
            return

        max_deviation = max(
            abs(v - overall_val) / abs(overall_val) for v in group_vals.values()
        )

        report.metrics.append(
            BiasMetricResult(
                name=f"custom_{metric_name}",
                sensitive_feature=feat_name,
                value=max_deviation,
                threshold=self.config.performance_disparity_threshold,
                is_biased=max_deviation > self.config.performance_disparity_threshold,
                details={
                    "overall": overall_val,
                    "per_group": group_vals,
                },
            )
        )
