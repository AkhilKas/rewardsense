"""
Evaluation metrics for the personalization regression model.

Core metrics:
- RMSE, MAE, R² (regression)

Segment-level metrics:
- Per-archetype, per-age_group, per-budget_quartile breakdowns

Overfitting diagnostics:
- Train vs validation gap analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    """Container for core regression metrics."""

    rmse: float
    mae: float
    r2: float

    def to_dict(self) -> Dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}


@dataclass
class EvaluationReport:
    """Full evaluation report including segment breakdowns."""

    overall: RegressionMetrics
    segment_metrics: Dict[str, Dict[str, RegressionMetrics]] = field(
        default_factory=dict
    )
    overfitting_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"overall": self.overall.to_dict()}
        if self.segment_metrics:
            result["segments"] = {
                dim: {seg: m.to_dict() for seg, m in segs.items()}
                for dim, segs in self.segment_metrics.items()
            }
        if self.overfitting_check:
            result["overfitting_check"] = self.overfitting_check
        return result


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Compute RMSE, MAE, R² from true and predicted arrays."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2)


def compute_segment_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    segment_columns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, RegressionMetrics]]:
    """Compute per-segment regression metrics.

    Parameters
    ----------
    y_true : pd.Series
        True target values (index-aligned with meta).
    y_pred : np.ndarray
        Predicted values.
    meta : pd.DataFrame
        Metadata frame with segment columns (same index as y_true).
    segment_columns : list of str or None
        Columns to segment by. Auto-detects archetype/age/budget columns if None.

    Returns
    -------
    dict
        {dimension: {segment_value: RegressionMetrics}}
    """
    if segment_columns is None:
        segment_columns = _detect_segment_columns(meta)

    results: Dict[str, Dict[str, RegressionMetrics]] = {}

    pred_series = pd.Series(y_pred, index=y_true.index)

    for col in segment_columns:
        if col not in meta.columns:
            logger.warning("Segment column '{}' not in metadata, skipping", col)
            continue

        seg_results: Dict[str, RegressionMetrics] = {}
        for val, group_idx in meta.groupby(col).groups.items():
            if len(group_idx) < 2:
                continue
            yt = y_true.loc[group_idx].values
            yp = pred_series.loc[group_idx].values
            seg_results[str(val)] = compute_regression_metrics(yt, yp)

        results[col] = seg_results
        logger.info("Segment '{}': {} groups evaluated", col, len(seg_results))

    return results


def check_overfitting(
    train_metrics: RegressionMetrics,
    val_metrics: RegressionMetrics,
    rmse_gap_threshold: float = 0.3,
    r2_gap_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Compare train vs val metrics for overfitting signals.

    Returns a dict with gap values and a boolean flag.
    """
    rmse_gap = val_metrics.rmse - train_metrics.rmse
    r2_gap = train_metrics.r2 - val_metrics.r2
    is_overfit = rmse_gap > rmse_gap_threshold or r2_gap > r2_gap_threshold

    result = {
        "rmse_gap": round(rmse_gap, 6),
        "r2_gap": round(r2_gap, 6),
        "rmse_gap_threshold": rmse_gap_threshold,
        "r2_gap_threshold": r2_gap_threshold,
        "is_overfit": is_overfit,
    }

    if is_overfit:
        logger.warning("Overfitting detected: RMSE gap={}, R² gap={}", rmse_gap, r2_gap)
    else:
        logger.info("No overfitting: RMSE gap={}, R² gap={}", rmse_gap, r2_gap)

    return result


def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta: Optional[pd.DataFrame] = None,
    segment_columns: Optional[List[str]] = None,
    train_metrics: Optional[RegressionMetrics] = None,
) -> EvaluationReport:
    """Full evaluation: overall + segments + overfitting check.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth values.
    y_pred : np.ndarray
        Model predictions.
    meta : pd.DataFrame or None
        Metadata for segment-level analysis.
    segment_columns : list of str or None
        Which columns to segment by.
    train_metrics : RegressionMetrics or None
        If provided, runs overfitting comparison.
    """
    overall = compute_regression_metrics(y_true.values, y_pred)
    logger.info(
        "Overall — RMSE: {:.6f}, MAE: {:.6f}, R²: {:.4f}",
        overall.rmse,
        overall.mae,
        overall.r2,
    )

    segments: Dict[str, Dict[str, RegressionMetrics]] = {}
    if meta is not None:
        segments = compute_segment_metrics(y_true, y_pred, meta, segment_columns)

    overfit: Optional[Dict[str, Any]] = None
    if train_metrics is not None:
        overfit = check_overfitting(train_metrics, overall)

    return EvaluationReport(
        overall=overall,
        segment_metrics=segments,
        overfitting_check=overfit,
    )


def _detect_segment_columns(meta: pd.DataFrame) -> List[str]:
    """Auto-detect segment columns from metadata."""
    candidates = ["archetype", "age_group", "location_type", "budget_quartile"]
    return [c for c in candidates if c in meta.columns]
