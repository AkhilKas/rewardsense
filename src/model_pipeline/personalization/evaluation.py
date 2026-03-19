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
class RankingMetrics:
    """Container for ranking-quality metrics.

    These evaluate the model's ability to correctly *rank* items
    (cards) by predicted point value, which matters because the
    downstream use-case is card recommendation.
    """

    ndcg_at_k: float
    map_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    k: int

    def to_dict(self) -> Dict[str, float]:
        return {
            f"ndcg@{self.k}": self.ndcg_at_k,
            f"map@{self.k}": self.map_at_k,
            f"precision@{self.k}": self.precision_at_k,
            f"recall@{self.k}": self.recall_at_k,
            "mrr": self.mrr,
            "k": self.k,
        }


@dataclass
class EvaluationReport:
    """Full evaluation report including segment breakdowns."""

    overall: RegressionMetrics
    ranking: Optional[RankingMetrics] = None
    segment_metrics: Dict[str, Dict[str, RegressionMetrics]] = field(
        default_factory=dict
    )
    overfitting_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"overall": self.overall.to_dict()}
        if self.ranking:
            result["ranking"] = self.ranking.to_dict()
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


def _dcg(relevances: np.ndarray) -> float:
    """Discounted Cumulative Gain."""
    return float(np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2))))


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 5,
) -> RankingMetrics:
    """Compute ranking-quality metrics by treating predictions as scores.

    Items are grouped into ``k`` "relevant" items (those with the highest
    true values).  Predicted scores are used to produce a ranking, and
    standard IR metrics are computed against the true top-k set.

    Parameters
    ----------
    y_true, y_pred : array-like  (same length)
    k : int
        Cut-off for @K metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    k = min(k, n)

    true_top_k_idx = set(np.argsort(y_true)[::-1][:k])
    pred_ranked_idx = np.argsort(y_pred)[::-1]

    # --- NDCG@K ---
    pred_relevances = np.array(
        [1.0 if pred_ranked_idx[i] in true_top_k_idx else 0.0 for i in range(k)]
    )
    ideal_relevances = np.ones(k)
    ndcg = (
        _dcg(pred_relevances) / _dcg(ideal_relevances)
        if _dcg(ideal_relevances) > 0
        else 0.0
    )

    # --- Precision@K ---
    hits_at_k = sum(1 for i in range(k) if pred_ranked_idx[i] in true_top_k_idx)
    precision = hits_at_k / k

    # --- Recall@K ---
    recall = hits_at_k / len(true_top_k_idx) if true_top_k_idx else 0.0

    # --- MAP@K (Average Precision) ---
    cum_hits = 0
    precision_sum = 0.0
    for i in range(k):
        if pred_ranked_idx[i] in true_top_k_idx:
            cum_hits += 1
            precision_sum += cum_hits / (i + 1)
    map_at_k = precision_sum / min(k, len(true_top_k_idx)) if true_top_k_idx else 0.0

    # --- MRR (Mean Reciprocal Rank) ---
    mrr = 0.0
    for i in range(n):
        if pred_ranked_idx[i] in true_top_k_idx:
            mrr = 1.0 / (i + 1)
            break

    result = RankingMetrics(
        ndcg_at_k=round(ndcg, 4),
        map_at_k=round(map_at_k, 4),
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
        k=k,
    )
    logger.info(
        "Ranking@{} — NDCG: {}, MAP: {}, P: {}, R: {}, MRR: {}",
        k,
        result.ndcg_at_k,
        result.map_at_k,
        result.precision_at_k,
        result.recall_at_k,
        result.mrr,
    )
    return result


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
    ranking_k: int = 5,
) -> EvaluationReport:
    """Full evaluation: overall + ranking + segments + overfitting check.

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
    ranking_k : int
        Cut-off K for ranking metrics.
    """
    overall = compute_regression_metrics(y_true.values, y_pred)
    logger.info(
        "Overall — RMSE: {:.6f}, MAE: {:.6f}, R²: {:.4f}",
        overall.rmse,
        overall.mae,
        overall.r2,
    )

    ranking = compute_ranking_metrics(y_true.values, y_pred, k=ranking_k)

    segments: Dict[str, Dict[str, RegressionMetrics]] = {}
    if meta is not None:
        segments = compute_segment_metrics(y_true, y_pred, meta, segment_columns)

    overfit: Optional[Dict[str, Any]] = None
    if train_metrics is not None:
        overfit = check_overfitting(train_metrics, overall)

    return EvaluationReport(
        overall=overall,
        ranking=ranking,
        segment_metrics=segments,
        overfitting_check=overfit,
    )


def _detect_segment_columns(meta: pd.DataFrame) -> List[str]:
    """Auto-detect segment columns from metadata."""
    candidates = ["archetype", "age_group", "location_type", "budget_quartile"]
    return [c for c in candidates if c in meta.columns]
