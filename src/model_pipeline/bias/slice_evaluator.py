"""
Model-Aware Slice Evaluator.

Extends DataSlicer to compute per-slice model metrics (NDCG, Precision@K, recommendation diversity) on predictions.

Reuses: src.data_pipeline.bias_detection.slicer.DataSlicer for data
partitioning. Adds model-specific metric aggregation on top.

Usage:
    evaluator = SliceEvaluator.from_yaml("config/bias_slices.yaml")
    report = evaluator.evaluate(
        df=test_data,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=["spending_tier", "card_portfolio_size"],
    )
    for s in report.slices:
        print(f"{s.name}: NDCG@5={s.metrics['ndcg_5']:.3f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass
class SliceMetrics:
    """Metrics for a single data slice."""

    name: str
    feature: str
    count: int
    fraction: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SliceEvaluationReport:
    """Aggregated report across all slices and features."""

    slices: List[SliceMetrics] = field(default_factory=list)
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    disparities: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "total_slices": len(self.slices),
            "disparities_found": len(self.disparities),
            "overall": self.overall_metrics,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_metrics": self.overall_metrics,
            "slices": [
                {
                    "name": s.name,
                    "feature": s.feature,
                    "count": s.count,
                    "fraction": s.fraction,
                    "metrics": s.metrics,
                }
                for s in self.slices
            ],
            "disparities": self.disparities,
            "summary": self.summary,
        }


# =====================================================================
# Built-in recommendation metrics
# =====================================================================


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Compute NDCG@K for a single user's ranked list.

    Parameters
    ----------
    y_true : array-like
        Ground truth relevance scores.
    y_pred : array-like
        Predicted scores (higher = more relevant).
    k : int
        Number of top positions to evaluate.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Rank by predicted scores
    order = np.argsort(-y_pred)[:k]
    gains = y_true[order]

    # DCG
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)

    # Ideal DCG
    ideal_order = np.argsort(-y_true)[:k]
    ideal_gains = y_true[ideal_order]
    idcg = np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2)))

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Precision@K: fraction of top-K predictions that are relevant."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(-y_pred)[:k]
    relevant = y_true[order] > 0
    return float(np.mean(relevant))


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Recall@K: fraction of relevant items in top-K."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_relevant = np.sum(y_true > 0)
    if total_relevant == 0:
        return 0.0
    order = np.argsort(-y_pred)[:k]
    hits = np.sum(y_true[order] > 0)
    return float(hits / total_relevant)


def recommendation_diversity(y_pred_labels: Sequence[Any]) -> float:
    """Diversity: fraction of unique items in predictions."""
    if len(y_pred_labels) == 0:
        return 0.0
    return len(set(y_pred_labels)) / len(y_pred_labels)


# Default metric registry
DEFAULT_METRICS: Dict[str, Callable] = {
    "ndcg_5": lambda yt, yp: ndcg_at_k(yt, yp, k=5),
    "precision_5": lambda yt, yp: precision_at_k(yt, yp, k=5),
    "recall_5": lambda yt, yp: recall_at_k(yt, yp, k=5),
}


# =====================================================================
# SliceEvaluator
# =====================================================================


class SliceEvaluator:
    """Evaluates model metrics across data slices for bias detection.

    Extends Phase 1's DataSlicer concept for model predictions.
    Computes per-slice metrics and flags disparities exceeding thresholds.

    Parameters
    ----------
    slicing_config : dict
        Defines slicing dimensions. Example:
        {
            "spending_tier": {"type": "quantile", "n_quantiles": 4, "column": "monthly_budget"},
            "card_portfolio_size": {"type": "categorical", "column": "num_cards"},
        }
    metrics : dict[str, Callable], optional
        Metric functions mapping name → callable(y_true, y_pred) → float.
    disparity_threshold : float
        Flag a slice if its metric deviates by more than this fraction
        from the overall mean. Default 0.10 (10%).
    """

    def __init__(
        self,
        slicing_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        disparity_threshold: float = 0.10,
    ) -> None:
        self.slicing_config = slicing_config or {}
        self.metrics = metrics or DEFAULT_METRICS
        self.disparity_threshold = disparity_threshold

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs) -> "SliceEvaluator":
        """Load slicing config from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")

        with open(p) as f:
            cfg = yaml.safe_load(f)

        return cls(
            slicing_config=cfg.get("slicing_dimensions", {}),
            disparity_threshold=cfg.get("disparity_threshold", 0.10),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Slicing logic
    # ------------------------------------------------------------------

    def _create_slices(
        self, df: pd.DataFrame, feature_config: Dict[str, Any]
    ) -> Dict[str, pd.Index]:
        """Partition DataFrame indices by a slicing dimension."""
        col = feature_config["column"]
        slice_type = feature_config.get("type", "categorical")

        if col not in df.columns:
            logger.warning("Slicing column '%s' not found in data", col)
            return {}

        slices: Dict[str, pd.Index] = {}

        if slice_type == "categorical":
            for val in df[col].dropna().unique():
                mask = df[col] == val
                slices[f"{col}={val}"] = df.index[mask]

        elif slice_type == "quantile":
            n = feature_config.get("n_quantiles", 4)
            labels = feature_config.get("labels", [f"Q{i+1}" for i in range(n)])
            try:
                bins = pd.qcut(df[col], q=n, labels=labels, duplicates="drop")
                for label in bins.dropna().unique():
                    mask = bins == label
                    slices[f"{col}={label}"] = df.index[mask]
            except ValueError as e:
                logger.warning("Quantile slicing failed for '%s': %s", col, e)

        return slices

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: Optional[List[str]] = None,
    ) -> SliceEvaluationReport:
        """Run per-slice evaluation across all configured dimensions.

        Parameters
        ----------
        df : DataFrame
            Test data with feature columns for slicing.
        y_true : array
            Ground truth relevance/labels.
        y_pred : array
            Model predictions/scores.
        sensitive_features : list[str], optional
            If provided, only evaluate these slicing dimensions.
            Otherwise, evaluate all configured dimensions.

        Returns
        -------
        SliceEvaluationReport
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        report = SliceEvaluationReport()

        # Overall metrics
        for mname, mfunc in self.metrics.items():
            try:
                report.overall_metrics[mname] = mfunc(y_true, y_pred)
            except Exception as e:
                logger.warning("Metric '%s' failed overall: %s", mname, e)
                report.overall_metrics[mname] = float("nan")

        # Determine which dimensions to evaluate
        dims = sensitive_features or list(self.slicing_config.keys())

        for dim_name in dims:
            if dim_name not in self.slicing_config:
                # Direct column name — treat as categorical
                cfg = {"column": dim_name, "type": "categorical"}
            else:
                cfg = self.slicing_config[dim_name]

            slices = self._create_slices(df, cfg)

            for slice_name, idx in slices.items():
                if len(idx) == 0:
                    continue

                # Compute per-slice metrics
                mask = df.index.isin(idx)
                yt_slice = y_true[mask]
                yp_slice = y_pred[mask]

                sm = SliceMetrics(
                    name=slice_name,
                    feature=cfg.get("column", dim_name),
                    count=len(idx),
                    fraction=len(idx) / len(df),
                )

                for mname, mfunc in self.metrics.items():
                    try:
                        sm.metrics[mname] = mfunc(yt_slice, yp_slice)
                    except Exception as e:
                        logger.warning(
                            "Metric '%s' failed for slice '%s': %s",
                            mname,
                            slice_name,
                            e,
                        )
                        sm.metrics[mname] = float("nan")

                report.slices.append(sm)

                # Check for disparities
                for mname in self.metrics:
                    overall = report.overall_metrics.get(mname, 0)
                    if overall == 0 or np.isnan(overall):
                        continue
                    slice_val = sm.metrics.get(mname, 0)
                    deviation = abs(slice_val - overall) / abs(overall)
                    if deviation > self.disparity_threshold:
                        report.disparities.append(
                            {
                                "slice": slice_name,
                                "feature": sm.feature,
                                "metric": mname,
                                "overall_value": overall,
                                "slice_value": slice_val,
                                "deviation": deviation,
                                "threshold": self.disparity_threshold,
                            }
                        )

        return report
