"""
Bias Detection Metrics

Fairness metrics using Fairlearn + custom metrics.

Supports:
  - Demographic parity (equal selection rates across groups)
  - Equalized odds (equal error rates across groups)
  - Outcome disparity (reward value gaps between groups)
  - Custom per-slice metric comparisons
  - Configurable bias thresholds
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fairlearn is an optional but expected dependency
try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
    )

    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logger.warning("Fairlearn not installed — some bias metrics will be unavailable")


# =========================================================================
# Data structures
# =========================================================================


@dataclass
class BiasMetric:
    """A single computed fairness metric."""

    name: str
    value: float
    threshold: float
    is_biased: bool
    sensitive_feature: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "threshold": self.threshold,
            "is_biased": self.is_biased,
            "sensitive_feature": self.sensitive_feature,
            "details": self.details,
        }


@dataclass
class BiasReport:
    """Aggregated bias detection results."""

    dataset: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    sensitive_features_checked: List[str] = field(default_factory=list)
    metrics: List[BiasMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_bias(self) -> bool:
        return any(m.is_biased for m in self.metrics)

    @property
    def biased_metrics(self) -> List[BiasMetric]:
        return [m for m in self.metrics if m.is_biased]

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "total_metrics": len(self.metrics),
            "biased": sum(1 for m in self.metrics if m.is_biased),
            "unbiased": sum(1 for m in self.metrics if not m.is_biased),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "timestamp": self.timestamp,
            "sensitive_features_checked": self.sensitive_features_checked,
            "summary": self.summary,
            "has_bias": self.has_bias,
            "metrics": [m.to_dict() for m in self.metrics],
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
# Bias Detector
# =========================================================================


@dataclass
class BiasConfig:
    """Thresholds for bias detection."""

    # Demographic parity: |P(Y=1|A=a) - P(Y=1|A=b)| threshold
    demographic_parity_threshold: float = 0.10

    # Equalized odds threshold
    equalized_odds_threshold: float = 0.10

    # Outcome disparity: max allowed ratio between best/worst group means
    outcome_disparity_threshold: float = 2.0

    # Representation: min fraction any group should have
    representation_min_fraction: float = 0.05

    # Group metric difference threshold (generic)
    group_metric_diff_threshold: float = 0.20

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BiasConfig":
        return cls(
            demographic_parity_threshold=float(
                d.get("demographic_parity_threshold", 0.10)
            ),
            equalized_odds_threshold=float(d.get("equalized_odds_threshold", 0.10)),
            outcome_disparity_threshold=float(
                d.get("outcome_disparity_threshold", 2.0)
            ),
            representation_min_fraction=float(
                d.get("representation_min_fraction", 0.05)
            ),
            group_metric_diff_threshold=float(
                d.get("group_metric_diff_threshold", 0.20)
            ),
        )


class BiasDetector:
    """Detect bias in RewardSense data using Fairlearn and custom metrics.

    Parameters
    ----------
    config : BiasConfig
        Thresholds for bias determination.
    """

    def __init__(self, config: Optional[BiasConfig] = None) -> None:
        self.config = config or BiasConfig()
        logger.info("BiasDetector initialized (fairlearn=%s)", FAIRLEARN_AVAILABLE)

    # ------------------------------------------------------------------
    # Representation bias
    # ------------------------------------------------------------------

    def check_representation(
        self,
        df: pd.DataFrame,
        sensitive_feature: str,
    ) -> List[BiasMetric]:
        """Check if any group is underrepresented below threshold."""
        if sensitive_feature not in df.columns:
            return []

        metrics: List[BiasMetric] = []
        counts = df[sensitive_feature].value_counts(normalize=True)

        for group, fraction in counts.items():
            if fraction < self.config.representation_min_fraction:
                metrics.append(
                    BiasMetric(
                        name="underrepresentation",
                        value=float(fraction),
                        threshold=self.config.representation_min_fraction,
                        is_biased=True,
                        sensitive_feature=sensitive_feature,
                        details={
                            "group": str(group),
                            "fraction": round(float(fraction), 4),
                            "count": int(counts[group] * len(df)),
                        },
                    )
                )

        # Overall imbalance: max/min ratio
        fractions = counts.values
        if len(fractions) >= 2 and min(fractions) > 0:
            imbalance = float(max(fractions) / min(fractions))
            metrics.append(
                BiasMetric(
                    name="representation_imbalance",
                    value=imbalance,
                    threshold=self.config.outcome_disparity_threshold,
                    is_biased=imbalance > self.config.outcome_disparity_threshold,
                    sensitive_feature=sensitive_feature,
                    details={
                        "max_group": str(counts.index[0]),
                        "min_group": str(counts.index[-1]),
                        "max_fraction": round(float(max(fractions)), 4),
                        "min_fraction": round(float(min(fractions)), 4),
                    },
                )
            )

        return metrics

    # ------------------------------------------------------------------
    # Outcome disparity (no model needed)
    # ------------------------------------------------------------------

    def check_outcome_disparity(
        self,
        df: pd.DataFrame,
        sensitive_feature: str,
        outcome_column: str,
    ) -> List[BiasMetric]:
        """Check if a numeric outcome differs significantly across groups.

        This is a data-level check — no trained model required.
        Computes per-group means and flags if the ratio between
        the highest and lowest group exceeds threshold.
        """
        if sensitive_feature not in df.columns or outcome_column not in df.columns:
            return []

        group_means = df.groupby(sensitive_feature)[outcome_column].mean()
        if len(group_means) < 2:
            return []

        max_mean = float(group_means.max())
        min_mean = float(group_means.min())

        if min_mean <= 0:
            ratio = float("inf") if max_mean > 0 else 1.0
        else:
            ratio = max_mean / min_mean

        is_biased = ratio > self.config.outcome_disparity_threshold

        return [
            BiasMetric(
                name="outcome_disparity",
                value=ratio,
                threshold=self.config.outcome_disparity_threshold,
                is_biased=is_biased,
                sensitive_feature=sensitive_feature,
                details={
                    "outcome_column": outcome_column,
                    "group_means": {
                        str(k): round(float(v), 4) for k, v in group_means.items()
                    },
                    "best_group": str(group_means.idxmax()),
                    "worst_group": str(group_means.idxmin()),
                    "max_mean": round(max_mean, 4),
                    "min_mean": round(min_mean, 4),
                },
            )
        ]

    # ------------------------------------------------------------------
    # Group metric differences
    # ------------------------------------------------------------------

    def check_group_metric_difference(
        self,
        df: pd.DataFrame,
        sensitive_feature: str,
        metric_column: str,
        metric_fn: Optional[Callable] = None,
    ) -> List[BiasMetric]:
        """Check if a metric differs across groups beyond threshold.

        By default uses group means. Pass a custom *metric_fn* that
        takes a Series and returns a scalar for other aggregations.
        """
        if sensitive_feature not in df.columns or metric_column not in df.columns:
            return []

        def _default_mean(s):
            return float(s.mean())

        if metric_fn is None:
            metric_fn = _default_mean

        group_values = df.groupby(sensitive_feature)[metric_column].agg(metric_fn)
        if len(group_values) < 2:
            return []

        overall = metric_fn(df[metric_column])
        max_diff = 0.0
        worst_group = None

        for group, val in group_values.items():
            diff = abs(val - overall) / max(abs(overall), 1e-10)
            if diff > max_diff:
                max_diff = diff
                worst_group = group

        is_biased = max_diff > self.config.group_metric_diff_threshold

        return [
            BiasMetric(
                name="group_metric_difference",
                value=max_diff,
                threshold=self.config.group_metric_diff_threshold,
                is_biased=is_biased,
                sensitive_feature=sensitive_feature,
                details={
                    "metric_column": metric_column,
                    "overall_value": round(float(overall), 4),
                    "group_values": {
                        str(k): round(float(v), 4) for k, v in group_values.items()
                    },
                    "worst_group": str(worst_group),
                    "max_relative_diff": round(max_diff, 4),
                },
            )
        ]

    # ------------------------------------------------------------------
    # Fairlearn: demographic parity
    # ------------------------------------------------------------------

    def check_demographic_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        feature_name: str = "sensitive",
    ) -> List[BiasMetric]:
        """Compute demographic parity difference using Fairlearn.

        Requires binary y_true/y_pred (e.g., is_premium_card recommended).
        """
        if not FAIRLEARN_AVAILABLE:
            logger.warning("Fairlearn not available — skipping demographic parity")
            return []

        dpd = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        is_biased = bool(abs(dpd) > self.config.demographic_parity_threshold)

        mf = MetricFrame(
            metrics={"selection_rate": lambda y, p: np.mean(p)},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        return [
            BiasMetric(
                name="demographic_parity",
                value=abs(float(dpd)),
                threshold=self.config.demographic_parity_threshold,
                is_biased=is_biased,
                sensitive_feature=feature_name,
                details={
                    "dpd": round(float(dpd), 6),
                    "group_rates": {
                        str(k): round(float(v), 4)
                        for k, v in mf.by_group["selection_rate"].items()
                    },
                },
            )
        ]

    # ------------------------------------------------------------------
    # Fairlearn: equalized odds
    # ------------------------------------------------------------------

    def check_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        feature_name: str = "sensitive",
    ) -> List[BiasMetric]:
        """Compute equalized odds difference using Fairlearn."""
        if not FAIRLEARN_AVAILABLE:
            logger.warning("Fairlearn not available — skipping equalized odds")
            return []

        eod = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        is_biased = bool(abs(eod) > self.config.equalized_odds_threshold)

        return [
            BiasMetric(
                name="equalized_odds",
                value=abs(float(eod)),
                threshold=self.config.equalized_odds_threshold,
                is_biased=is_biased,
                sensitive_feature=feature_name,
                details={"eod": round(float(eod), 6)},
            )
        ]

    # ------------------------------------------------------------------
    # Full bias scan (data-level, no model required)
    # ------------------------------------------------------------------

    def run_data_bias_scan(
        self,
        df: pd.DataFrame,
        sensitive_features: List[str],
        outcome_columns: Optional[List[str]] = None,
        dataset: str = "",
    ) -> BiasReport:
        """Run all data-level bias checks across multiple features/outcomes.

        This does NOT require a trained model — it checks for
        data-level representation and outcome disparities.
        """
        logger.info(
            "[%s] Running data bias scan: %d features, %d outcomes",
            dataset,
            len(sensitive_features),
            len(outcome_columns or []),
        )

        all_metrics: List[BiasMetric] = []

        for sf in sensitive_features:
            # Representation check
            all_metrics.extend(self.check_representation(df, sf))

            # Outcome disparity for each outcome column
            for oc in outcome_columns or []:
                all_metrics.extend(self.check_outcome_disparity(df, sf, oc))

            # Group metric difference
            for oc in outcome_columns or []:
                all_metrics.extend(self.check_group_metric_difference(df, sf, oc))

        report = BiasReport(
            dataset=dataset,
            sensitive_features_checked=sensitive_features,
            metrics=all_metrics,
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "fairlearn_available": FAIRLEARN_AVAILABLE,
            },
        )

        logger.info(
            "[%s] Bias scan complete: %d metrics, %d biased",
            dataset,
            len(all_metrics),
            len(report.biased_metrics),
        )
        return report
