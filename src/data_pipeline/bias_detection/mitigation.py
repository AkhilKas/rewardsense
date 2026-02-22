"""
Bias Mitigation Strategies

Implements three mitigation strategies:
  1. Resampling (oversampling minority / undersampling majority)
  2. Feature reweighting (sample weights inversely proportional to group size)
  3. Threshold adjustment recommendations

Each strategy returns a before/after comparison so trade-offs are explicitly documented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MitigationResult:
    """Result of applying a single mitigation strategy."""

    strategy: str
    description: str
    before_distribution: Dict[str, float]
    after_distribution: Dict[str, float]
    before_imbalance_ratio: float
    after_imbalance_ratio: float
    rows_before: int
    rows_after: int
    trade_offs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """Ratio improvement: 1.0 means no change, lower is better."""
        if self.before_imbalance_ratio <= 0:
            return 1.0
        return self.after_imbalance_ratio / self.before_imbalance_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "description": self.description,
            "before_distribution": {
                k: round(v, 4) for k, v in self.before_distribution.items()
            },
            "after_distribution": {
                k: round(v, 4) for k, v in self.after_distribution.items()
            },
            "before_imbalance_ratio": round(self.before_imbalance_ratio, 4),
            "after_imbalance_ratio": round(self.after_imbalance_ratio, 4),
            "improvement": round(self.improvement, 4),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "trade_offs": self.trade_offs,
            "metadata": self.metadata,
        }


class BiasMitigator:
    """Apply mitigation strategies to reduce data bias.

    All methods return a ``MitigationResult`` with before/after
    distributions and documented trade-offs.
    """

    def __init__(self) -> None:
        logger.info("BiasMitigator initialized")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_distribution(df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Compute normalized distribution of a column."""
        counts = df[column].value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}

    @staticmethod
    def _imbalance_ratio(dist: Dict[str, float]) -> float:
        """Compute max/min ratio from a distribution dict."""
        vals = [v for v in dist.values() if v > 0]
        if len(vals) < 2:
            return 1.0
        return max(vals) / min(vals)

    # ------------------------------------------------------------------
    # Strategy 1: Resampling
    # ------------------------------------------------------------------

    def resample_oversample(
        self,
        df: pd.DataFrame,
        sensitive_column: str,
        target_ratio: Optional[float] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, MitigationResult]:
        """Oversample minority groups to match the majority group size.

        Parameters
        ----------
        target_ratio : float, optional
            If set, oversample minorities to this fraction of the majority.
            Default (None) oversamples to equal counts.
        random_state : int
            Seed for reproducibility.
        """
        before_dist = self._get_distribution(df, sensitive_column)
        before_imbalance = self._imbalance_ratio(before_dist)

        rng = np.random.RandomState(random_state)
        groups = df.groupby(sensitive_column)
        max_count = groups.size().max()
        target_count = (
            max_count if target_ratio is None else int(max_count * target_ratio)
        )

        resampled_parts: List[pd.DataFrame] = []
        for _, group in groups:
            if len(group) >= target_count:
                resampled_parts.append(group)
            else:
                n_extra = target_count - len(group)
                extra = group.sample(n=n_extra, replace=True, random_state=rng)
                resampled_parts.append(pd.concat([group, extra], ignore_index=True))

        result_df = pd.concat(resampled_parts, ignore_index=True)
        after_dist = self._get_distribution(result_df, sensitive_column)

        return result_df, MitigationResult(
            strategy="oversample",
            description=(
                f"Oversampled minority groups in '{sensitive_column}' to match "
                f"majority group size ({target_count} per group)"
            ),
            before_distribution=before_dist,
            after_distribution=after_dist,
            before_imbalance_ratio=before_imbalance,
            after_imbalance_ratio=self._imbalance_ratio(after_dist),
            rows_before=len(df),
            rows_after=len(result_df),
            trade_offs=[
                "Increases dataset size (may increase training time)",
                "Duplicate rows may cause overfitting on minority patterns",
                "Does not add new information — only replicates existing samples",
            ],
        )

    def resample_undersample(
        self,
        df: pd.DataFrame,
        sensitive_column: str,
        target_ratio: Optional[float] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, MitigationResult]:
        """Undersample majority groups to match the minority group size.

        Parameters
        ----------
        target_ratio : float, optional
            If set, undersample majority to this multiple of the minority.
            Default (None) undersamples to equal counts.
        """
        before_dist = self._get_distribution(df, sensitive_column)
        before_imbalance = self._imbalance_ratio(before_dist)

        rng = np.random.RandomState(random_state)
        groups = df.groupby(sensitive_column)
        min_count = groups.size().min()
        target_count = (
            min_count if target_ratio is None else int(min_count * target_ratio)
        )
        target_count = max(target_count, 1)

        resampled_parts: List[pd.DataFrame] = []
        for _, group in groups:
            if len(group) <= target_count:
                resampled_parts.append(group)
            else:
                resampled_parts.append(
                    group.sample(n=target_count, replace=False, random_state=rng)
                )

        result_df = pd.concat(resampled_parts, ignore_index=True)
        after_dist = self._get_distribution(result_df, sensitive_column)

        return result_df, MitigationResult(
            strategy="undersample",
            description=(
                f"Undersampled majority groups in '{sensitive_column}' to match "
                f"minority group size ({target_count} per group)"
            ),
            before_distribution=before_dist,
            after_distribution=after_dist,
            before_imbalance_ratio=before_imbalance,
            after_imbalance_ratio=self._imbalance_ratio(after_dist),
            rows_before=len(df),
            rows_after=len(result_df),
            trade_offs=[
                "Reduces dataset size (loses information from majority groups)",
                "May reduce model performance if majority group data was informative",
                "Balances representation but doesn't address feature-level bias",
            ],
        )

    # ------------------------------------------------------------------
    # Strategy 2: Feature reweighting
    # ------------------------------------------------------------------

    def compute_sample_weights(
        self,
        df: pd.DataFrame,
        sensitive_column: str,
    ) -> Tuple[np.ndarray, MitigationResult]:
        """Compute sample weights inversely proportional to group frequency.

        Useful for cost-sensitive learning: pass weights to
        ``sample_weight`` in sklearn/XGBoost fit methods.

        Returns
        -------
        weights : ndarray of shape (n_samples,)
        result : MitigationResult
        """
        before_dist = self._get_distribution(df, sensitive_column)
        before_imbalance = self._imbalance_ratio(before_dist)

        group_counts = df[sensitive_column].value_counts()
        n = len(df)
        n_groups = len(group_counts)

        # Weight = n / (n_groups * count_of_this_group)
        weight_map = {
            group: n / (n_groups * count) for group, count in group_counts.items()
        }
        weights = df[sensitive_column].map(weight_map).to_numpy(dtype=float)

        # Normalize so mean weight = 1
        weights = weights / weights.mean()

        # Effective distribution (weighted)
        weighted_dist: Dict[str, float] = {}
        for group in group_counts.index:
            mask = df[sensitive_column] == group
            weighted_dist[str(group)] = float(weights[mask].sum() / weights.sum())

        return weights, MitigationResult(
            strategy="sample_weights",
            description=(
                f"Computed inverse-frequency sample weights on '{sensitive_column}' "
                f"({n_groups} groups)"
            ),
            before_distribution=before_dist,
            after_distribution=weighted_dist,
            before_imbalance_ratio=before_imbalance,
            after_imbalance_ratio=self._imbalance_ratio(weighted_dist),
            rows_before=n,
            rows_after=n,
            trade_offs=[
                "No data is added or removed — preserves all information",
                "Requires the model to support sample_weight parameter",
                "May increase variance if minority groups have noisy data",
            ],
            metadata={
                "weight_map": {str(k): round(v, 4) for k, v in weight_map.items()},
                "mean_weight": round(float(weights.mean()), 4),
                "min_weight": round(float(weights.min()), 4),
                "max_weight": round(float(weights.max()), 4),
            },
        )

    # ------------------------------------------------------------------
    # Strategy 3: Threshold adjustment recommendations
    # ------------------------------------------------------------------

    def recommend_threshold_adjustments(
        self,
        df: pd.DataFrame,
        sensitive_column: str,
        score_column: str,
        target_rate: Optional[float] = None,
    ) -> MitigationResult:
        """Recommend per-group decision thresholds to equalize outcomes.

        Given a continuous score column, computes the threshold per
        group that would yield equal positive rates across groups.

        Parameters
        ----------
        score_column : str
            Numeric column representing the model score / reward value.
        target_rate : float, optional
            Desired positive rate. Default uses the overall median.
        """
        before_dist = self._get_distribution(df, sensitive_column)
        before_imbalance = self._imbalance_ratio(before_dist)

        if target_rate is None:
            target_rate = 0.5  # median split

        group_thresholds: Dict[str, float] = {}
        group_rates_before: Dict[str, float] = {}
        overall_threshold = float(df[score_column].quantile(1 - target_rate))

        for group, gdf in df.groupby(sensitive_column):
            # Current positive rate at overall threshold
            rate = float((gdf[score_column] >= overall_threshold).mean())
            group_rates_before[str(group)] = round(rate, 4)

            # Per-group threshold to achieve target_rate
            group_threshold = float(gdf[score_column].quantile(1 - target_rate))
            group_thresholds[str(group)] = round(group_threshold, 4)

        return MitigationResult(
            strategy="threshold_adjustment",
            description=(
                f"Recommended per-group thresholds on '{score_column}' "
                f"to achieve {target_rate:.0%} positive rate per group"
            ),
            before_distribution=group_rates_before,
            after_distribution={g: round(target_rate, 4) for g in group_thresholds},
            before_imbalance_ratio=before_imbalance,
            after_imbalance_ratio=1.0,  # target is equal rates
            rows_before=len(df),
            rows_after=len(df),
            trade_offs=[
                "Different thresholds per group may raise fairness concerns",
                "Equalizes outcomes but may sacrifice overall accuracy",
                "Requires score column to be meaningful and calibrated",
            ],
            metadata={
                "overall_threshold": round(overall_threshold, 4),
                "group_thresholds": group_thresholds,
                "target_rate": target_rate,
            },
        )

    # ------------------------------------------------------------------
    # Run all strategies and compare
    # ------------------------------------------------------------------

    def run_all_strategies(
        self,
        df: pd.DataFrame,
        sensitive_column: str,
        score_column: Optional[str] = None,
        random_state: int = 42,
    ) -> Dict[str, MitigationResult]:
        """Run all three mitigation strategies and return comparison.

        Returns a dict keyed by strategy name.
        """
        logger.info(
            "Running all mitigation strategies on '%s' (%d rows)",
            sensitive_column,
            len(df),
        )

        results: Dict[str, MitigationResult] = {}

        # Oversampling
        _, oversample_result = self.resample_oversample(
            df, sensitive_column, random_state=random_state
        )
        results["oversample"] = oversample_result

        # Undersampling
        _, undersample_result = self.resample_undersample(
            df, sensitive_column, random_state=random_state
        )
        results["undersample"] = undersample_result

        # Sample weights
        _, weights_result = self.compute_sample_weights(df, sensitive_column)
        results["sample_weights"] = weights_result

        # Threshold adjustment (only if score column provided)
        if score_column and score_column in df.columns:
            threshold_result = self.recommend_threshold_adjustments(
                df, sensitive_column, score_column
            )
            results["threshold_adjustment"] = threshold_result

        logger.info("Completed %d mitigation strategies", len(results))
        return results
