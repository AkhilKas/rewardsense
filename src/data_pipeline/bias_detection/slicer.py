"""
Data Slicing Module

Build data slicing capabilities for bias analysis across demographic and categorical features.

The DataSlicer produces per-slice statistics so downstream bias metrics can compare outcomes across groups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SliceStats:
    """Statistics for a single data slice."""

    slice_column: str
    slice_value: Any
    count: int
    fraction: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_column": self.slice_column,
            "slice_value": str(self.slice_value),
            "count": self.count,
            "fraction": round(self.fraction, 4),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


@dataclass
class SliceReport:
    """Aggregated slicing report for one dimension."""

    column: str
    total_rows: int
    num_slices: int
    slices: List[SliceStats] = field(default_factory=list)
    imbalance_ratio: float = 0.0  # max_fraction / min_fraction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "total_rows": self.total_rows,
            "num_slices": self.num_slices,
            "imbalance_ratio": round(self.imbalance_ratio, 4),
            "slices": [s.to_dict() for s in self.slices],
        }


class DataSlicer:
    """Slice DataFrames by categorical or quantile-based features.

    Supports:
      - Categorical slicing (age_group, location_type, archetype, etc.)
      - Quantile-based slicing (transaction amounts, budgets)
      - Custom bin slicing (user-defined boundaries)
      - Per-slice metric computation on any numeric column
    """

    def __init__(self) -> None:
        logger.info("DataSlicer initialized")

    # ------------------------------------------------------------------
    # Categorical slicing
    # ------------------------------------------------------------------

    def slice_by_column(
        self,
        df: pd.DataFrame,
        column: str,
        metric_columns: Optional[List[str]] = None,
    ) -> SliceReport:
        """Slice *df* by unique values in *column* and compute per-slice stats.

        Parameters
        ----------
        df : DataFrame
            The dataset to slice.
        column : str
            Categorical column to group by.
        metric_columns : list of str, optional
            Numeric columns to compute mean/std/median per slice.

        Returns
        -------
        SliceReport
        """
        if column not in df.columns:
            logger.warning("Column '%s' not in DataFrame — skipping", column)
            return SliceReport(column=column, total_rows=len(df), num_slices=0)

        if metric_columns is None:
            metric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            metric_columns = [c for c in metric_columns if c != column]

        n = len(df)
        groups = df.groupby(column, dropna=False, observed=True)
        slices: List[SliceStats] = []

        for val, group in groups:
            count = len(group)
            metrics: Dict[str, float] = {}
            for mc in metric_columns:
                if mc in group.columns:
                    metrics[f"{mc}_mean"] = float(group[mc].mean())
                    metrics[f"{mc}_median"] = float(group[mc].median())
                    metrics[f"{mc}_std"] = float(group[mc].std())

            slices.append(
                SliceStats(
                    slice_column=column,
                    slice_value=val,
                    count=count,
                    fraction=count / max(n, 1),
                    metrics=metrics,
                )
            )

        # Imbalance ratio: largest group / smallest group
        fractions = [s.fraction for s in slices if s.fraction > 0]
        imbalance = max(fractions) / min(fractions) if len(fractions) >= 2 else 1.0

        report = SliceReport(
            column=column,
            total_rows=n,
            num_slices=len(slices),
            slices=slices,
            imbalance_ratio=imbalance,
        )

        logger.info(
            "[%s] %d slices, imbalance ratio: %.2f",
            column,
            len(slices),
            imbalance,
        )
        return report

    # ------------------------------------------------------------------
    # Quantile-based slicing
    # ------------------------------------------------------------------

    def slice_by_quantiles(
        self,
        df: pd.DataFrame,
        column: str,
        n_quantiles: int = 4,
        metric_columns: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> SliceReport:
        """Slice *df* by quantile bins of a numeric *column*.

        Parameters
        ----------
        n_quantiles : int
            Number of quantile bins (default 4 = quartiles).
        labels : list of str, optional
            Labels for each bin. If None, auto-generated.
        """
        if column not in df.columns:
            logger.warning("Column '%s' not in DataFrame — skipping", column)
            return SliceReport(column=column, total_rows=len(df), num_slices=0)

        if labels is None:
            labels = [f"Q{i + 1}" for i in range(n_quantiles)]

        bin_col = f"__{column}_quantile"
        df = df.copy()
        df[bin_col] = pd.qcut(
            df[column], q=n_quantiles, labels=labels, duplicates="drop"
        )

        report = self.slice_by_column(df, bin_col, metric_columns)
        # Override column name to be more descriptive
        report.column = f"{column}_quantile"
        for s in report.slices:
            s.slice_column = f"{column}_quantile"

        return report

    # ------------------------------------------------------------------
    # Custom bin slicing
    # ------------------------------------------------------------------

    def slice_by_bins(
        self,
        df: pd.DataFrame,
        column: str,
        bins: List[float],
        bin_labels: Optional[List[str]] = None,
        metric_columns: Optional[List[str]] = None,
    ) -> SliceReport:
        """Slice *df* by custom bin boundaries on a numeric *column*."""
        if column not in df.columns:
            logger.warning("Column '%s' not in DataFrame — skipping", column)
            return SliceReport(column=column, total_rows=len(df), num_slices=0)

        bin_col = f"__{column}_bin"
        df = df.copy()
        df[bin_col] = pd.cut(
            df[column], bins=bins, labels=bin_labels, include_lowest=True
        )

        report = self.slice_by_column(df, bin_col, metric_columns)
        report.column = f"{column}_binned"
        for s in report.slices:
            s.slice_column = f"{column}_binned"

        return report

    # ------------------------------------------------------------------
    # Multi-dimension slicing
    # ------------------------------------------------------------------

    def slice_all_dimensions(
        self,
        df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        quantile_columns: Optional[List[str]] = None,
        metric_columns: Optional[List[str]] = None,
        n_quantiles: int = 4,
    ) -> Dict[str, SliceReport]:
        """Slice across multiple dimensions and return all reports.

        Parameters
        ----------
        categorical_columns : list of str
            Columns to slice categorically.
        quantile_columns : list of str
            Numeric columns to slice by quantiles.
        metric_columns : list of str
            Numeric columns to compute stats on per slice.
        """
        if categorical_columns is None:
            categorical_columns = []
        if quantile_columns is None:
            quantile_columns = []

        reports: Dict[str, SliceReport] = {}

        for col in categorical_columns:
            reports[col] = self.slice_by_column(df, col, metric_columns)

        for col in quantile_columns:
            reports[f"{col}_quantile"] = self.slice_by_quantiles(
                df, col, n_quantiles, metric_columns
            )

        logger.info(
            "Sliced across %d dimensions (%d categorical, %d quantile)",
            len(reports),
            len(categorical_columns),
            len(quantile_columns),
        )
        return reports
