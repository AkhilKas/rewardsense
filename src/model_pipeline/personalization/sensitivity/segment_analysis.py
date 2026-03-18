"""
Segment-level feature importance analysis.

Breaks down SHAP-based feature importance by user segments
(e.g. high-spender vs low-spender, single-card vs multi-card)
to reveal whether different user groups rely on different features.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


def define_segments(meta: pd.DataFrame) -> Dict[str, pd.Series]:
    """Auto-detect segment columns and return boolean masks.

    Built-in segments:
    - **spending_tier** — median-split on ``total_spending``
    - **card_count** — single-card vs multi-card on ``num_cards``

    Also picks up any categorical columns from the default segment set
    (``archetype``, ``age_group``, ``budget_quartile``, ``location_type``).

    Returns
    -------
    dict mapping ``"segment_name::value"`` → boolean index mask.
    """
    segments: Dict[str, pd.Series] = {}

    if "total_spending" in meta.columns:
        median_spend = meta["total_spending"].median()
        segments["spending_tier::high_spender"] = meta["total_spending"] >= median_spend
        segments["spending_tier::low_spender"] = meta["total_spending"] < median_spend

    if "num_cards" in meta.columns:
        segments["card_count::single_card"] = meta["num_cards"] == 1
        segments["card_count::multi_card"] = meta["num_cards"] > 1

    categorical_cols = ["archetype", "age_group", "budget_quartile", "location_type"]
    for col in categorical_cols:
        if col in meta.columns:
            for val in meta[col].dropna().unique():
                segments[f"{col}::{val}"] = meta[col] == val

    return segments


class SegmentAnalyzer:
    """Compute per-segment SHAP importance and compare across groups.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP matrix (n_samples × n_features).
    X : pd.DataFrame
        Feature matrix (same order as shap_values rows).
    meta : pd.DataFrame
        Metadata frame (same index / row-order as X) containing segment
        columns like ``total_spending``, ``num_cards``, ``archetype``, etc.
    feature_names : list of str or None
        Column names for features.
    """

    def __init__(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        meta: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.shap_values = np.asarray(shap_values)
        self.X = X.reset_index(drop=True)
        self.meta = meta.reset_index(drop=True)
        self.feature_names = feature_names or list(X.columns)
        self._segments = define_segments(self.meta)

    @property
    def segment_names(self) -> List[str]:
        return list(self._segments.keys())

    def importance_for_segment(self, segment_key: str) -> pd.DataFrame:
        """Compute mean |SHAP| for samples matching a segment mask.

        Parameters
        ----------
        segment_key : str
            Key in the form ``"dimension::value"`` (e.g. ``"spending_tier::high_spender"``).

        Returns
        -------
        pd.DataFrame with ``feature``, ``mean_abs_shap``, ``rank``.
        """
        mask = self._segments[segment_key]
        sv_seg = self.shap_values[mask.values]
        if len(sv_seg) == 0:
            return pd.DataFrame(columns=["feature", "mean_abs_shap", "rank"])

        mean_abs = np.abs(sv_seg).mean(axis=0)
        df = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def analyze_all_segments(self) -> Dict[str, pd.DataFrame]:
        """Compute importance for every detected segment.

        Returns
        -------
        dict mapping segment_key → importance DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}
        for key in self._segments:
            count = int(self._segments[key].sum())
            if count < 2:
                logger.warning("Segment '{}' has <2 samples, skipping", key)
                continue
            results[key] = self.importance_for_segment(key)
            logger.debug(
                "Segment '{}': {} samples, top feature='{}'",
                key,
                count,
                results[key].iloc[0]["feature"] if len(results[key]) else "N/A",
            )
        logger.info("Analyzed {} segments", len(results))
        return results

    def compare_segments(
        self,
        segment_results: Dict[str, pd.DataFrame],
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Build a comparison table of top-N features per segment.

        Returns a DataFrame with one row per segment and columns
        ``segment``, ``top_1`` … ``top_N``.
        """
        rows: List[Dict[str, str]] = []
        for key, imp in segment_results.items():
            row: Dict[str, str] = {"segment": key}
            top_feats = imp.head(top_n)["feature"].tolist()
            for i, feat in enumerate(top_feats, start=1):
                row[f"top_{i}"] = feat
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def generate_segment_comparison_plot(
        self,
        segment_results: Dict[str, pd.DataFrame],
        top_n: int = 10,
    ) -> plt.Figure:
        """Grouped bar chart comparing top feature importance across segments."""
        all_top_features: List[str] = []
        for imp in segment_results.values():
            all_top_features.extend(imp.head(top_n)["feature"].tolist())
        unique_features = list(dict.fromkeys(all_top_features))[:top_n]

        data: Dict[str, List[float]] = {}
        for key, imp in segment_results.items():
            label = key.split("::")[-1] if "::" in key else key
            lookup = dict(zip(imp["feature"], imp["mean_abs_shap"]))
            data[label] = [lookup.get(f, 0.0) for f in unique_features]

        x = np.arange(len(unique_features))
        width = 0.8 / max(len(data), 1)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (label, vals) in enumerate(data.items()):
            ax.bar(x + i * width, vals, width, label=label)

        ax.set_xticks(x + width * (len(data) - 1) / 2)
        ax.set_xticklabels(unique_features, rotation=45, ha="right")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title("Feature Importance by Segment")
        ax.legend(fontsize=8)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        segment_results: Dict[str, pd.DataFrame],
        tracker: Optional[Any] = None,
        artifact_subdir: str = "sensitivity/segments",
    ) -> None:
        """Save segment analysis artifacts and log to MLflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            comparison = self.compare_segments(segment_results)
            comparison.to_csv(tmp / "segment_comparison.csv", index=False)

            for key, imp in segment_results.items():
                safe = key.replace("::", "__").replace("/", "_")
                imp.to_csv(tmp / f"importance_{safe}.csv", index=False)

            try:
                fig = self.generate_segment_comparison_plot(segment_results)
                fig.savefig(
                    tmp / "segment_comparison.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig)
            except Exception as exc:
                logger.warning("Segment comparison plot failed: {}", exc)

            if tracker is not None:
                tracker.log_artifacts(str(tmp), artifact_path=artifact_subdir)
                logger.info("Logged segment analysis artifacts to MLflow")
