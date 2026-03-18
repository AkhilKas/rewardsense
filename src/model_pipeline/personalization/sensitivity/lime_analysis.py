"""
LIME-based local explanation analysis for the personalization model.

Generates per-instance explanations and compares feature-importance
rankings with SHAP for consistency validation.
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
from scipy import stats

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    lime = None  # type: ignore[assignment]
    LIME_AVAILABLE = False


class LIMEAnalyzer:
    """Generate LIME explanations for a fitted regression model.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
        Must expose a ``predict`` method.
    X_train : pd.DataFrame
        Training data used to build the LIME explainer background.
    feature_names : list of str or None
        Override column names.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if not LIME_AVAILABLE:
            raise ImportError(
                "lime is required for LIMEAnalyzer. Install with: pip install lime"
            )
        self.model = model
        self.X_train = X_train.copy()
        self.feature_names = feature_names or list(X_train.columns)
        self._explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            mode="regression",
            verbose=False,
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """Explain a single prediction.

        Parameters
        ----------
        instance : 1-D array
            Feature vector for one sample.
        num_features : int
            How many features to include in the explanation.

        Returns
        -------
        dict with keys ``feature_weights``, ``predicted_value``, ``intercept``.
        """
        exp = self._explainer.explain_instance(
            instance,
            self.model.predict,
            num_features=num_features,
        )
        weights = {feat: weight for feat, weight in exp.as_list()}
        return {
            "feature_weights": weights,
            "predicted_value": (
                float(exp.predicted_value) if hasattr(exp, "predicted_value") else None
            ),
            "intercept": float(exp.intercept[0]) if hasattr(exp, "intercept") else None,
            "score": float(exp.score) if hasattr(exp, "score") else None,
        }

    def explain_batch(
        self,
        X: pd.DataFrame,
        num_features: int = 10,
        max_samples: int = 50,
    ) -> List[Dict[str, Any]]:
        """Explain multiple instances and aggregate importance.

        Parameters
        ----------
        X : pd.DataFrame
            Samples to explain.
        num_features : int
            Features per explanation.
        max_samples : int
            Cap the number of samples to explain (LIME is slow).

        Returns
        -------
        list of explanation dicts
        """
        n = min(len(X), max_samples)
        explanations: List[Dict[str, Any]] = []
        for i in range(n):
            row = X.iloc[i].values
            exp = self.explain_instance(row, num_features=num_features)
            explanations.append(exp)
        logger.info("Generated LIME explanations for {} samples", n)
        return explanations

    def aggregate_importance(
        self,
        explanations: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Aggregate LIME weights across multiple explanations.

        Returns a DataFrame with columns ``feature``, ``mean_abs_weight``,
        sorted descending.
        """
        all_weights: Dict[str, List[float]] = {}
        for exp in explanations:
            for feat, weight in exp["feature_weights"].items():
                all_weights.setdefault(feat, []).append(abs(weight))

        rows = [
            {"feature": feat, "mean_abs_weight": float(np.mean(ws))}
            for feat, ws in all_weights.items()
        ]
        df = pd.DataFrame(rows).sort_values("mean_abs_weight", ascending=False)
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def compare_with_shap(
        self,
        shap_importance: pd.DataFrame,
        lime_importance: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute rank-correlation between SHAP and LIME importance.

        Both DataFrames must have a ``feature`` and a ``rank`` column.

        Returns
        -------
        dict with ``spearman_rho``, ``p_value``, ``top_5_overlap_pct``.
        """
        merged = shap_importance[["feature", "rank"]].merge(
            lime_importance[["feature", "rank"]],
            on="feature",
            suffixes=("_shap", "_lime"),
        )
        if merged.empty:
            return {"spearman_rho": None, "p_value": None, "top_5_overlap_pct": 0.0}

        rho, p = stats.spearmanr(merged["rank_shap"], merged["rank_lime"])

        shap_top5 = set(shap_importance.head(5)["feature"])
        lime_top5 = set(lime_importance.head(5)["feature"])
        overlap = len(shap_top5 & lime_top5) / 5.0 * 100

        result = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "top_5_overlap_pct": round(overlap, 1),
            "n_features_compared": len(merged),
        }
        logger.info(
            "SHAP vs LIME consistency — rho={}, top-5 overlap={}%",
            result["spearman_rho"],
            result["top_5_overlap_pct"],
        )
        return result

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def generate_importance_plot(
        self, importance: pd.DataFrame, max_display: int = 15
    ) -> plt.Figure:
        """Bar chart of aggregated LIME importance."""
        df = importance.head(max_display)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df["feature"][::-1], df["mean_abs_weight"][::-1])
        ax.set_xlabel("Mean |LIME weight|")
        ax.set_title("Feature Importance (LIME)")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        importance: pd.DataFrame,
        consistency: Optional[Dict[str, Any]] = None,
        tracker: Optional[Any] = None,
        artifact_subdir: str = "sensitivity/lime",
    ) -> None:
        """Save LIME artifacts and log via MLflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            importance.to_csv(tmp / "lime_feature_importance.csv", index=False)

            if consistency:
                import json

                (tmp / "shap_lime_consistency.json").write_text(
                    json.dumps(consistency, indent=2)
                )

            try:
                fig = self.generate_importance_plot(importance)
                fig.savefig(tmp / "lime_importance.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                logger.warning("LIME importance plot failed: {}", exc)

            if tracker is not None:
                tracker.log_artifacts(str(tmp), artifact_path=artifact_subdir)
                logger.info("Logged LIME artifacts to MLflow")
