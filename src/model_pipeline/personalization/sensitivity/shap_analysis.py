"""
SHAP-based feature importance analysis for the personalization model.

Provides global importance (mean |SHAP|), dependence analysis, and
force-plot explanations.  Supports tree-based models via TreeExplainer
and falls back to KernelExplainer for other estimators.
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

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False


def _is_tree_model(model: Any) -> bool:
    """Return True if the model is a tree-based estimator."""
    tree_types = (
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor",
        "LGBMRegressor",
    )
    return type(model).__name__ in tree_types


class SHAPAnalyzer:
    """Compute and visualise SHAP values for a fitted regression model.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    X : pd.DataFrame
        Feature matrix used for explanation (typically validation set).
    feature_names : list of str or None
        Override column names. Defaults to ``X.columns``.
    """

    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if not SHAP_AVAILABLE:
            raise ImportError(
                "shap is required for SHAPAnalyzer. Install with: pip install shap"
            )
        self.model = model
        self.X = X.copy()
        self.feature_names = feature_names or list(X.columns)
        self._shap_values: Optional[np.ndarray] = None
        self._explainer: Optional[Any] = None

    def compute_shap_values(self) -> np.ndarray:
        """Compute SHAP values for every sample in *X*.

        Uses TreeExplainer for tree models, KernelExplainer otherwise.
        Caches the result for subsequent calls.
        """
        if self._shap_values is not None:
            return self._shap_values

        if _is_tree_model(self.model):
            self._explainer = shap.TreeExplainer(self.model)
            self._shap_values = self._explainer.shap_values(self.X)
        else:
            background = shap.sample(self.X, min(50, len(self.X)))
            self._explainer = shap.KernelExplainer(self.model.predict, background)
            self._shap_values = self._explainer.shap_values(self.X, nsamples=100)

        self._shap_values = np.asarray(self._shap_values)
        logger.info(
            "Computed SHAP values: shape {}",
            self._shap_values.shape,
        )
        return self._shap_values

    def global_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of features ranked by mean |SHAP| value."""
        sv = self.compute_shap_values()
        mean_abs = np.abs(sv).mean(axis=0)

        importance = pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": mean_abs}
        )
        importance = importance.sort_values("mean_abs_shap", ascending=False)
        importance = importance.reset_index(drop=True)
        importance["rank"] = importance.index + 1
        return importance

    def top_features(self, n: int = 10) -> List[str]:
        """Return the top-N feature names by global importance."""
        imp = self.global_feature_importance()
        return imp["feature"].head(n).tolist()

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def generate_summary_plot(self, max_display: int = 20) -> plt.Figure:
        """Generate a SHAP beeswarm summary plot."""
        sv = self.compute_shap_values()
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv,
            self.X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        fig = plt.gcf()
        plt.tight_layout()
        return fig

    def generate_bar_plot(self, max_display: int = 20) -> plt.Figure:
        """Generate a bar chart of mean |SHAP| values."""
        imp = self.global_feature_importance().head(max_display)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        return fig

    def generate_dependence_plot(
        self, feature: str, interaction_feature: Optional[str] = None
    ) -> plt.Figure:
        """Generate a SHAP dependence plot for a single feature."""
        sv = self.compute_shap_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            sv,
            self.X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            ax=ax,
            show=False,
        )
        plt.tight_layout()
        return fig

    def generate_force_plot_data(self, idx: int) -> Dict[str, Any]:
        """Return force-plot data for a single sample as a dict.

        Avoids rendering issues with SHAP's JS-based force_plot by
        returning raw numeric data instead.
        """
        sv = self.compute_shap_values()
        ev = self._explainer.expected_value
        base_value = float(ev.item()) if hasattr(ev, "item") else float(ev)
        sample_shap = sv[idx]
        sample_features = self.X.iloc[idx].to_dict()

        return {
            "base_value": base_value,
            "shap_values": dict(zip(self.feature_names, sample_shap.tolist())),
            "feature_values": sample_features,
            "predicted_value": base_value + float(np.sum(sample_shap)),
        }

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        tracker: Optional[Any] = None,
        artifact_subdir: str = "sensitivity/shap",
    ) -> None:
        """Save all SHAP artifacts to a temp dir and log via MLflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            imp = self.global_feature_importance()
            imp.to_csv(tmp / "shap_feature_importance.csv", index=False)

            try:
                fig = self.generate_summary_plot()
                fig.savefig(tmp / "shap_summary.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                logger.warning("Summary plot failed: {}", exc)

            try:
                fig = self.generate_bar_plot()
                fig.savefig(tmp / "shap_bar.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                logger.warning("Bar plot failed: {}", exc)

            for feat in self.top_features(5):
                try:
                    fig = self.generate_dependence_plot(feat)
                    safe = feat.replace("/", "_")
                    fig.savefig(
                        tmp / f"shap_dep_{safe}.png", dpi=150, bbox_inches="tight"
                    )
                    plt.close(fig)
                except Exception as exc:
                    logger.warning("Dependence plot for '{}' failed: {}", feat, exc)

            if tracker is not None:
                tracker.log_artifacts(str(tmp), artifact_path=artifact_subdir)
                logger.info("Logged SHAP artifacts to MLflow")
            else:
                logger.info("SHAP artifacts saved to {}", tmpdir)
