"""
Counterfactual Fairness Analysis.

What-if analysis: for each user, flip their demographic attributes
and check if the model recommendation changes. If changing a
protected attribute (e.g., age_group, location_type) significantly
alters the output, the model may be relying on that attribute
unfairly.

Usage:
    analyzer = CounterfactualAnalyzer(model=trained_model)

    # Single user
    result = analyzer.analyze_user(
        user_features=feature_vector,
        sensitive_columns=["age_group", "location_type"],
    )

    # Batch analysis
    report = analyzer.analyze_batch(
        X=X_test,
        sensitive_columns=["age_group", "location_type"],
    )
    print(report.summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass
class CounterfactualResult:
    """Result for a single user across all counterfactual flips."""

    user_index: int
    original_prediction: float
    flips: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def max_change(self) -> float:
        if not self.flips:
            return 0.0
        return max(abs(f["prediction_change"]) for f in self.flips)

    @property
    def is_sensitive(self) -> bool:
        """True if any flip causes a significant prediction change."""
        return self.max_change > 0.05  # >5% change


@dataclass
class CounterfactualReport:
    """Aggregated counterfactual analysis across a dataset."""

    n_users: int
    sensitive_columns: List[str]
    results: List[CounterfactualResult] = field(default_factory=list)
    flip_threshold: float = 0.05

    @property
    def sensitive_users(self) -> List[CounterfactualResult]:
        return [r for r in self.results if r.max_change > self.flip_threshold]

    @property
    def sensitivity_rate(self) -> float:
        if self.n_users == 0:
            return 0.0
        return len(self.sensitive_users) / self.n_users

    @property
    def per_feature_sensitivity(self) -> Dict[str, float]:
        """Fraction of users affected by flipping each feature."""
        counts: Dict[str, int] = {col: 0 for col in self.sensitive_columns}
        for r in self.results:
            for flip in r.flips:
                if abs(flip["prediction_change"]) > self.flip_threshold:
                    counts[flip["feature"]] = counts.get(flip["feature"], 0) + 1
        return {
            col: count / self.n_users if self.n_users > 0 else 0.0
            for col, count in counts.items()
        }

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "n_users": self.n_users,
            "sensitive_users": len(self.sensitive_users),
            "sensitivity_rate": self.sensitivity_rate,
            "per_feature_sensitivity": self.per_feature_sensitivity,
            "flip_threshold": self.flip_threshold,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "results": [
                {
                    "user_index": r.user_index,
                    "original_prediction": r.original_prediction,
                    "max_change": r.max_change,
                    "is_sensitive": r.is_sensitive,
                    "flips": r.flips,
                }
                for r in self.results
                if r.max_change > self.flip_threshold  # only include affected
            ],
        }

    def log_to_mlflow(self, tracker: Any) -> None:
        """Log counterfactual report to MLflow."""
        if tracker is None:
            return
        tracker.log_metrics(
            {
                "cf_sensitivity_rate": self.sensitivity_rate,
                "cf_sensitive_users": len(self.sensitive_users),
                "cf_total_users": self.n_users,
            }
        )
        for feat, rate in self.per_feature_sensitivity.items():
            tracker.log_metric(f"cf_sensitivity_{feat}", rate)
        tracker.log_dict(self.to_dict(), "counterfactual_report.json")

        # Visualization
        try:
            self._log_chart(tracker)
        except ImportError:
            pass

    def _log_chart(self, tracker: Any) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        from src.model_pipeline.bias.visualizations import COLORS

        per_feat = self.per_feature_sensitivity
        if not per_feat:
            return

        features = list(per_feat.keys())
        rates = [per_feat[f] for f in features]

        fig, ax = _plt.subplots(figsize=(8, 4))
        bars = ax.barh(features, rates, color=COLORS[0], alpha=0.85)

        for bar, rate in zip(bars, rates):
            ax.text(
                rate + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}",
                va="center",
                fontsize=9,
            )

        ax.axvline(
            x=self.flip_threshold,
            color="#D55E00",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({self.flip_threshold:.0%})",
        )
        ax.set_xlabel("Fraction of Users Affected")
        ax.set_title("Counterfactual Sensitivity by Feature")
        ax.legend(fontsize=8)
        ax.set_xlim(0, max(rates + [self.flip_threshold]) * 1.3)
        fig.tight_layout()

        tracker.log_figure(fig, "counterfactual_sensitivity.png")
        _plt.close(fig)


# =====================================================================
# CounterfactualAnalyzer
# =====================================================================


class CounterfactualAnalyzer:
    """
    Counterfactual fairness analysis for any sklearn-compatible model.

    For each user, iterates over sensitive columns, replaces the value
    with each alternative, re-predicts, and measures the change.

    Parameters
    ----------
    predict_fn : Callable
        Function that takes a DataFrame/array and returns predictions.
        Can be ``model.predict``, ``model.predict_proba[:, 1]``, or
        a custom function.
    flip_threshold : float
        Minimum prediction change to flag as sensitive. Default 0.05.
    """

    def __init__(
        self,
        predict_fn: Optional[Callable] = None,
        model: Optional[Any] = None,
        flip_threshold: float = 0.05,
    ) -> None:
        if predict_fn is not None:
            self._predict = predict_fn
        elif model is not None:
            if hasattr(model, "predict_proba"):
                self._predict = lambda X: model.predict_proba(X)[:, 1]
            else:
                self._predict = model.predict
        else:
            raise ValueError("Provide either predict_fn or model")

        self.flip_threshold = flip_threshold

    # ------------------------------------------------------------------
    # Single user analysis
    # ------------------------------------------------------------------

    def analyze_user(
        self,
        X: pd.DataFrame,
        user_index: int,
        sensitive_columns: List[str],
        alternative_values: Optional[Dict[str, List[Any]]] = None,
    ) -> CounterfactualResult:
        """
        Analyze one user across all counterfactual flips.

        Parameters
        ----------
        X : DataFrame
            Full feature matrix (needed for column structure).
        user_index : int
            Row index of the user to analyze.
        sensitive_columns : list[str]
            Columns to flip.
        alternative_values : dict, optional
            Maps column → list of alternative values. If None,
            uses all unique values in the column.
        """
        user_row = X.iloc[[user_index]].copy()
        original_pred = float(self._predict(user_row)[0])

        result = CounterfactualResult(
            user_index=user_index,
            original_prediction=original_pred,
        )

        for col in sensitive_columns:
            if col not in X.columns:
                logger.warning("Column '%s' not in data — skipping", col)
                continue

            original_val = user_row[col].iloc[0]

            # Determine alternatives
            if alternative_values and col in alternative_values:
                alternatives = alternative_values[col]
            else:
                alternatives = X[col].dropna().unique().tolist()

            for alt_val in alternatives:
                if alt_val == original_val:
                    continue

                # Create counterfactual
                cf_row = user_row.copy()
                cf_row[col] = alt_val
                cf_pred = float(self._predict(cf_row)[0])

                result.flips.append(
                    {
                        "feature": col,
                        "original_value": str(original_val),
                        "counterfactual_value": str(alt_val),
                        "original_prediction": original_pred,
                        "counterfactual_prediction": cf_pred,
                        "prediction_change": cf_pred - original_pred,
                        "absolute_change": abs(cf_pred - original_pred),
                    }
                )

        return result

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        X: pd.DataFrame,
        sensitive_columns: List[str],
        alternative_values: Optional[Dict[str, List[Any]]] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> CounterfactualReport:
        """
        Run counterfactual analysis on a batch of users.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        sensitive_columns : list[str]
            Columns to flip.
        alternative_values : dict, optional
            Maps column → list of alternatives.
        sample_size : int, optional
            If set, randomly sample this many users (for large datasets).
        seed : int
            Random seed for sampling.

        Returns
        -------
        CounterfactualReport
        """
        if sample_size is not None and sample_size < len(X):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(X), size=sample_size, replace=False)
        else:
            indices = np.arange(len(X))

        report = CounterfactualReport(
            n_users=len(indices),
            sensitive_columns=sensitive_columns,
            flip_threshold=self.flip_threshold,
        )

        for idx in indices:
            result = self.analyze_user(
                X,
                int(idx),
                sensitive_columns,
                alternative_values,
            )
            report.results.append(result)

        logger.info(
            "Counterfactual analysis complete: %d/%d users sensitive (%.1f%%)",
            len(report.sensitive_users),
            report.n_users,
            report.sensitivity_rate * 100,
        )

        return report
