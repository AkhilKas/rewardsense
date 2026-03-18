"""
Hyperparameter sensitivity analysis for the personalization model (Story 5.2).

Uses Optuna trial data (logged during Story 3.3) to:
- Rank hyperparameters by importance (correlation with metric)
- Generate per-parameter sensitivity curves
- Identify interaction effects between parameter pairs
- Define safe operating ranges
"""

from __future__ import annotations

import itertools
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class ParameterRange:
    """Safe operating range for a single hyperparameter."""

    name: str
    best_value: float
    lower: float
    upper: float
    importance_rank: int
    correlation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "best_value": self.best_value,
            "lower_bound": self.lower,
            "upper_bound": self.upper,
            "importance_rank": self.importance_rank,
            "correlation_with_metric": self.correlation,
        }


@dataclass
class HPSensitivityResult:
    """Full result of hyperparameter sensitivity analysis."""

    importance_ranking: pd.DataFrame
    safe_ranges: List[ParameterRange] = field(default_factory=list)
    interactions: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "importance_ranking": self.importance_ranking.to_dict(orient="records"),
            "safe_ranges": [r.to_dict() for r in self.safe_ranges],
            "interactions": (
                self.interactions.to_dict(orient="records")
                if self.interactions is not None
                else []
            ),
        }


class HyperparameterAnalyzer:
    """Analyze how the model's performance varies with hyperparameters.

    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame of Optuna trials.  Expected columns include
        ``value`` (objective metric, e.g. CV-RMSE) and
        ``params_<name>`` columns for each hyperparameter.
    metric_column : str
        Name of the metric column in *trials_df*.
    metric_direction : str
        ``"minimize"`` or ``"maximize"`` — controls how "better" is defined.
    """

    def __init__(
        self,
        trials_df: pd.DataFrame,
        metric_column: str = "value",
        metric_direction: str = "minimize",
    ) -> None:
        self.trials_df = trials_df.copy()
        self.metric_column = metric_column
        self.metric_direction = metric_direction
        self._param_cols: List[str] = [
            c for c in self.trials_df.columns if c.startswith("params_")
        ]
        self._param_names: List[str] = [
            c.replace("params_", "") for c in self._param_cols
        ]

    @classmethod
    def from_optuna_study(
        cls, study: Any, metric_column: str = "value"
    ) -> "HyperparameterAnalyzer":
        """Construct from a live Optuna study object."""
        df = study.trials_dataframe()
        direction = (
            "minimize" if str(study.direction).endswith("MINIMIZE") else "maximize"
        )
        return cls(df, metric_column=metric_column, metric_direction=direction)

    @property
    def param_names(self) -> List[str]:
        return list(self._param_names)

    def compute_parameter_importance(self) -> pd.DataFrame:
        """Rank hyperparameters by absolute Spearman correlation with metric.

        Returns
        -------
        pd.DataFrame with ``param``, ``abs_correlation``, ``p_value``, ``rank``.
        """
        rows: List[Dict[str, Any]] = []
        metric = self.trials_df[self.metric_column].astype(float)

        for pcol, pname in zip(self._param_cols, self._param_names):
            vals = pd.to_numeric(self.trials_df[pcol], errors="coerce")
            valid = vals.notna() & metric.notna()
            if valid.sum() < 3:
                continue
            rho, p = stats.spearmanr(vals[valid], metric[valid])
            rows.append(
                {
                    "param": pname,
                    "correlation": round(float(rho), 4),
                    "abs_correlation": round(abs(float(rho)), 4),
                    "p_value": round(float(p), 6),
                }
            )

        df = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1
        logger.info("HP importance ranking:\n{}", df.to_string(index=False))
        return df

    def identify_safe_ranges(
        self,
        importance: pd.DataFrame,
        quantile_low: float = 0.1,
        quantile_high: float = 0.9,
    ) -> List[ParameterRange]:
        """Define safe operating ranges based on the top-performing trials.

        Uses the best 25% of trials to define the [q_low, q_high] range
        for each parameter.
        """
        metric = self.trials_df[self.metric_column].astype(float)
        if self.metric_direction == "minimize":
            threshold = metric.quantile(0.25)
            good_mask = metric <= threshold
        else:
            threshold = metric.quantile(0.75)
            good_mask = metric >= threshold

        good_trials = self.trials_df[good_mask]
        best_idx = (
            metric.idxmin() if self.metric_direction == "minimize" else metric.idxmax()
        )

        ranges: List[ParameterRange] = []
        for _, row in importance.iterrows():
            pname = row["param"]
            pcol = f"params_{pname}"
            if pcol not in good_trials.columns:
                continue
            vals = pd.to_numeric(good_trials[pcol], errors="coerce").dropna()
            if vals.empty:
                continue

            best_val = pd.to_numeric(
                self.trials_df.loc[best_idx, pcol], errors="coerce"
            )
            ranges.append(
                ParameterRange(
                    name=pname,
                    best_value=(
                        float(best_val) if pd.notna(best_val) else float(vals.median())
                    ),
                    lower=float(vals.quantile(quantile_low)),
                    upper=float(vals.quantile(quantile_high)),
                    importance_rank=int(row["rank"]),
                    correlation=float(row["correlation"]),
                )
            )

        logger.info("Identified safe ranges for {} parameters", len(ranges))
        return ranges

    def compute_interactions(self, top_n: int = 5) -> pd.DataFrame:
        """Compute pairwise interaction strength for the top-N parameters.

        Interaction is measured as |correlation between param_i * param_j
        and the metric| minus the max of their individual correlations.

        Returns
        -------
        pd.DataFrame with ``param_1``, ``param_2``, ``interaction_strength``.
        """
        importance = self.compute_parameter_importance()
        top_params = importance.head(top_n)["param"].tolist()

        metric = self.trials_df[self.metric_column].astype(float)
        indiv_corr: Dict[str, float] = {}
        for p in top_params:
            pcol = f"params_{p}"
            vals = pd.to_numeric(self.trials_df[pcol], errors="coerce")
            valid = vals.notna() & metric.notna()
            if valid.sum() >= 3:
                rho, _ = stats.spearmanr(vals[valid], metric[valid])
                indiv_corr[p] = abs(float(rho))
            else:
                indiv_corr[p] = 0.0

        rows: List[Dict[str, Any]] = []
        for p1, p2 in itertools.combinations(top_params, 2):
            c1 = f"params_{p1}"
            c2 = f"params_{p2}"
            v1 = pd.to_numeric(self.trials_df[c1], errors="coerce")
            v2 = pd.to_numeric(self.trials_df[c2], errors="coerce")
            valid = v1.notna() & v2.notna() & metric.notna()
            if valid.sum() < 3:
                continue
            product = v1[valid] * v2[valid]
            rho, _ = stats.spearmanr(product, metric[valid])
            interaction = abs(float(rho)) - max(
                indiv_corr.get(p1, 0), indiv_corr.get(p2, 0)
            )
            rows.append(
                {
                    "param_1": p1,
                    "param_2": p2,
                    "interaction_strength": round(max(interaction, 0.0), 4),
                    "joint_correlation": round(abs(float(rho)), 4),
                }
            )

        df = pd.DataFrame(rows).sort_values("interaction_strength", ascending=False)
        df = df.reset_index(drop=True)
        return df

    def run(self) -> HPSensitivityResult:
        """Execute the full sensitivity analysis pipeline."""
        importance = self.compute_parameter_importance()
        safe_ranges = self.identify_safe_ranges(importance)
        interactions = self.compute_interactions()

        return HPSensitivityResult(
            importance_ranking=importance,
            safe_ranges=safe_ranges,
            interactions=interactions,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def generate_param_vs_metric_plot(self, param_name: str) -> plt.Figure:
        """Scatter plot of a single parameter vs the metric."""
        pcol = f"params_{param_name}"
        metric = self.trials_df[self.metric_column].astype(float)
        vals = pd.to_numeric(self.trials_df[pcol], errors="coerce")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(vals, metric, alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.metric_column)
        ax.set_title(f"{param_name} vs {self.metric_column}")
        plt.tight_layout()
        return fig

    def generate_interaction_plot(self, param_1: str, param_2: str) -> plt.Figure:
        """Scatter plot of two parameters, coloured by the metric."""
        c1 = f"params_{param_1}"
        c2 = f"params_{param_2}"
        metric = self.trials_df[self.metric_column].astype(float)
        v1 = pd.to_numeric(self.trials_df[c1], errors="coerce")
        v2 = pd.to_numeric(self.trials_df[c2], errors="coerce")

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            v1, v2, c=metric, cmap="viridis", alpha=0.7, edgecolors="k", linewidths=0.3
        )
        plt.colorbar(sc, ax=ax, label=self.metric_column)
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        ax.set_title(f"Interaction: {param_1} × {param_2}")
        plt.tight_layout()
        return fig

    def generate_importance_bar_plot(self, importance: pd.DataFrame) -> plt.Figure:
        """Horizontal bar chart of parameter importance."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance["param"][::-1], importance["abs_correlation"][::-1])
        ax.set_xlabel("|Spearman ρ| with metric")
        ax.set_title("Hyperparameter Importance")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        result: HPSensitivityResult,
        tracker: Optional[Any] = None,
        artifact_subdir: str = "sensitivity/hyperparams",
    ) -> None:
        """Save HP sensitivity artifacts and log to MLflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            result.importance_ranking.to_csv(tmp / "hp_importance.csv", index=False)

            import json

            (tmp / "hp_safe_ranges.json").write_text(
                json.dumps([r.to_dict() for r in result.safe_ranges], indent=2)
            )

            if result.interactions is not None and not result.interactions.empty:
                result.interactions.to_csv(tmp / "hp_interactions.csv", index=False)

            try:
                fig = self.generate_importance_bar_plot(result.importance_ranking)
                fig.savefig(tmp / "hp_importance.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                logger.warning("HP importance plot failed: {}", exc)

            for pname in result.importance_ranking.head(5)["param"]:
                try:
                    fig = self.generate_param_vs_metric_plot(pname)
                    fig.savefig(
                        tmp / f"hp_scatter_{pname}.png", dpi=150, bbox_inches="tight"
                    )
                    plt.close(fig)
                except Exception as exc:
                    logger.warning("HP scatter for '{}' failed: {}", pname, exc)

            if result.interactions is not None and len(result.interactions) > 0:
                top_inter = result.interactions.head(3)
                for _, row in top_inter.iterrows():
                    try:
                        fig = self.generate_interaction_plot(
                            row["param_1"], row["param_2"]
                        )
                        fig.savefig(
                            tmp
                            / f"hp_interaction_{row['param_1']}_{row['param_2']}.png",
                            dpi=150,
                            bbox_inches="tight",
                        )
                        plt.close(fig)
                    except Exception as exc:
                        logger.warning("Interaction plot failed: {}", exc)

            if tracker is not None:
                tracker.log_artifacts(str(tmp), artifact_path=artifact_subdir)
                logger.info("Logged HP sensitivity artifacts to MLflow")
