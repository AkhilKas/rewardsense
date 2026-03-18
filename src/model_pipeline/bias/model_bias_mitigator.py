"""
Model-Level Bias Mitigation.

Extends BiasMitigator with model-specific strategies: ExponentiatedGradient, ThresholdOptimizer on model predictions, scoring weight adjustments, and prompt modifications.

Logs before/after comparisons to MLflow for trade-off documentation.

Usage:
    mitigator = ModelBiasMitigator()

    # For the personalization model
    result = mitigator.mitigate_with_exponentiated_gradient(
        estimator=xgb_model,
        X_train=X_train, y_train=y_train,
        sensitive_features=train_df["archetype"],
    )

    # For the scoring engine
    result = mitigator.adjust_scoring_weights(
        scoring_engine=scorer,
        bias_report=scoring_bias_report,
    )

    # Log trade-offs
    mitigator.log_comparison(tracker, before_report, after_report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Fairlearn import
# ---------------------------------------------------------------------------
try:
    from fairlearn.reductions import (
        DemographicParity,
        EqualizedOdds,
        ExponentiatedGradient,
    )
    from fairlearn.postprocessing import ThresholdOptimizer

    FAIRLEARN_REDUCTIONS_AVAILABLE = True
except ImportError:
    FAIRLEARN_REDUCTIONS_AVAILABLE = False
    logger.warning(
        "fairlearn.reductions not available — "
        "ExponentiatedGradient mitigation disabled"
    )


# =====================================================================
# Result dataclass
# =====================================================================


@dataclass
class MitigationResult:
    """Result of a single mitigation strategy."""

    strategy: str
    component: str  # "personalization", "scoring_engine", "llm"
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)
    trade_offs: Dict[str, str] = field(default_factory=dict)
    mitigated_model: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> Dict[str, float]:
        """Compute improvement for each metric (after - before)."""
        return {
            k: self.after_metrics.get(k, 0) - self.before_metrics.get(k, 0)
            for k in set(self.before_metrics) | set(self.after_metrics)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "component": self.component,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "improvement": self.improvement,
            "trade_offs": self.trade_offs,
            "metadata": self.metadata,
        }


# =====================================================================
# ModelBiasMitigator
# =====================================================================


class ModelBiasMitigator:
    """
    Apply bias mitigation strategies to RewardSense model components.

    Strategies available:
    1. ExponentiatedGradient (Fairlearn) — in-processing for personalization
    2. ThresholdOptimizer (Fairlearn) — post-processing for personalization
    3. Sample reweighting — pre-processing (reuses Phase 1 logic)
    4. Scoring weight adjustment — for deterministic scoring engine
    5. Prompt modification — for LLM explainability

    Parameters
    ----------
    constraint : str
        Fairlearn constraint type: "demographic_parity" or "equalized_odds".
    """

    def __init__(self, constraint: str = "demographic_parity") -> None:
        self.constraint = constraint

    # ------------------------------------------------------------------
    # Strategy 1: ExponentiatedGradient (in-processing)
    # ------------------------------------------------------------------

    def mitigate_with_exponentiated_gradient(
        self,
        estimator: Any,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        sensitive_features: np.ndarray | pd.Series,
        eps: float = 0.01,
        max_iter: int = 50,
    ) -> MitigationResult:
        """
        Apply ExponentiatedGradient reduction for fair classification.

        Parameters
        ----------
        estimator : sklearn-compatible estimator
            Base model to wrap (e.g., XGBClassifier).
        X_train, y_train : array-like
            Training data.
        sensitive_features : array-like
            Protected attribute values for training samples.
        eps : float
            Constraint violation tolerance.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        MitigationResult
            Contains the mitigated (wrapped) model in `mitigated_model`.
        """
        result = MitigationResult(
            strategy="exponentiated_gradient",
            component="personalization",
        )

        if not FAIRLEARN_REDUCTIONS_AVAILABLE:
            result.trade_offs["error"] = (
                "fairlearn.reductions not installed — cannot apply"
            )
            logger.error("ExponentiatedGradient unavailable")
            return result

        # Select constraint
        if self.constraint == "equalized_odds":
            constraint_obj = EqualizedOdds()
        else:
            constraint_obj = DemographicParity()

        mitigator = ExponentiatedGradient(
            estimator=estimator,
            constraints=constraint_obj,
            eps=eps,
            max_iter=max_iter,
        )

        logger.info(
            "Fitting ExponentiatedGradient (constraint=%s, eps=%s, max_iter=%s)",
            self.constraint,
            eps,
            max_iter,
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)

        result.mitigated_model = mitigator
        result.metadata = {
            "constraint": self.constraint,
            "eps": eps,
            "max_iter": max_iter,
            "n_predictors": len(mitigator.predictors_),
        }
        result.trade_offs["fairness_vs_accuracy"] = (
            "ExponentiatedGradient may reduce overall accuracy to improve "
            "fairness across groups. Check after_metrics for the trade-off."
        )

        return result

    # ------------------------------------------------------------------
    # Strategy 2: ThresholdOptimizer (post-processing)
    # ------------------------------------------------------------------

    def mitigate_with_threshold_optimizer(
        self,
        estimator: Any,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        sensitive_features: np.ndarray | pd.Series,
        objective: str = "balanced_accuracy_score",
    ) -> MitigationResult:
        """
        Apply per-group threshold optimization for fair predictions.

        Parameters
        ----------
        estimator : sklearn-compatible estimator
            Already-trained model with predict_proba or decision_function.
        X_train, y_train : array-like
            Validation data for threshold fitting.
        sensitive_features : array-like
            Protected attributes.
        objective : str
            Scikit-learn metric to optimize subject to fairness constraint.
        """
        result = MitigationResult(
            strategy="threshold_optimizer",
            component="personalization",
        )

        if not FAIRLEARN_REDUCTIONS_AVAILABLE:
            result.trade_offs["error"] = "fairlearn not installed"
            return result

        if self.constraint == "equalized_odds":
            constraint_str = "equalized_odds"
        else:
            constraint_str = "demographic_parity"

        to = ThresholdOptimizer(
            estimator=estimator,
            constraints=constraint_str,
            objective=objective,
            prefit=True,
        )

        logger.info("Fitting ThresholdOptimizer (constraint=%s)", constraint_str)
        to.fit(X_train, y_train, sensitive_features=sensitive_features)

        result.mitigated_model = to
        result.metadata = {
            "constraint": constraint_str,
            "objective": objective,
        }
        result.trade_offs["per_group_thresholds"] = (
            "Different decision thresholds per group may raise transparency "
            "concerns. Document and justify in model card."
        )

        return result

    # ------------------------------------------------------------------
    # Strategy 3: Sample reweighting (delegates to Phase 1)
    # ------------------------------------------------------------------

    def compute_sample_weights(
        self,
        sensitive_features: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """
        Compute inverse-frequency sample weights for fair training.

        Reuses the BiasMitigator concept but returns weights directly for model training integration.
        """
        sf = pd.Series(sensitive_features)
        group_counts = sf.value_counts()
        total = len(sf)
        n_groups = len(group_counts)

        weights = sf.map(lambda g: total / (n_groups * group_counts[g])).values.astype(
            np.float64
        )

        return weights

    # ------------------------------------------------------------------
    # Strategy 4: Scoring weight adjustment
    # ------------------------------------------------------------------

    def recommend_scoring_adjustments(
        self,
        bias_report: Any,
        max_adjustment: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Recommend scoring engine adjustments based on bias findings.

        If the scoring engine disproportionately recommends certain
        issuers for specific user segments, suggest diversity constraints.

        Parameters
        ----------
        bias_report : ComponentBiasReport
            Output from ScoringBiasChecker.
        max_adjustment : float
            Maximum weight adjustment magnitude.

        Returns
        -------
        dict
            Recommended adjustments with rationale.
        """
        adjustments = {
            "apply_diversity_penalty": False,
            "issuer_caps": {},
            "rationale": [],
        }

        biased = [m for m in bias_report.metrics if m.is_biased]
        if not biased:
            adjustments["rationale"].append("No issuer bias detected.")
            return adjustments

        adjustments["apply_diversity_penalty"] = True

        for m in biased:
            issuer = m.details.get("issuer", "unknown")
            rates = m.details.get("per_group_rates", {})
            if rates:
                max_rate = max(rates.values())
                adjustments["issuer_caps"][issuer] = min(
                    max_rate, max_rate - max_adjustment
                )
                adjustments["rationale"].append(
                    f"Issuer '{issuer}' over-recommended in some segments "
                    f"(max rate={max_rate:.2%}). Suggest cap at "
                    f"{max_rate - max_adjustment:.2%}."
                )

        return adjustments

    # ------------------------------------------------------------------
    # Strategy 5: Prompt modification suggestions
    # ------------------------------------------------------------------

    def recommend_prompt_adjustments(
        self,
        bias_report: Any,
    ) -> Dict[str, Any]:
        """
        Recommend LLM prompt changes based on explanation bias findings.

        Parameters
        ----------
        bias_report : ComponentBiasReport
            Output from ExplanationBiasChecker.
        """
        adjustments = {
            "modify_prompts": False,
            "suggestions": [],
        }

        biased = [m for m in bias_report.metrics if m.is_biased]
        if not biased:
            adjustments["suggestions"].append("No explanation bias detected.")
            return adjustments

        adjustments["modify_prompts"] = True

        for m in biased:
            if "length" in m.check_name:
                adjustments["suggestions"].append(
                    "Add explicit length guidance to prompts: "
                    "'Provide explanations of 50-100 words regardless of user profile.'"
                )
            elif "readability" in m.check_name:
                adjustments["suggestions"].append(
                    "Add readability constraint: "
                    "'Write at an 8th-grade reading level for all users.'"
                )
            elif "detail" in m.check_name:
                adjustments["suggestions"].append(
                    "Standardize detail level: "
                    "'Include exactly 3 key factors in every explanation.'"
                )

        return adjustments

    # ------------------------------------------------------------------
    # Comparison & MLflow logging
    # ------------------------------------------------------------------

    @staticmethod
    def log_comparison(
        tracker: Any,
        before_report: Any,
        after_report: Any,
        strategy_name: str = "",
    ) -> None:
        """
        Log before/after bias comparison to MLflow.

        Parameters
        ----------
        tracker : RewardSenseTracker
            MLflow tracking wrapper.
        before_report, after_report
            ModelBiasReport or ComponentBiasReport instances.
        strategy_name : str
            Label for this mitigation experiment.
        """
        if tracker is None:
            return

        comparison = {
            "strategy": strategy_name,
            "before": (
                before_report.to_dict()
                if hasattr(before_report, "to_dict")
                else str(before_report)
            ),
            "after": (
                after_report.to_dict()
                if hasattr(after_report, "to_dict")
                else str(after_report)
            ),
        }

        # Compute improvement summary
        before_biased = (
            len(before_report.biased_metrics)
            if hasattr(before_report, "biased_metrics")
            else 0
        )
        after_biased = (
            len(after_report.biased_metrics)
            if hasattr(after_report, "biased_metrics")
            else 0
        )

        tracker.log_metrics(
            {
                f"mitigation_{strategy_name}_before_biased": before_biased,
                f"mitigation_{strategy_name}_after_biased": after_biased,
                f"mitigation_{strategy_name}_reduction": before_biased - after_biased,
            }
        )

        tracker.log_dict(comparison, f"mitigation_comparison_{strategy_name}.json")
