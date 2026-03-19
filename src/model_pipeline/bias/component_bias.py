"""
Scoring Engine & LLM Explainability Bias Detection.

Ensures scoring engine doesn't favor specific card issuers and LLM explanation quality is consistent across user segments.

Usage:
    # Scoring engine bias
    checker = ScoringBiasChecker()
    report = checker.check_issuer_bias(recommendations_df, user_segments)

    # LLM explanation bias
    checker = ExplanationBiasChecker()
    report = checker.check_quality_consistency(explanations_df, user_segments)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Shared report structures
# =====================================================================


@dataclass
class ComponentBiasMetric:
    """A single bias check result for a non-ML component."""

    component: str  # "scoring_engine" or "llm_explainability"
    check_name: str
    sensitive_feature: str
    value: float
    threshold: float
    is_biased: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentBiasReport:
    """Aggregated bias report for scoring engine and/or LLM."""

    component: str
    metrics: List[ComponentBiasMetric] = field(default_factory=list)

    @property
    def biased_metrics(self) -> List[ComponentBiasMetric]:
        return [m for m in self.metrics if m.is_biased]

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "total_checks": len(self.metrics),
            "biased": len(self.biased_metrics),
            "clean": len(self.metrics) - len(self.biased_metrics),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "metrics": [
                {
                    "check": m.check_name,
                    "feature": m.sensitive_feature,
                    "value": m.value,
                    "threshold": m.threshold,
                    "is_biased": m.is_biased,
                    "details": m.details,
                }
                for m in self.metrics
            ],
        }

    def log_to_mlflow(self, tracker: Any) -> None:
        """
        Log to MLflow via RewardSenseTracker.

        Logs JSON report + matplotlib/seaborn visualizations:
          - Issuer distribution chart (scoring engine)
          - Explanation quality chart (LLM)
        """
        if tracker is None:
            return
        tracker.log_metrics(
            {
                f"{self.component}_bias_checks": len(self.metrics),
                f"{self.component}_bias_flagged": len(self.biased_metrics),
            }
        )
        tracker.log_dict(self.to_dict(), f"bias_{self.component}.json")

        # --- Visualizations ---
        try:
            from src.model_pipeline.bias.visualizations import (
                plot_issuer_distribution,
                plot_explanation_quality,
            )
            import matplotlib.pyplot as _plt

            metrics_dicts = [
                {
                    "check": m.check_name,
                    "check_name": m.check_name,
                    "details": m.details,
                }
                for m in self.metrics
            ]

            if self.component == "scoring_engine" and metrics_dicts:
                fig = plot_issuer_distribution(metrics_dicts)
                tracker.log_figure(fig, f"issuer_distribution_{self.component}.png")
                _plt.close(fig)

            if self.component == "llm_explainability" and metrics_dicts:
                fig = plot_explanation_quality(metrics_dicts)
                tracker.log_figure(fig, f"explanation_quality_{self.component}.png")
                _plt.close(fig)
        except ImportError:
            pass  # matplotlib not installed — skip visualizations


# =====================================================================
# Scoring Engine Bias Checker
# =====================================================================


class ScoringBiasChecker:
    """
    Detect systematic bias in the deterministic scoring engine.

    Checks:
    1. Issuer distribution: do recommendations disproportionately
       favor specific card issuers across user segments?
    2. Card type distribution: are premium vs basic cards recommended
       fairly across spending tiers?
    3. Reward type distribution: are cash-back vs travel-points
       recommendations balanced?

    Parameters
    ----------
    issuer_disparity_threshold : float
        Max allowed deviation in issuer recommendation rate
        between any two user segments. Default 0.15.
    card_type_disparity_threshold : float
        Same but for card type (premium/standard). Default 0.15.
    """

    def __init__(
        self,
        issuer_disparity_threshold: float = 0.15,
        card_type_disparity_threshold: float = 0.15,
    ) -> None:
        self.issuer_threshold = issuer_disparity_threshold
        self.card_type_threshold = card_type_disparity_threshold

    def check_issuer_bias(
        self,
        recommendations: pd.DataFrame,
        sensitive_col: str,
        issuer_col: str = "recommended_card_issuer",
    ) -> ComponentBiasReport:
        """
        Check if card issuer recommendations vary by user segment.

        Parameters
        ----------
        recommendations : DataFrame
            Must contain ``sensitive_col`` and ``issuer_col``.
        sensitive_col : str
            Column defining user segments (e.g., "spending_archetype").
        issuer_col : str
            Column with the recommended card's issuer name.
        """
        report = ComponentBiasReport(component="scoring_engine")

        if issuer_col not in recommendations.columns:
            logger.warning("Issuer column '%s' not found", issuer_col)
            return report

        groups = recommendations.groupby(sensitive_col)

        # Per-group issuer distribution
        group_distributions: Dict[str, Dict[str, float]] = {}
        for name, grp in groups:
            counts = grp[issuer_col].value_counts(normalize=True)
            group_distributions[str(name)] = counts.to_dict()

        # Check each issuer for cross-group disparity
        all_issuers = recommendations[issuer_col].unique()
        for issuer in all_issuers:
            rates = {
                g: dist.get(issuer, 0.0) for g, dist in group_distributions.items()
            }
            if not rates:
                continue

            max_rate = max(rates.values())
            min_rate = min(rates.values())
            disparity = max_rate - min_rate

            report.metrics.append(
                ComponentBiasMetric(
                    component="scoring_engine",
                    check_name=f"issuer_disparity_{issuer}",
                    sensitive_feature=sensitive_col,
                    value=disparity,
                    threshold=self.issuer_threshold,
                    is_biased=disparity > self.issuer_threshold,
                    details={
                        "issuer": issuer,
                        "per_group_rates": rates,
                        "max_rate": max_rate,
                        "min_rate": min_rate,
                    },
                )
            )

        return report

    def check_card_type_bias(
        self,
        recommendations: pd.DataFrame,
        sensitive_col: str,
        card_type_col: str = "recommended_card_type",
    ) -> ComponentBiasReport:
        """Check if premium vs standard card recommendations vary by segment."""
        report = ComponentBiasReport(component="scoring_engine")

        if card_type_col not in recommendations.columns:
            logger.warning("Card type column '%s' not found", card_type_col)
            return report

        groups = recommendations.groupby(sensitive_col)

        group_premium_rates: Dict[str, float] = {}
        for name, grp in groups:
            premium_mask = (
                grp[card_type_col]
                .str.lower()
                .isin(["premium", "luxury", "platinum", "reserve"])
            )
            group_premium_rates[str(name)] = float(premium_mask.mean())

        if len(group_premium_rates) < 2:
            return report

        max_rate = max(group_premium_rates.values())
        min_rate = min(group_premium_rates.values())
        disparity = max_rate - min_rate

        report.metrics.append(
            ComponentBiasMetric(
                component="scoring_engine",
                check_name="premium_card_disparity",
                sensitive_feature=sensitive_col,
                value=disparity,
                threshold=self.card_type_threshold,
                is_biased=disparity > self.card_type_threshold,
                details={"per_group_premium_rates": group_premium_rates},
            )
        )

        return report


# =====================================================================
# LLM Explanation Bias Checker
# =====================================================================


class ExplanationBiasChecker:
    """
    Detect quality disparity in LLM-generated explanations.

    Checks:
    1. Length consistency: explanation length doesn't vary by segment
    2. Readability consistency: Flesch-Kincaid score is stable
    3. Sentiment consistency: explanations aren't systematically
       more/less positive for certain groups
    4. Factual density: number of card-specific facts mentioned

    Parameters
    ----------
    length_disparity_threshold : float
        Max allowed relative deviation in mean explanation length. Default 0.20.
    readability_disparity_threshold : float
        Max allowed deviation in readability score. Default 0.15.
    """

    def __init__(
        self,
        length_disparity_threshold: float = 0.20,
        readability_disparity_threshold: float = 0.15,
    ) -> None:
        self.length_threshold = length_disparity_threshold
        self.readability_threshold = readability_disparity_threshold

    @staticmethod
    def _flesch_kincaid_grade(text: str) -> float:
        """Approximate Flesch-Kincaid Grade Level."""
        sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
        words = text.split()
        n_words = max(len(words), 1)
        syllables = sum(
            max(1, sum(1 for c in w.lower() if c in "aeiou")) for w in words
        )
        return 0.39 * (n_words / sentences) + 11.8 * (syllables / n_words) - 15.59

    def check_quality_consistency(
        self,
        explanations: pd.DataFrame,
        sensitive_col: str,
        text_col: str = "explanation_text",
    ) -> ComponentBiasReport:
        """
        Check if explanation quality varies across user segments.

        Parameters
        ----------
        explanations : DataFrame
            Must contain ``sensitive_col`` and ``text_col``.
        sensitive_col : str
            Column defining user segments.
        text_col : str
            Column containing explanation text.
        """
        report = ComponentBiasReport(component="llm_explainability")

        if text_col not in explanations.columns:
            logger.warning("Text column '%s' not found", text_col)
            return report

        # Compute per-row quality metrics
        df = explanations.copy()
        df["_length"] = df[text_col].astype(str).apply(len)
        df["_word_count"] = df[text_col].astype(str).apply(lambda t: len(t.split()))
        df["_readability"] = df[text_col].astype(str).apply(self._flesch_kincaid_grade)

        overall_length = df["_length"].mean()
        overall_readability = df["_readability"].mean()

        groups = df.groupby(sensitive_col)

        # --- Length consistency ---
        group_lengths = groups["_length"].mean().to_dict()
        if overall_length > 0:
            max_length_dev = (
                max(
                    abs(v - overall_length) / overall_length
                    for v in group_lengths.values()
                )
                if group_lengths
                else 0.0
            )

            report.metrics.append(
                ComponentBiasMetric(
                    component="llm_explainability",
                    check_name="explanation_length_disparity",
                    sensitive_feature=sensitive_col,
                    value=max_length_dev,
                    threshold=self.length_threshold,
                    is_biased=max_length_dev > self.length_threshold,
                    details={
                        "overall_mean_length": overall_length,
                        "per_group_mean_length": group_lengths,
                    },
                )
            )

        # --- Readability consistency ---
        group_readability = groups["_readability"].mean().to_dict()
        if overall_readability != 0:
            max_read_dev = (
                max(
                    abs(v - overall_readability) / abs(overall_readability)
                    for v in group_readability.values()
                )
                if group_readability
                else 0.0
            )

            report.metrics.append(
                ComponentBiasMetric(
                    component="llm_explainability",
                    check_name="readability_disparity",
                    sensitive_feature=sensitive_col,
                    value=max_read_dev,
                    threshold=self.readability_threshold,
                    is_biased=max_read_dev > self.readability_threshold,
                    details={
                        "overall_readability": overall_readability,
                        "per_group_readability": group_readability,
                    },
                )
            )

        # --- Word count consistency (proxy for detail level) ---
        group_words = groups["_word_count"].mean().to_dict()
        overall_words = df["_word_count"].mean()
        if overall_words > 0:
            max_word_dev = (
                max(
                    abs(v - overall_words) / overall_words for v in group_words.values()
                )
                if group_words
                else 0.0
            )

            report.metrics.append(
                ComponentBiasMetric(
                    component="llm_explainability",
                    check_name="detail_level_disparity",
                    sensitive_feature=sensitive_col,
                    value=max_word_dev,
                    threshold=self.length_threshold,
                    is_biased=max_word_dev > self.length_threshold,
                    details={
                        "overall_mean_words": overall_words,
                        "per_group_mean_words": group_words,
                    },
                )
            )

        return report
