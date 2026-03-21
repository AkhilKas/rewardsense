"""
Holdout test-set validation and reporting (Story 3.4).

Final gate before a model is promoted to the registry:
1. Evaluate on the unseen test set
2. Compute per-segment metrics
3. Run overfitting check (val vs test)
4. Generate a JSON validation report
5. Return pass/fail verdict
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.model_pipeline.personalization.evaluation import (
    EvaluationReport,
    RegressionMetrics,
    check_overfitting,
    evaluate,
)
from src.model_pipeline.personalization.splits import SplitResult


@dataclass
class ValidationVerdict:
    """Pass/fail gate result for model promotion."""

    passed: bool
    test_report: EvaluationReport
    reason: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "test_metrics": self.test_report.to_dict(),
        }


class HoldoutValidator:
    """Validate a trained model on the holdout test set.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model.
    split : SplitResult
        Must contain X_test, y_test, and optionally meta_test.
    val_metrics : RegressionMetrics
        Validation-set metrics from training (for overfitting check).
    rmse_threshold : float
        Maximum acceptable test RMSE for promotion.
    r2_threshold : float
        Minimum acceptable test R² for promotion.
    max_overfit_gap : float
        Maximum RMSE gap (test - val) before flagging overfitting.
    artifact_dir : str or Path
        Where to save the validation report.
    """

    def __init__(
        self,
        model: Any,
        split: SplitResult,
        val_metrics: RegressionMetrics,
        rmse_threshold: float = 0.01,
        r2_threshold: float = 0.30,
        max_overfit_gap: float = 0.005,
        artifact_dir: str = "models/personalization",
    ) -> None:
        self.model = model
        self.split = split
        self.val_metrics = val_metrics
        self.rmse_threshold = rmse_threshold
        self.r2_threshold = r2_threshold
        self.max_overfit_gap = max_overfit_gap
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> ValidationVerdict:
        """Run full validation on the holdout test set.

        Returns
        -------
        ValidationVerdict
            Contains pass/fail, metrics, and the reason string.
        """
        test_pred = self.model.predict(self.split.X_test)

        test_report = evaluate(
            self.split.y_test,
            test_pred,
            meta=self.split.meta_test,
            train_metrics=self.val_metrics,
        )

        try:
            self.generate_validation_plots(test_pred)
        except Exception as exc:
            logger.warning("Validation plot generation failed: {}", exc)

        reasons: List[str] = []
        passed = True

        if test_report.overall.rmse > self.rmse_threshold:
            reasons.append(
                f"Test RMSE {test_report.overall.rmse:.6f} > threshold {self.rmse_threshold}"
            )
            passed = False

        if test_report.overall.r2 < self.r2_threshold:
            reasons.append(
                f"Test R² {test_report.overall.r2:.4f} < threshold {self.r2_threshold}"
            )
            passed = False

        overfit = check_overfitting(self.val_metrics, test_report.overall)
        if overfit.get("is_overfit", False):
            reasons.append(f"Overfitting detected: RMSE gap={overfit['rmse_gap']:.6f}")
            passed = False

        reason = "All checks passed" if passed else "; ".join(reasons)

        verdict = ValidationVerdict(
            passed=passed,
            test_report=test_report,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._save_report(verdict)

        status = "PASSED" if passed else "FAILED"
        logger.info(
            "Holdout validation {}: RMSE={:.6f}, R²={:.4f} — {}",
            status,
            test_report.overall.rmse,
            test_report.overall.r2,
            reason,
        )

        return verdict

    def _save_report(self, verdict: ValidationVerdict) -> Path:
        """Save validation report JSON to artifact directory."""
        report_path = self.artifact_dir / "holdout_validation_report.json"
        report_path.write_text(
            json.dumps(verdict.to_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Saved validation report to {}", report_path)
        return report_path

    def generate_validation_plots(
        self,
        test_pred: Optional[np.ndarray] = None,
    ) -> Dict[str, Path]:
        """Generate and save validation visualizations.

        Produces:
        - **Prediction vs Actual** scatter plot
        - **Residual distribution** histogram
        - **Segment performance** bar chart (if meta_test available)

        Returns dict mapping plot name → saved file path.
        """
        if test_pred is None:
            test_pred = self.model.predict(self.split.X_test)

        y_test = self.split.y_test.values
        residuals = y_test - test_pred
        saved: Dict[str, Path] = {}

        # --- Prediction vs Actual ---
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test, test_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
        lo = min(y_test.min(), test_pred.min())
        hi = max(y_test.max(), test_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Holdout: Predicted vs Actual")
        ax.legend()
        plt.tight_layout()
        path = self.artifact_dir / "pred_vs_actual.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved["pred_vs_actual"] = path

        # --- Residual histogram ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (actual - predicted)")
        ax.set_ylabel("Count")
        ax.set_title("Holdout: Residual Distribution")
        plt.tight_layout()
        path = self.artifact_dir / "residual_histogram.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved["residual_histogram"] = path

        # --- Segment performance bar chart ---
        if self.split.meta_test is not None:
            from src.model_pipeline.personalization.evaluation import (
                compute_segment_metrics,
            )

            import pandas as pd

            y_series = pd.Series(y_test, index=self.split.y_test.index)
            seg_metrics = compute_segment_metrics(
                y_series, test_pred, self.split.meta_test
            )

            for dim, segs in seg_metrics.items():
                if not segs:
                    continue
                names = list(segs.keys())
                rmses = [segs[n].rmse for n in names]

                fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
                ax.bar(names, rmses, edgecolor="black", alpha=0.8)
                ax.set_xlabel(dim)
                ax.set_ylabel("RMSE")
                ax.set_title(f"Holdout RMSE by {dim}")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                safe_dim = dim.replace("/", "_")
                path = self.artifact_dir / f"segment_rmse_{safe_dim}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved[f"segment_rmse_{dim}"] = path

        logger.info(
            "Generated {} validation plots in {}", len(saved), self.artifact_dir
        )
        return saved
