"""
Sensitivity analysis report generator.

Aggregates results from SHAP, LIME, segment, and hyperparameter analyses
into a single structured JSON report and a human-readable Markdown summary.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class SensitivityReportGenerator:
    """Build and persist a comprehensive sensitivity analysis report.

    Call ``add_*`` methods to supply results from each analysis module,
    then call ``generate`` to produce the final report dict.
    """

    def __init__(self, model_name: str = "personalization") -> None:
        self.model_name = model_name
        self._sections: Dict[str, Any] = {}
        self._generated_at: Optional[str] = None

    # ------------------------------------------------------------------
    # Section adders
    # ------------------------------------------------------------------

    def add_shap_results(
        self,
        global_importance: Any,
        top_features: List[str],
    ) -> None:
        """Add SHAP analysis results.

        Parameters
        ----------
        global_importance : pd.DataFrame or dict
            Feature importance table.
        top_features : list of str
            Top-N feature names.
        """
        imp = (
            global_importance.to_dict(orient="records")
            if hasattr(global_importance, "to_dict")
            else global_importance
        )
        self._sections["shap"] = {
            "global_importance": imp,
            "top_features": top_features,
            "n_features_analyzed": len(imp),
        }

    def add_lime_results(
        self,
        aggregated_importance: Any,
        consistency_check: Optional[Dict[str, Any]] = None,
    ) -> None:
        imp = (
            aggregated_importance.to_dict(orient="records")
            if hasattr(aggregated_importance, "to_dict")
            else aggregated_importance
        )
        self._sections["lime"] = {
            "aggregated_importance": imp,
            "consistency_check": consistency_check,
        }

    def add_segment_results(
        self,
        comparison_table: Any,
        segment_count: int,
    ) -> None:
        comp = (
            comparison_table.to_dict(orient="records")
            if hasattr(comparison_table, "to_dict")
            else comparison_table
        )
        self._sections["segments"] = {
            "comparison": comp,
            "n_segments": segment_count,
        }

    def add_hp_results(
        self,
        hp_result_dict: Dict[str, Any],
    ) -> None:
        self._sections["hyperparameters"] = hp_result_dict

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self) -> Dict[str, Any]:
        """Produce the full report as a nested dict."""
        self._generated_at = datetime.now(timezone.utc).isoformat()
        return {
            "report_type": "sensitivity_analysis",
            "model": self.model_name,
            "generated_at": self._generated_at,
            "sections": self._sections,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.generate(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Render a human-readable Markdown summary."""
        report = self.generate()
        lines: List[str] = [
            f"# Sensitivity Analysis Report — {self.model_name}",
            f"*Generated: {report['generated_at']}*",
            "",
        ]

        if "shap" in self._sections:
            s = self._sections["shap"]
            lines.append("## SHAP Feature Importance")
            lines.append(f"- Features analyzed: {s['n_features_analyzed']}")
            lines.append(f"- Top features: {', '.join(s['top_features'][:5])}")
            lines.append("")

        if "lime" in self._sections:
            s = self._sections["lime"]
            lines.append("## LIME Analysis")
            cc = s.get("consistency_check") or {}
            if cc:
                lines.append(
                    f"- SHAP/LIME Spearman rho: {cc.get('spearman_rho', 'N/A')}"
                )
                lines.append(f"- Top-5 overlap: {cc.get('top_5_overlap_pct', 'N/A')}%")
            lines.append("")

        if "segments" in self._sections:
            s = self._sections["segments"]
            lines.append("## Segment Analysis")
            lines.append(f"- Segments analyzed: {s['n_segments']}")
            lines.append("")

        if "hyperparameters" in self._sections:
            s = self._sections["hyperparameters"]
            lines.append("## Hyperparameter Sensitivity")
            ranking = s.get("importance_ranking", [])
            if ranking:
                lines.append("| Rank | Parameter | |ρ| |")
                lines.append("|------|-----------|-----|")
                for r in ranking[:5]:
                    lines.append(
                        f"| {r.get('rank', '')} | {r.get('param', '')} "
                        f"| {r.get('abs_correlation', '')} |"
                    )
            lines.append("")
            safe = s.get("safe_ranges", [])
            if safe:
                lines.append("### Safe Operating Ranges")
                for sr in safe[:5]:
                    lines.append(
                        f"- **{sr['name']}**: [{sr['lower_bound']:.4f}, "
                        f"{sr['upper_bound']:.4f}] (best={sr['best_value']:.4f})"
                    )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence & MLflow
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path) -> Dict[str, Path]:
        """Write JSON and Markdown reports to *output_dir*.

        Returns dict with keys ``json`` and ``markdown`` pointing to paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "sensitivity_report.json"
        json_path.write_text(self.to_json(), encoding="utf-8")

        md_path = out / "sensitivity_report.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")

        logger.info("Saved sensitivity report to {}", out)
        return {"json": json_path, "markdown": md_path}

    def log_to_mlflow(
        self,
        tracker: Optional[Any] = None,
        artifact_subdir: str = "sensitivity/report",
    ) -> None:
        """Save report and log to MLflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save(tmpdir)
            if tracker is not None:
                tracker.log_artifacts(str(tmpdir), artifact_path=artifact_subdir)
                logger.info("Logged sensitivity report to MLflow")
