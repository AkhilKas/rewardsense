"""
RewardSense - Pipeline Execution Report Generator

Collects metadata from all upstream Airflow tasks (via XCom),
computes timing statistics, and writes a structured JSON report
to ``data/reports/``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("airflow.task")

# Default output directory (relative to repo root)
DEFAULT_REPORTS_DIR = Path("data/reports")


class PipelineReportGenerator:
    """Build a pipeline execution summary report.

    Parameters
    ----------
    reports_dir : Path | str, optional
        Directory where reports are written.  Created if it does not exist.
    """

    def __init__(self, reports_dir: Path | str = DEFAULT_REPORTS_DIR) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate a pipeline execution report from the Airflow *context*.

        Parameters
        ----------
        context : dict
            The Airflow ``**context`` dict passed to a PythonOperator callable.

        Returns
        -------
        dict
            A report summary suitable for pushing to XCom.
        """
        dag_run = context.get("dag_run")
        ti = context.get("ti")  # TaskInstance of the reporting task

        # ── Collect XCom values from upstream tasks ──────────────────────
        task_results = self._collect_upstream_xcoms(ti, dag_run)

        # ── Compute timing stats ─────────────────────────────────────────
        timing = self._compute_timing(dag_run)

        # ── Build report ─────────────────────────────────────────────────
        report: dict[str, Any] = {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "dag_id": dag_run.dag_id if dag_run else "unknown",
            "run_id": str(dag_run.run_id) if dag_run else "unknown",
            "execution_date": str(dag_run.execution_date) if dag_run else "unknown",
            "state": str(dag_run.state) if dag_run else "unknown",
            "timing": timing,
            "task_results": task_results,
        }

        # ── Persist to disk ──────────────────────────────────────────────
        filepath = self._write_report(report)
        logger.info("📊 Pipeline report written to %s", filepath)

        return {
            "report_path": str(filepath),
            "dag_run_id": report["run_id"],
            "total_duration_sec": timing.get("total_duration_sec"),
            "status": "completed",
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_upstream_xcoms(ti: Any, dag_run: Any) -> dict[str, Any]:
        """Pull XCom return values from every task in the current DAG run."""
        results: dict[str, Any] = {}
        if dag_run is None or ti is None:
            return results

        for task_id in dag_run.dag.task_ids:
            try:
                value = ti.xcom_pull(task_ids=task_id, dag_id=dag_run.dag_id)
                if value is not None:
                    results[task_id] = value
            except Exception:  # noqa: BLE001
                results[task_id] = {"error": "xcom_pull_failed"}
        return results

    @staticmethod
    def _compute_timing(dag_run: Any) -> dict[str, Any]:
        """Derive duration stats from the DAG run and its task instances."""
        timing: dict[str, Any] = {}
        if dag_run is None:
            return timing

        if dag_run.start_date:
            end = dag_run.end_date or datetime.now(timezone.utc)
            total_seconds = (end - dag_run.start_date).total_seconds()
            timing["start"] = str(dag_run.start_date)
            timing["end"] = str(end)
            timing["total_duration_sec"] = round(total_seconds, 2)

        # Per-task durations
        task_durations: dict[str, float | None] = {}
        try:
            for ti in dag_run.get_task_instances():
                if ti.start_date and ti.end_date:
                    task_durations[ti.task_id] = round(
                        (ti.end_date - ti.start_date).total_seconds(), 2
                    )
                else:
                    task_durations[ti.task_id] = None
        except Exception:  # noqa: BLE001
            pass
        timing["task_durations"] = task_durations
        return timing

    def _write_report(self, report: dict[str, Any]) -> Path:
        """Persist *report* as a timestamped JSON file."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_report_{ts}.json"
        filepath = self.reports_dir / filename
        filepath.write_text(json.dumps(report, indent=2, default=str))
        return filepath
