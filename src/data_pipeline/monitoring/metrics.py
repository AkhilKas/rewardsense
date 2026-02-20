"""
RewardSense - Pipeline Metrics Logger

Records per-task and aggregate pipeline metrics (durations, record
counts, error counts) and writes them to ``data/metrics/``.
Optionally pushes to a Prometheus push-gateway if configured.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("airflow.task")

DEFAULT_METRICS_DIR = Path("data/metrics")


class PipelineMetricsLogger:
    """Collect and persist pipeline execution metrics.

    Parameters
    ----------
    metrics_dir : Path | str, optional
        Directory where metric JSON files are stored.
    """

    def __init__(self, metrics_dir: Path | str = DEFAULT_METRICS_DIR) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metrics(self, context: dict[str, Any]) -> dict[str, Any]:
        """Extract and persist metrics from the Airflow *context*.

        Parameters
        ----------
        context : dict
            Airflow ``**context`` dict.

        Returns
        -------
        dict
            Summary of collected metrics (pushed to XCom).
        """
        dag_run = context.get("dag_run")
        metrics = self._collect(dag_run)

        filepath = self._write(metrics)
        logger.info("📈 Pipeline metrics written to %s", filepath)

        # Optional: push to Prometheus
        self._push_prometheus(metrics)

        return {
            "metrics_path": str(filepath),
            "total_tasks": metrics.get("total_tasks", 0),
            "failed_tasks": metrics.get("failed_tasks", 0),
            "status": "completed",
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _collect(dag_run: Any) -> dict[str, Any]:
        """Build a metrics dict from the DAG run."""
        metrics: dict[str, Any] = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        if dag_run is None:
            return metrics

        metrics["dag_id"] = dag_run.dag_id
        metrics["run_id"] = str(dag_run.run_id)
        metrics["state"] = str(dag_run.state)

        if dag_run.start_date:
            end = dag_run.end_date or datetime.now(timezone.utc)
            metrics["total_duration_sec"] = round(
                (end - dag_run.start_date).total_seconds(), 2
            )

        # Per-task stats
        task_metrics: list[dict[str, Any]] = []
        total = 0
        failed = 0
        try:
            for ti in dag_run.get_task_instances():
                total += 1
                entry: dict[str, Any] = {
                    "task_id": ti.task_id,
                    "state": str(ti.state),
                }
                if ti.start_date and ti.end_date:
                    entry["duration_sec"] = round(
                        (ti.end_date - ti.start_date).total_seconds(), 2
                    )
                if str(ti.state) == "failed":
                    failed += 1
                task_metrics.append(entry)
        except Exception:  # noqa: BLE001
            pass

        metrics["total_tasks"] = total
        metrics["failed_tasks"] = failed
        metrics["tasks"] = task_metrics
        return metrics

    def _write(self, metrics: dict[str, Any]) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = self.metrics_dir / f"pipeline_metrics_{ts}.json"
        filepath.write_text(json.dumps(metrics, indent=2, default=str))
        return filepath

    @staticmethod
    def _push_prometheus(metrics: dict[str, Any]) -> None:
        """Push metrics to Prometheus push-gateway if configured."""
        gateway = os.getenv("PROMETHEUS_PUSHGATEWAY_URL")
        if not gateway:
            return

        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

            registry = CollectorRegistry()

            duration_gauge = Gauge(
                "rewardsense_pipeline_duration_seconds",
                "Total pipeline duration",
                registry=registry,
            )
            duration_gauge.set(metrics.get("total_duration_sec", 0))

            failed_gauge = Gauge(
                "rewardsense_pipeline_failed_tasks",
                "Number of failed tasks",
                registry=registry,
            )
            failed_gauge.set(metrics.get("failed_tasks", 0))

            push_to_gateway(gateway, job="rewardsense_pipeline", registry=registry)
            logger.info("Metrics pushed to Prometheus gateway at %s", gateway)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to push to Prometheus: %s", exc)
