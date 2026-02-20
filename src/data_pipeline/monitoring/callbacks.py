"""
RewardSense - Airflow Task / DAG Callbacks

Provides callback functions that Airflow invokes on task-level and
DAG-level state changes.  Each callback dispatches an alert through
the ``AlertDispatcher``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("airflow.task")


def _get_dispatcher():
    """Lazy-load an ``AlertDispatcher`` (deferred import to keep DAG parsing fast)."""
    from data_pipeline.monitoring.alerting import AlertDispatcher

    return AlertDispatcher()


# =========================================================================
# Task-level callbacks
# =========================================================================


def on_task_failure_callback(context: dict[str, Any]) -> None:
    """Called when **any** task fails.  Sends a CRITICAL alert."""
    from data_pipeline.monitoring.alerting import Severity

    ti = context.get("ti")
    task_id = ti.task_id if ti else "unknown"
    dag_id = ti.dag_id if ti else "unknown"
    execution_date = context.get("execution_date", "unknown")
    exception = context.get("exception", "N/A")

    message = (
        f"Task *{task_id}* in DAG *{dag_id}* failed!\n"
        f"Execution date: {execution_date}\n"
        f"Exception: {exception}"
    )

    logger.error("🚨 Task failure: %s.%s — %s", dag_id, task_id, exception)

    try:
        dispatcher = _get_dispatcher()
        dispatcher.dispatch(
            message=message,
            severity=Severity.CRITICAL,
            subject=f"Task Failure: {dag_id}.{task_id}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to dispatch failure alert: %s", exc)


def on_task_success_callback(context: dict[str, Any]) -> None:
    """Called when a task succeeds.  Logs an INFO-level metric."""
    ti = context.get("ti")
    task_id = ti.task_id if ti else "unknown"
    duration = None
    if ti and ti.start_date and ti.end_date:
        duration = round((ti.end_date - ti.start_date).total_seconds(), 2)

    logger.info(
        "✅ Task %s succeeded (duration: %s sec).",
        task_id,
        duration or "N/A",
    )


# =========================================================================
# DAG-level callbacks
# =========================================================================


def on_dag_success_callback(context: dict[str, Any]) -> None:
    """Called when the entire DAG run succeeds.  Sends an INFO summary alert."""
    from data_pipeline.monitoring.alerting import Severity

    dag_run = context.get("dag_run")
    dag_id = dag_run.dag_id if dag_run else "unknown"
    run_id = str(dag_run.run_id) if dag_run else "unknown"

    duration_str = "N/A"
    if dag_run and dag_run.start_date and dag_run.end_date:
        secs = (dag_run.end_date - dag_run.start_date).total_seconds()
        duration_str = f"{secs:.1f}s"

    message = (
        f"DAG *{dag_id}* completed successfully 🎉\n"
        f"Run: {run_id}\n"
        f"Duration: {duration_str}"
    )

    logger.info("🎉 DAG %s run %s completed successfully.", dag_id, run_id)

    try:
        dispatcher = _get_dispatcher()
        dispatcher.dispatch(
            message=message,
            severity=Severity.INFO,
            subject=f"Pipeline Success: {dag_id}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to dispatch success alert: %s", exc)
