"""
Monitoring Pipeline DAG

Daily DAG that:
  1. Collects inference logs from GCS
  2. Runs Evidently drift detection against training reference
  3. Computes serving performance metrics
  4. Evaluates thresholds and decides whether to trigger retraining
  5. Triggers the model pipeline DAG if drift/decay detected
  6. Sends Slack notification regardless of drift status

Schedule: Daily at 06:00 UTC
Dependencies: Requires inference logs from the serving API

Guards:
  - Max 1 retrain per 24 hours (prevents thrashing)
  - Skip retrain if model pipeline DAG is already running
  - Log all trigger decisions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable, DagRun
from airflow.utils.session import create_session

logger = logging.getLogger("airflow.task")

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------
DAG_ID = "rewardsense_monitoring_pipeline"
MODEL_PIPELINE_DAG_ID = "rewardsense_model_pipeline"
SCHEDULE = "0 6 * * *"  # Daily at 06:00 UTC

default_args = {
    "owner": "rewardsense",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _collect_inference_data(**context):
    """Task 1: Collect last 7 days of inference logs."""
    from src.monitoring.data_collector import InferenceDataCollector

    collector = InferenceDataCollector()
    summary = collector.collect(days=7)

    logger.info(
        "Collected %d inference records across %d model version(s)",
        summary.total_records,
        len(summary.model_versions),
    )

    if summary.total_records == 0:
        logger.warning("No inference logs found for the last 7 days")

    # Push summary stats to XCom for downstream tasks
    context["ti"].xcom_push(key="summary_stats", value=summary.summary_stats)
    context["ti"].xcom_push(key="total_records", value=summary.total_records)

    # Serialize DataFrames to JSON for downstream tasks
    if summary.input_features_df is not None and not summary.input_features_df.empty:
        context["ti"].xcom_push(
            key="input_features_json",
            value=summary.input_features_df.to_json(),
        )
    if summary.predictions_df is not None and not summary.predictions_df.empty:
        context["ti"].xcom_push(
            key="predictions_json",
            value=summary.predictions_df.to_json(),
        )
    if summary.latency_df is not None and not summary.latency_df.empty:
        context["ti"].xcom_push(
            key="latency_json",
            value=summary.latency_df.to_json(),
        )
    if summary.metadata_df is not None and not summary.metadata_df.empty:
        context["ti"].xcom_push(
            key="metadata_json",
            value=summary.metadata_df.to_json(),
        )

    return {"status": "success", "total_records": summary.total_records}


def _run_drift_detection(**context):
    """Task 2: Run Evidently drift detection."""
    import pandas as pd
    from src.monitoring.drift_detector import DriftDetector

    ti = context["ti"]
    input_json = ti.xcom_pull(
        task_ids="collect_inference_data", key="input_features_json"
    )

    if not input_json:
        logger.warning("No input features data — skipping drift detection")
        context["ti"].xcom_push(key="drift_detected", value=False)
        return {"status": "skipped", "reason": "no_data"}

    current_df = pd.read_json(input_json)

    reference_path = Variable.get(
        "DRIFT_REFERENCE_PATH",
        default_var="data/reference/training_reference.csv",
    )

    detector = DriftDetector(reference_path=reference_path)

    try:
        result = detector.detect(current_df=current_df)
    except FileNotFoundError:
        logger.warning("Reference dataset not found — skipping drift detection")
        context["ti"].xcom_push(key="drift_detected", value=False)
        return {"status": "skipped", "reason": "no_reference"}

    ti.xcom_push(key="drift_detected", value=result.drift_detected)
    ti.xcom_push(key="drift_share", value=result.dataset_drift_share)
    ti.xcom_push(key="drifted_features", value=result.drifted_features)
    ti.xcom_push(key="drift_report_path", value=result.html_report_path)
    ti.xcom_push(key="drift_result_json", value=json.dumps(result.summary, default=str))

    logger.info(
        "Drift detection: detected=%s, share=%.2f%%, features=%s",
        result.drift_detected,
        result.dataset_drift_share * 100,
        result.drifted_features,
    )

    return {"status": "success", "drift_detected": result.drift_detected}


def _compute_performance_metrics(**context):
    """Task 3: Compute serving performance metrics."""
    import pandas as pd
    from src.monitoring.data_collector import InferenceDataSummary
    from src.monitoring.performance_tracker import PerformanceTracker

    ti = context["ti"]

    # Reconstruct summary from XCom
    summary = InferenceDataSummary(
        start_date=datetime.utcnow() - timedelta(days=7),
        end_date=datetime.utcnow(),
        total_records=ti.xcom_pull(
            task_ids="collect_inference_data", key="total_records"
        )
        or 0,
    )

    for key, attr in [
        ("latency_json", "latency_df"),
        ("predictions_json", "predictions_df"),
        ("metadata_json", "metadata_df"),
    ]:
        json_data = ti.xcom_pull(task_ids="collect_inference_data", key=key)
        if json_data:
            setattr(summary, attr, pd.read_json(json_data))

    tracker = PerformanceTracker()
    snapshot = tracker.compute(summary)
    tracker.save_snapshot(snapshot)

    ti.xcom_push(key="has_alerts", value=snapshot.has_alerts)
    ti.xcom_push(key="alerts", value=snapshot.alerts)
    ti.xcom_push(key="latency_p95", value=snapshot.latency_p95_ms)
    ti.xcom_push(
        key="perf_snapshot_json",
        value=json.dumps(snapshot.to_dict(), default=str),
    )

    logger.info(
        "Performance: %d requests, p95=%.0fms, alerts=%d",
        snapshot.total_requests,
        snapshot.latency_p95_ms,
        len(snapshot.alerts),
    )

    return {"status": "success", "has_alerts": snapshot.has_alerts}


def _evaluate_thresholds(**context):
    """Task 4: Decide whether to trigger retraining.

    Returns the branch task_id to follow:
      - 'trigger_retrain' if drift or decay detected
      - 'skip_retrain' if everything is healthy
    """
    ti = context["ti"]

    drift_detected = ti.xcom_pull(task_ids="run_drift_detection", key="drift_detected")
    has_alerts = ti.xcom_pull(task_ids="compute_performance_metrics", key="has_alerts")

    reasons = []

    if drift_detected:
        drift_share = (
            ti.xcom_pull(task_ids="run_drift_detection", key="drift_share") or 0
        )
        reasons.append(f"data_drift (share={drift_share:.2%})")

    if has_alerts:
        alerts = (
            ti.xcom_pull(task_ids="compute_performance_metrics", key="alerts") or []
        )
        for alert in alerts:
            reasons.append(f"performance_alert: {alert}")

    should_retrain = len(reasons) > 0

    ti.xcom_push(key="should_retrain", value=should_retrain)
    ti.xcom_push(key="retrain_reasons", value=reasons)

    logger.info(
        "Threshold evaluation: retrain=%s, reasons=%s",
        should_retrain,
        reasons,
    )

    if should_retrain:
        return "trigger_retrain"
    return "skip_retrain"


def _trigger_retrain(**context):
    """Task 5: Trigger the model pipeline DAG for retraining (Story 5.1).

    Guards:
      - Max 1 retrain per 24 hours
      - Skip if model pipeline DAG is already running
    """
    from airflow.api.common.trigger_dag import trigger_dag

    ti = context["ti"]
    reasons = ti.xcom_pull(task_ids="evaluate_thresholds", key="retrain_reasons") or []
    drift_report = ti.xcom_pull(task_ids="run_drift_detection", key="drift_report_path")

    # --- Guard: check if already running ---
    with create_session() as session:
        running = (
            session.query(DagRun)
            .filter(
                DagRun.dag_id == MODEL_PIPELINE_DAG_ID,
                DagRun.state == "running",
            )
            .first()
        )
        if running:
            logger.warning(
                "Model pipeline DAG is already running (run_id=%s) — skipping retrain",
                running.run_id,
            )
            ti.xcom_push(key="retrain_triggered", value=False)
            ti.xcom_push(key="retrain_skip_reason", value="already_running")
            return {"status": "skipped", "reason": "already_running"}

    # --- Guard: max 1 retrain per 24 hours ---
    max_retrain_hours = int(
        Variable.get("MAX_RETRAIN_INTERVAL_HOURS", default_var="24")
    )
    with create_session() as session:
        recent = (
            session.query(DagRun)
            .filter(
                DagRun.dag_id == MODEL_PIPELINE_DAG_ID,
                DagRun.execution_date
                >= datetime.utcnow() - timedelta(hours=max_retrain_hours),
            )
            .first()
        )
        if recent:
            logger.warning(
                "Model pipeline ran within last %d hours (run_id=%s) — skipping retrain",
                max_retrain_hours,
                recent.run_id,
            )
            ti.xcom_push(key="retrain_triggered", value=False)
            ti.xcom_push(key="retrain_skip_reason", value="cooldown_active")
            return {"status": "skipped", "reason": "cooldown_active"}

    # --- Trigger ---
    run_id = f"monitoring_triggered_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    conf = {
        "trigger_reason": "; ".join(reasons),
        "drift_report_path": drift_report,
        "triggered_by": DAG_ID,
        "trigger_timestamp": datetime.utcnow().isoformat(),
    }

    trigger_dag(
        dag_id=MODEL_PIPELINE_DAG_ID,
        run_id=run_id,
        conf=conf,
        replace_microseconds=False,
    )

    ti.xcom_push(key="retrain_triggered", value=True)
    ti.xcom_push(key="retrain_run_id", value=run_id)

    logger.info(
        "Triggered model pipeline DAG (run_id=%s, reasons=%s)",
        run_id,
        reasons,
    )

    return {"status": "triggered", "run_id": run_id}


def _skip_retrain(**context):
    """No-op task when retraining is not needed."""
    logger.info("No retraining needed — all metrics within thresholds")
    context["ti"].xcom_push(key="retrain_triggered", value=False)
    return {"status": "skipped", "reason": "healthy"}


def _send_notification(**context):
    """Task 6: Send Slack notification with monitoring summary."""
    from src.monitoring.notifier import SlackNotifier

    ti = context["ti"]

    drift_json = ti.xcom_pull(task_ids="run_drift_detection", key="drift_result_json")
    perf_json = ti.xcom_pull(
        task_ids="compute_performance_metrics", key="perf_snapshot_json"
    )
    retrain_triggered = ti.xcom_pull(
        task_ids="trigger_retrain", key="retrain_triggered"
    )

    # Use whichever branch ran
    was_triggered = retrain_triggered if retrain_triggered is not None else False

    notifier = SlackNotifier()

    # Create lightweight objects for the notifier
    class _DriftProxy:
        def __init__(self, data):
            self.summary = json.loads(data) if isinstance(data, str) else (data or {})

    class _PerfProxy:
        def __init__(self, data):
            self._data = json.loads(data) if isinstance(data, str) else (data or {})

        def to_dict(self):
            return self._data

    drift_proxy = _DriftProxy(drift_json)
    perf_proxy = _PerfProxy(perf_json)

    notifier.send_monitoring_summary(drift_proxy, perf_proxy)

    if was_triggered:
        reasons = (
            ti.xcom_pull(task_ids="evaluate_thresholds", key="retrain_reasons") or []
        )
        drift_report = ti.xcom_pull(
            task_ids="run_drift_detection", key="drift_report_path"
        )
        notifier.send_retrain_trigger(
            reason="; ".join(reasons),
            drift_report_path=drift_report,
        )

    logger.info("Monitoring notification sent (retrain_triggered=%s)", was_triggered)

    return {"status": "notified", "retrain_triggered": was_triggered}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Daily monitoring: drift detection, performance tracking, retrain trigger",
    schedule_interval=SCHEDULE,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["monitoring", "rewardsense", "phase3"],
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    collect = PythonOperator(
        task_id="collect_inference_data",
        python_callable=_collect_inference_data,
        doc_md="Collect last 7 days of inference logs from GCS.",
    )

    drift = PythonOperator(
        task_id="run_drift_detection",
        python_callable=_run_drift_detection,
        doc_md="Run Evidently data drift detection against training reference.",
    )

    performance = PythonOperator(
        task_id="compute_performance_metrics",
        python_callable=_compute_performance_metrics,
        doc_md="Compute latency percentiles, score distributions, alerts.",
    )

    evaluate = BranchPythonOperator(
        task_id="evaluate_thresholds",
        python_callable=_evaluate_thresholds,
        doc_md="Decide whether to trigger retraining based on drift + performance.",
    )

    retrain = PythonOperator(
        task_id="trigger_retrain",
        python_callable=_trigger_retrain,
        doc_md="Trigger model pipeline DAG with guards (max 1/24h, no concurrent).",
    )

    skip = PythonOperator(
        task_id="skip_retrain",
        python_callable=_skip_retrain,
        doc_md="No-op when retraining is not needed.",
    )

    notify = PythonOperator(
        task_id="send_notification",
        python_callable=_send_notification,
        trigger_rule="none_failed_min_one_success",
        doc_md="Send Slack notification with monitoring summary.",
    )

    # Task dependencies
    #
    # start → collect → [drift, performance] → evaluate
    #                                             ├── trigger_retrain ──┐
    #                                             └── skip_retrain ─────┤
    #                                                                   └── notify → end
    start >> collect
    collect >> [drift, performance]
    [drift, performance] >> evaluate
    evaluate >> [retrain, skip]
    [retrain, skip] >> notify
    notify >> end
