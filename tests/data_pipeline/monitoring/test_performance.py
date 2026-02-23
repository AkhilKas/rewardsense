from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from unittest.mock import patch

from data_pipeline.monitoring.performance import (
    PipelinePerformanceMonitor,
    TaskPerformanceLogger,
    timed_python_task,
)


def _mock_ti(task_id: str, seconds: int, state: str = "success"):
    start = datetime(2026, 2, 23, 4, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(seconds=seconds)
    return SimpleNamespace(
        task_id=task_id,
        state=state,
        start_date=start,
        end_date=end,
    )


def _mock_dag_run(run_id: str, total_seconds: int, task_seconds: dict[str, int]):
    start = datetime(2026, 2, 23, 4, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(seconds=total_seconds)
    tis = [_mock_ti(task_id, seconds) for task_id, seconds in task_seconds.items()]
    return SimpleNamespace(
        dag_id="rewardsense_data_pipeline",
        run_id=run_id,
        state="success",
        start_date=start,
        end_date=end,
        get_task_instances=lambda: tis,
    )


def test_timed_python_task_writes_jsonl():
    with patch.object(TaskPerformanceLogger, "append_event") as mock_append:

        @timed_python_task("unit.test_task")
        def _do_work():
            return {"status": "ok"}

        result = _do_work()
        assert result == {"status": "ok"}
        mock_append.assert_called_once()
        payload = mock_append.call_args.args[0]
        assert payload["task_name"] == "unit.test_task"
        assert payload["status"] == "success"


def test_generate_dashboard_contains_gantt_and_bottlenecks(tmp_path: Path):
    dag_run = _mock_dag_run(
        run_id="manual__dashboard",
        total_seconds=300,
        task_seconds={
            "preprocessing.clean_data": 80,
            "preprocessing.engineer_features": 60,
            "versioning.push_to_remote": 45,
        },
    )
    monitor = PipelinePerformanceMonitor(perf_dir=tmp_path / "perf")
    dashboard = monitor.generate_dashboard({"dag_run": dag_run})

    assert dashboard["airflow_gantt_available"] is True
    assert dashboard["summary"]["total_duration_sec"] == 300.0
    assert len(dashboard["gantt_task_spans"]) == 3
    assert dashboard["bottlenecks"][0]["task_id"] == "preprocessing.clean_data"
    assert Path(dashboard["dashboard_path"]).exists()


def test_detect_regression_flags_slow_run(tmp_path: Path):
    perf_dir = tmp_path / "perf"
    monitor = PipelinePerformanceMonitor(perf_dir=perf_dir, regression_threshold=0.2)

    baseline_runs = [
        {
            "run_id": "run_a",
            "total_duration_sec": 100.0,
            "task_durations": {"preprocessing.clean_data": 40.0},
        },
        {
            "run_id": "run_b",
            "total_duration_sec": 110.0,
            "task_durations": {"preprocessing.clean_data": 45.0},
        },
        {
            "run_id": "run_c",
            "total_duration_sec": 90.0,
            "task_durations": {"preprocessing.clean_data": 42.0},
        },
    ]
    for idx, payload in enumerate(baseline_runs):
        (perf_dir / f"perf_snapshot_20260223_04010{idx}.json").write_text(
            json.dumps(payload)
        )

    current = _mock_dag_run(
        run_id="run_new",
        total_seconds=170,
        task_seconds={"preprocessing.clean_data": 70},
    )
    result = monitor.detect_regression({"dag_run": current})

    assert result["regression_detected"] is True
    assert result["run_regression_ratio"] is not None
    assert result["task_regressions"][0]["task_id"] == "preprocessing.clean_data"
