"""
RewardSense - Pipeline Performance Monitoring

Provides:
1. Python-task timing instrumentation for Airflow callables.
2. Run-level performance snapshots with task spans for Gantt analysis.
3. Historical dashboard generation for trend visibility.
4. Regression detection against recent runs.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from statistics import median
from typing import Any, Callable

logger = logging.getLogger("airflow.task")

DEFAULT_PERF_DIR = Path("data/metrics/performance")
TASK_TIMINGS_PATH = DEFAULT_PERF_DIR / "task_timings.jsonl"


def _safe_utc_iso(dt: Any) -> str | None:
    if not dt:
        return None
    try:
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        try:
            return datetime.fromisoformat(str(dt)).astimezone(timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            return str(dt)


def timed_python_task(task_name: str | None = None) -> Callable[..., Any]:
    """
    Decorate a Python task callable and persist execution timing.

    Notes
    -----
    - This decorator targets PythonOperator callables.
    - BashOperator/EmptyOperator timings are still captured from DAG run task
      instances by ``PipelinePerformanceMonitor``.
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = task_name or func.__name__

        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = datetime.now(timezone.utc)
            t0 = time.perf_counter()
            status = "success"
            error: str | None = None
            result: Any = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                duration_sec = round(time.perf_counter() - t0, 3)
                finished_at = datetime.now(timezone.utc)
                TaskPerformanceLogger().append_event(
                    {
                        "event_type": "python_callable_timing",
                        "task_name": resolved_name,
                        "status": status,
                        "error": error,
                        "started_at": started_at.isoformat(),
                        "finished_at": finished_at.isoformat(),
                        "duration_sec": duration_sec,
                    }
                )
                logger.info("[PERF] %s finished in %.3fs", resolved_name, duration_sec)

        return _wrapper

    return _decorator


class TaskPerformanceLogger:
    """Append JSONL performance events for task-callable timings."""

    def __init__(self, perf_dir: Path | str = DEFAULT_PERF_DIR) -> None:
        self.perf_dir = Path(perf_dir)
        self.perf_dir.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: dict[str, Any]) -> None:
        path = self.perf_dir / TASK_TIMINGS_PATH.name
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")


class PipelinePerformanceMonitor:
    """Generate run snapshots, dashboard summaries, and regression signals."""

    def __init__(
        self,
        perf_dir: Path | str = DEFAULT_PERF_DIR,
        history_limit: int = 20,
        regression_threshold: float = 0.2,
    ) -> None:
        self.perf_dir = Path(perf_dir)
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.history_limit = max(3, history_limit)
        self.regression_threshold = max(0.05, regression_threshold)

    def generate_snapshot(self, context: dict[str, Any]) -> dict[str, Any]:
        """Collect task spans and persist a run-level performance snapshot."""
        dag_run = context.get("dag_run")
        snapshot = self._collect_snapshot(dag_run)
        snapshot_path = self._write_json("perf_snapshot", snapshot)
        snapshot["snapshot_path"] = str(snapshot_path)
        return snapshot

    def generate_dashboard(self, context: dict[str, Any]) -> dict[str, Any]:
        """Build a dashboard artifact with trends and bottlenecks."""
        current = self.generate_snapshot(context)
        history = self._load_history()
        dashboard = self._build_dashboard(current=current, history=history)
        dashboard_path = self._write_json("perf_dashboard", dashboard)
        dashboard["dashboard_path"] = str(dashboard_path)
        return dashboard

    def detect_regression(self, context: dict[str, Any]) -> dict[str, Any]:
        """Detect run and task-level regressions against recent history."""
        current = self.generate_snapshot(context)
        history = self._load_history()
        prior = [h for h in history if h.get("run_id") != current.get("run_id")]
        baseline_pool = prior[-10:]

        result: dict[str, Any] = {
            "run_id": current.get("run_id"),
            "dag_id": current.get("dag_id"),
            "regression_detected": False,
            "run_regression_ratio": None,
            "task_regressions": [],
        }
        if not baseline_pool:
            result["note"] = "insufficient_history"
            return result

        baseline_total = [
            r.get("total_duration_sec")
            for r in baseline_pool
            if isinstance(r.get("total_duration_sec"), (int, float))
        ]
        current_total = current.get("total_duration_sec")
        if baseline_total and isinstance(current_total, (int, float)):
            base = median(baseline_total)
            ratio = (current_total - base) / base if base > 0 else 0
            result["run_regression_ratio"] = round(ratio, 3)
            if ratio >= self.regression_threshold:
                result["regression_detected"] = True

        baseline_task_medians = self._task_duration_baseline(baseline_pool)
        for task_id, curr_dur in current.get("task_durations", {}).items():
            if not isinstance(curr_dur, (int, float)):
                continue
            base = baseline_task_medians.get(task_id)
            if not base or base <= 0:
                continue
            ratio = (curr_dur - base) / base
            if ratio >= self.regression_threshold and (curr_dur - base) >= 5:
                result["task_regressions"].append(
                    {
                        "task_id": task_id,
                        "current_duration_sec": round(curr_dur, 2),
                        "baseline_median_sec": round(base, 2),
                        "regression_ratio": round(ratio, 3),
                    }
                )

        if result["task_regressions"]:
            result["regression_detected"] = True
        return result

    @staticmethod
    def _collect_snapshot(dag_run: Any) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "dag_id": dag_run.dag_id if dag_run else "unknown",
            "run_id": str(dag_run.run_id) if dag_run else "unknown",
            "state": str(dag_run.state) if dag_run else "unknown",
            "airflow_gantt_available": True,
            "task_spans": [],
            "task_durations": {},
            "bottlenecks_top5": [],
            "critical_path_sec": None,
            "total_duration_sec": None,
        }

        if not dag_run:
            return snapshot

        run_start = dag_run.start_date
        run_end = dag_run.end_date or datetime.now(timezone.utc)
        if run_start:
            snapshot["total_duration_sec"] = round(
                (run_end - run_start).total_seconds(), 2
            )

        spans: list[dict[str, Any]] = []
        try:
            for ti in dag_run.get_task_instances():
                start = ti.start_date
                end = ti.end_date
                duration = None
                if start and end:
                    duration = round((end - start).total_seconds(), 2)
                    snapshot["task_durations"][ti.task_id] = duration
                else:
                    snapshot["task_durations"][ti.task_id] = None

                spans.append(
                    {
                        "task_id": ti.task_id,
                        "state": str(ti.state),
                        "start": _safe_utc_iso(start),
                        "end": _safe_utc_iso(end),
                        "duration_sec": duration,
                    }
                )
        except Exception:  # noqa: BLE001
            pass

        spans = sorted(spans, key=lambda s: (s["start"] is None, s["start"]))
        snapshot["task_spans"] = spans
        durations = [
            (s["task_id"], s["duration_sec"])
            for s in spans
            if isinstance(s.get("duration_sec"), (int, float))
        ]
        durations.sort(key=lambda pair: pair[1], reverse=True)
        snapshot["bottlenecks_top5"] = [
            {"task_id": task_id, "duration_sec": duration}
            for task_id, duration in durations[:5]
        ]
        if spans:
            starts = [s.get("start") for s in spans if s.get("start")]
            ends = [s.get("end") for s in spans if s.get("end")]
            if starts and ends:
                try:
                    min_start = min(datetime.fromisoformat(s) for s in starts)
                    max_end = max(datetime.fromisoformat(e) for e in ends)
                    snapshot["critical_path_sec"] = round(
                        (max_end - min_start).total_seconds(), 2
                    )
                except Exception:  # noqa: BLE001
                    pass
        return snapshot

    def _write_json(self, prefix: str, payload: dict[str, Any]) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = self.perf_dir / f"{prefix}_{ts}.json"
        filepath.write_text(json.dumps(payload, indent=2, default=str))
        latest = self.perf_dir / f"{prefix}_latest.json"
        latest.write_text(json.dumps(payload, indent=2, default=str))
        return filepath

    def _load_history(self) -> list[dict[str, Any]]:
        files = sorted(self.perf_dir.glob("perf_snapshot_*.json"))[
            -self.history_limit :
        ]
        out: list[dict[str, Any]] = []
        for file in files:
            try:
                out.append(json.loads(file.read_text()))
            except Exception:  # noqa: BLE001
                continue
        return out

    @staticmethod
    def _task_duration_baseline(history: list[dict[str, Any]]) -> dict[str, float]:
        bucket: dict[str, list[float]] = {}
        for run in history:
            for task_id, duration in run.get("task_durations", {}).items():
                if isinstance(duration, (int, float)):
                    bucket.setdefault(task_id, []).append(float(duration))
        return {task_id: median(values) for task_id, values in bucket.items() if values}

    def _build_dashboard(
        self, current: dict[str, Any], history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        baseline = self._task_duration_baseline(
            [h for h in history if h.get("run_id") != current.get("run_id")]
        )
        trend_rows: list[dict[str, Any]] = []
        for task_id, current_duration in current.get("task_durations", {}).items():
            if not isinstance(current_duration, (int, float)):
                continue
            baseline_duration = baseline.get(task_id)
            delta_pct = None
            if baseline_duration and baseline_duration > 0:
                delta_pct = round(
                    ((current_duration - baseline_duration) / baseline_duration) * 100,
                    2,
                )
            trend_rows.append(
                {
                    "task_id": task_id,
                    "current_duration_sec": round(current_duration, 2),
                    "baseline_median_sec": (
                        round(baseline_duration, 2) if baseline_duration else None
                    ),
                    "delta_percent": delta_pct,
                }
            )

        trend_rows.sort(
            key=lambda row: row["current_duration_sec"],
            reverse=True,
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dag_id": current.get("dag_id"),
            "run_id": current.get("run_id"),
            "airflow_gantt_available": True,
            "summary": {
                "total_duration_sec": current.get("total_duration_sec"),
                "critical_path_sec": current.get("critical_path_sec"),
                "history_window_runs": len(history),
            },
            "bottlenecks": current.get("bottlenecks_top5", []),
            "task_trends": trend_rows,
            "gantt_task_spans": current.get("task_spans", []),
        }
