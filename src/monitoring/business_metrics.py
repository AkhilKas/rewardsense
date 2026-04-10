"""
Business Metrics Collection.

Reads inference log records (from GCS or local filesystem, same
source as ``InferenceDataCollector``) and computes aggregated
business metrics: request counts, stage latencies, LLM call counts,
estimated token/cost metrics, success/fallback/error rates, and
top-used recommendation flows.

Aggregation is by day and configurable report window so the
on-demand report generator can produce period summaries
without exposing raw records to the demo UI.

Usage::

    collector = BusinessMetricsCollector()
    period = collector.collect(days=7)
    print(period.total_requests)
    print(period.daily_breakdown[0].avg_latency_ms)
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GCS_INFERENCE_LOG_BUCKET: str = os.getenv(
    "INFERENCE_LOG_BUCKET", "rewardsense-inference-logs"
)
LOCAL_LOG_DIR: str = os.getenv(
    "LOCAL_INFERENCE_LOG_DIR", "/tmp/rewardsense-inference-logs"
)

# Lazy GCS import
try:
    from google.cloud import storage as gcs_storage

    GCS_AVAILABLE = True
except ImportError:
    gcs_storage = None  # type: ignore[assignment]
    GCS_AVAILABLE = False

# Estimated cost per 1K tokens (input + output blended)
DEFAULT_COST_PER_1K_TOKENS: float = float(os.getenv("LLM_COST_PER_1K_TOKENS", "0.005"))


# =====================================================================
# Data classes
# =====================================================================


@dataclass
class LatencyStats:
    """Latency statistics for a set of requests."""

    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_values(cls, values: List[float]) -> "LatencyStats":
        if not values:
            return cls()
        sorted_v = sorted(values)
        n = len(sorted_v)
        return cls(
            mean_ms=round(statistics.mean(sorted_v), 3),
            median_ms=round(statistics.median(sorted_v), 3),
            p50_ms=round(sorted_v[int(n * 0.50)], 3),
            p95_ms=round(sorted_v[min(int(n * 0.95), n - 1)], 3),
            p99_ms=round(sorted_v[min(int(n * 0.99), n - 1)], 3),
            min_ms=round(sorted_v[0], 3),
            max_ms=round(sorted_v[-1], 3),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class StageLatencyBreakdown:
    """Per-stage latency averages."""

    normalize_ms: float = 0.0
    deterministic_ms: float = 0.0
    personalization_ms: float = 0.0
    rank_ms: float = 0.0
    llm_explanation_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "normalize_ms": round(self.normalize_ms, 3),
            "deterministic_ms": round(self.deterministic_ms, 3),
            "personalization_ms": round(self.personalization_ms, 3),
            "rank_ms": round(self.rank_ms, 3),
            "llm_explanation_ms": round(self.llm_explanation_ms, 3),
        }


@dataclass
class LLMMetrics:
    """Aggregated LLM telemetry."""

    total_calls: int = 0
    total_successes: int = 0
    total_fallbacks: int = 0
    success_rate: float = 0.0
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    avg_explanation_latency_ms: float = 0.0
    model_names_used: List[str] = field(default_factory=list)
    prompt_versions_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_fallbacks": self.total_fallbacks,
            "success_rate": round(self.success_rate, 4),
            "estimated_tokens": self.estimated_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "avg_explanation_latency_ms": round(self.avg_explanation_latency_ms, 3),
            "model_names_used": self.model_names_used,
            "prompt_versions_used": self.prompt_versions_used,
        }


@dataclass
class DailyMetrics:
    """Business metrics aggregated for a single day."""

    date: str = ""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)
    stage_latency: StageLatencyBreakdown = field(default_factory=StageLatencyBreakdown)
    llm: LLMMetrics = field(default_factory=LLMMetrics)
    flow_counts: Dict[str, int] = field(default_factory=dict)
    top_cards: Dict[str, int] = field(default_factory=dict)
    personalization_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "latency": self.latency.to_dict(),
            "stage_latency": self.stage_latency.to_dict(),
            "llm": self.llm.to_dict(),
            "flow_counts": self.flow_counts,
            "top_cards": dict(
                sorted(self.top_cards.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "personalization_rate": round(self.personalization_rate, 4),
        }


@dataclass
class PeriodMetrics:
    """Business metrics aggregated over a date range."""

    start_date: str = ""
    end_date: str = ""
    total_requests: int = 0
    total_successes: int = 0
    total_errors: int = 0
    overall_error_rate: float = 0.0
    overall_latency: LatencyStats = field(default_factory=LatencyStats)
    overall_stage_latency: StageLatencyBreakdown = field(
        default_factory=StageLatencyBreakdown
    )
    overall_llm: LLMMetrics = field(default_factory=LLMMetrics)
    overall_flow_counts: Dict[str, int] = field(default_factory=dict)
    overall_top_cards: Dict[str, int] = field(default_factory=dict)
    overall_personalization_rate: float = 0.0
    daily_breakdown: List[DailyMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_errors": self.total_errors,
            "overall_error_rate": round(self.overall_error_rate, 4),
            "overall_latency": self.overall_latency.to_dict(),
            "overall_stage_latency": self.overall_stage_latency.to_dict(),
            "overall_llm": self.overall_llm.to_dict(),
            "overall_flow_counts": self.overall_flow_counts,
            "overall_top_cards": dict(
                sorted(
                    self.overall_top_cards.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
            "overall_personalization_rate": round(self.overall_personalization_rate, 4),
            "daily_breakdown": [day.to_dict() for day in self.daily_breakdown],
        }


# =====================================================================
# Collector
# =====================================================================


class BusinessMetricsCollector:
    """Reads inference logs and computes business metrics.

    Follows the same GCS-first, local-fallback pattern as the Phase 3
    ``InferenceDataCollector``.
    """

    def __init__(
        self,
        local_log_dir: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        cost_per_1k_tokens: Optional[float] = None,
    ) -> None:
        self._local_dir = Path(local_log_dir or LOCAL_LOG_DIR)
        self._gcs_bucket = gcs_bucket or GCS_INFERENCE_LOG_BUCKET
        self._cost_per_1k = cost_per_1k_tokens or DEFAULT_COST_PER_1K_TOKENS

    # -----------------------------------------------------------------
    # Log loading
    # -----------------------------------------------------------------

    def _load_records_local(
        self, start: datetime, end: datetime
    ) -> List[Dict[str, Any]]:
        """Load JSON log records from local filesystem for the date range."""
        records: List[Dict[str, Any]] = []
        current = start.date()
        end_date = end.date()

        while current <= end_date:
            day_dir = (
                self._local_dir
                / f"{current.year:04d}"
                / f"{current.month:02d}"
                / f"{current.day:02d}"
            )
            if day_dir.exists():
                for filepath in day_dir.glob("*.json"):
                    try:
                        data = json.loads(filepath.read_text(encoding="utf-8"))
                        if isinstance(data, dict):
                            records.append(data)
                    except Exception as exc:
                        logger.debug("Skipping %s: %s", filepath, exc)
            current += timedelta(days=1)

        return records

    def _load_records_gcs(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Load JSON log records from GCS for the date range."""
        if not GCS_AVAILABLE:
            return []

        try:
            from google.cloud import storage as gcs_mod

            client = gcs_mod.Client()
        except Exception:
            return []

        records: List[Dict[str, Any]] = []
        current = start.date()
        end_date = end.date()

        try:
            bucket = client.bucket(self._gcs_bucket)
        except Exception:
            return []

        while current <= end_date:
            prefix = f"{current.year:04d}/{current.month:02d}/{current.day:02d}/"
            try:
                blobs = list(bucket.list_blobs(prefix=prefix))
                for blob in blobs:
                    if not blob.name.endswith(".json"):
                        continue
                    try:
                        content = blob.download_as_text(encoding="utf-8")
                        data = json.loads(content)
                        if isinstance(data, dict):
                            records.append(data)
                    except Exception as exc:
                        logger.debug("Skipping GCS blob %s: %s", blob.name, exc)
            except Exception as exc:
                logger.debug("GCS list failed for %s: %s", prefix, exc)
            current += timedelta(days=1)

        return records

    def _load_records(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Load records from GCS first, fall back to local."""
        records = self._load_records_gcs(start, end)
        if records:
            return records
        return self._load_records_local(start, end)

    # -----------------------------------------------------------------
    # Aggregation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _aggregate_daily(
        self, date_str: str, records: List[Dict[str, Any]]
    ) -> DailyMetrics:
        """Compute metrics for a single day's records."""
        if not records:
            return DailyMetrics(date=date_str)

        total_latencies: List[float] = []
        stage_totals: Dict[str, List[float]] = defaultdict(list)
        explanation_latencies: List[float] = []
        flow_counter: Counter = Counter()
        card_counter: Counter = Counter()
        success_count = 0
        error_count = 0
        personalized_count = 0

        # LLM accumulators
        llm_calls = 0
        llm_successes = 0
        llm_fallbacks = 0
        llm_tokens = 0
        model_names: set = set()
        prompt_versions: set = set()

        for record in records:
            # Request status
            status = record.get("request_status", "success")
            if status == "error":
                error_count += 1
            else:
                success_count += 1

            # Latency
            breakdown = record.get("latency_breakdown_ms", {})
            total_ms = self._safe_float(breakdown.get("total"))
            if total_ms > 0:
                total_latencies.append(total_ms)

            for stage in (
                "normalize",
                "deterministic",
                "personalization",
                "rank",
                "llm_explanation",
            ):
                val = self._safe_float(breakdown.get(stage))
                if val > 0:
                    stage_totals[stage].append(val)

            # Explanation latency (top-level field, pre-Story-5.1 compat)
            expl_lat = self._safe_float(record.get("explanation_latency_ms"))
            if expl_lat > 0:
                explanation_latencies.append(expl_lat)

            # Recommendation flow
            flow = record.get("recommendation_flow", "predict")
            flow_counter[flow] += 1

            # Top card
            top_card = record.get("top_card", "")
            if top_card and top_card != "none":
                card_counter[top_card] += 1

            # Personalization
            if record.get("is_personalized", False):
                personalized_count += 1

            # LLM telemetry (Story 5.1 enriched records)
            llm_telem = record.get("llm_telemetry")
            if llm_telem and isinstance(llm_telem, dict):
                llm_calls += int(llm_telem.get("llm_calls", 0))
                llm_successes += int(llm_telem.get("llm_successes", 0))
                llm_fallbacks += int(llm_telem.get("llm_fallbacks", 0))
                llm_tokens += int(llm_telem.get("llm_token_estimate", 0))
                model_name = llm_telem.get("llm_model_name")
                if model_name:
                    model_names.add(str(model_name))
                prompt_ver = llm_telem.get("llm_prompt_version")
                if prompt_ver:
                    prompt_versions.add(str(prompt_ver))

        n = len(records)

        # Stage latency averages
        stage_breakdown = StageLatencyBreakdown(
            normalize_ms=(
                statistics.mean(stage_totals["normalize"])
                if stage_totals["normalize"]
                else 0.0
            ),
            deterministic_ms=(
                statistics.mean(stage_totals["deterministic"])
                if stage_totals["deterministic"]
                else 0.0
            ),
            personalization_ms=(
                statistics.mean(stage_totals["personalization"])
                if stage_totals["personalization"]
                else 0.0
            ),
            rank_ms=(
                statistics.mean(stage_totals["rank"]) if stage_totals["rank"] else 0.0
            ),
            llm_explanation_ms=(
                statistics.mean(stage_totals["llm_explanation"])
                if stage_totals["llm_explanation"]
                else 0.0
            ),
        )

        # LLM metrics
        llm_metrics = LLMMetrics(
            total_calls=llm_calls,
            total_successes=llm_successes,
            total_fallbacks=llm_fallbacks,
            success_rate=(llm_successes / llm_calls) if llm_calls > 0 else 0.0,
            estimated_tokens=llm_tokens,
            estimated_cost_usd=(llm_tokens / 1000.0) * self._cost_per_1k,
            avg_explanation_latency_ms=(
                statistics.mean(explanation_latencies) if explanation_latencies else 0.0
            ),
            model_names_used=sorted(model_names),
            prompt_versions_used=sorted(prompt_versions),
        )

        return DailyMetrics(
            date=date_str,
            request_count=n,
            success_count=success_count,
            error_count=error_count,
            error_rate=error_count / n if n > 0 else 0.0,
            latency=LatencyStats.from_values(total_latencies),
            stage_latency=stage_breakdown,
            llm=llm_metrics,
            flow_counts=dict(flow_counter),
            top_cards=dict(card_counter),
            personalization_rate=personalized_count / n if n > 0 else 0.0,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def collect(
        self,
        days: int = 7,
        end_date: Optional[datetime] = None,
    ) -> PeriodMetrics:
        """Collect and aggregate business metrics for the last *days* days.

        Parameters
        ----------
        days : int
            Number of days to look back (inclusive of end_date).
        end_date : datetime, optional
            End of the reporting window.  Defaults to now (UTC).

        Returns
        -------
        PeriodMetrics
            Aggregated metrics with daily breakdown.
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days - 1)

        records = self._load_records(start_date, end_date)
        if not records:
            daily: List[DailyMetrics] = []
            current = start_date.date()
            end_d = end_date.date()
            while current <= end_d:
                daily.append(DailyMetrics(date=current.strftime("%Y-%m-%d")))
                current += timedelta(days=1)
            return PeriodMetrics(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                daily_breakdown=daily,
            )

        # Group by date
        by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            ts = record.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                day_key = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                day_key = "unknown"
            by_date[day_key].append(record)

        # Compute daily metrics
        daily: List[DailyMetrics] = []
        current = start_date.date()
        end_d = end_date.date()
        while current <= end_d:
            day_str = current.strftime("%Y-%m-%d")
            day_records = by_date.get(day_str, [])
            daily.append(self._aggregate_daily(day_str, day_records))
            current += timedelta(days=1)

        # Compute period-level aggregates
        all_latencies: List[float] = []
        all_stage: Dict[str, List[float]] = defaultdict(list)
        all_expl_lat: List[float] = []
        total_requests = 0
        total_successes = 0
        total_errors = 0
        total_personalized = 0
        flow_counter: Counter = Counter()
        card_counter: Counter = Counter()
        total_llm_calls = 0
        total_llm_successes = 0
        total_llm_fallbacks = 0
        total_llm_tokens = 0
        all_model_names: set = set()
        all_prompt_versions: set = set()

        for day in daily:
            total_requests += day.request_count
            total_successes += day.success_count
            total_errors += day.error_count
            total_personalized += int(day.personalization_rate * day.request_count)
            flow_counter.update(day.flow_counts)
            card_counter.update(day.top_cards)

            # LLM
            total_llm_calls += day.llm.total_calls
            total_llm_successes += day.llm.total_successes
            total_llm_fallbacks += day.llm.total_fallbacks
            total_llm_tokens += day.llm.estimated_tokens
            all_model_names.update(day.llm.model_names_used)
            all_prompt_versions.update(day.llm.prompt_versions_used)

        # Re-compute latencies from raw records for period-level percentiles
        for record in records:
            breakdown = record.get("latency_breakdown_ms", {})
            total_ms = self._safe_float(breakdown.get("total"))
            if total_ms > 0:
                all_latencies.append(total_ms)
            for stage in (
                "normalize",
                "deterministic",
                "personalization",
                "rank",
                "llm_explanation",
            ):
                val = self._safe_float(breakdown.get(stage))
                if val > 0:
                    all_stage[stage].append(val)
            expl = self._safe_float(record.get("explanation_latency_ms"))
            if expl > 0:
                all_expl_lat.append(expl)

        period_stage = StageLatencyBreakdown(
            normalize_ms=(
                statistics.mean(all_stage["normalize"])
                if all_stage["normalize"]
                else 0.0
            ),
            deterministic_ms=(
                statistics.mean(all_stage["deterministic"])
                if all_stage["deterministic"]
                else 0.0
            ),
            personalization_ms=(
                statistics.mean(all_stage["personalization"])
                if all_stage["personalization"]
                else 0.0
            ),
            rank_ms=(statistics.mean(all_stage["rank"]) if all_stage["rank"] else 0.0),
            llm_explanation_ms=(
                statistics.mean(all_stage["llm_explanation"])
                if all_stage["llm_explanation"]
                else 0.0
            ),
        )

        period_llm = LLMMetrics(
            total_calls=total_llm_calls,
            total_successes=total_llm_successes,
            total_fallbacks=total_llm_fallbacks,
            success_rate=(
                total_llm_successes / total_llm_calls if total_llm_calls > 0 else 0.0
            ),
            estimated_tokens=total_llm_tokens,
            estimated_cost_usd=(total_llm_tokens / 1000.0) * self._cost_per_1k,
            avg_explanation_latency_ms=(
                statistics.mean(all_expl_lat) if all_expl_lat else 0.0
            ),
            model_names_used=sorted(all_model_names),
            prompt_versions_used=sorted(all_prompt_versions),
        )

        return PeriodMetrics(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            total_requests=total_requests,
            total_successes=total_successes,
            total_errors=total_errors,
            overall_error_rate=(
                total_errors / total_requests if total_requests > 0 else 0.0
            ),
            overall_latency=LatencyStats.from_values(all_latencies),
            overall_stage_latency=period_stage,
            overall_llm=period_llm,
            overall_flow_counts=dict(flow_counter),
            overall_top_cards=dict(card_counter),
            overall_personalization_rate=(
                total_personalized / total_requests if total_requests > 0 else 0.0
            ),
            daily_breakdown=daily,
        )

    def collect_from_records(
        self,
        records: List[Dict[str, Any]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> PeriodMetrics:
        """Compute metrics from pre-loaded records (useful for testing).

        Parameters
        ----------
        records : list of dict
            Raw inference log records.
        start_date, end_date : str, optional
            Override date labels for the period.
        """
        if not records:
            return PeriodMetrics(
                start_date=start_date or "",
                end_date=end_date or "",
            )

        # Group by date
        by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            ts = record.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                day_key = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                day_key = "unknown"
            by_date[day_key].append(record)

        daily = [
            self._aggregate_daily(day, recs) for day, recs in sorted(by_date.items())
        ]

        # Re-derive period-level from daily (same logic as collect)
        all_latencies: List[float] = []
        for record in records:
            total_ms = self._safe_float(
                record.get("latency_breakdown_ms", {}).get("total")
            )
            if total_ms > 0:
                all_latencies.append(total_ms)

        total_requests = sum(d.request_count for d in daily)
        total_errors = sum(d.error_count for d in daily)
        total_llm_calls = sum(d.llm.total_calls for d in daily)
        total_llm_successes = sum(d.llm.total_successes for d in daily)
        total_llm_tokens = sum(d.llm.estimated_tokens for d in daily)

        flow_counter: Counter = Counter()
        card_counter: Counter = Counter()
        for d in daily:
            flow_counter.update(d.flow_counts)
            card_counter.update(d.top_cards)

        all_model_names: set = set()
        all_prompt_versions: set = set()
        for d in daily:
            all_model_names.update(d.llm.model_names_used)
            all_prompt_versions.update(d.llm.prompt_versions_used)

        return PeriodMetrics(
            start_date=start_date or (daily[0].date if daily else ""),
            end_date=end_date or (daily[-1].date if daily else ""),
            total_requests=total_requests,
            total_successes=sum(d.success_count for d in daily),
            total_errors=total_errors,
            overall_error_rate=(
                total_errors / total_requests if total_requests > 0 else 0.0
            ),
            overall_latency=LatencyStats.from_values(all_latencies),
            overall_llm=LLMMetrics(
                total_calls=total_llm_calls,
                total_successes=total_llm_successes,
                total_fallbacks=sum(d.llm.total_fallbacks for d in daily),
                success_rate=(
                    total_llm_successes / total_llm_calls
                    if total_llm_calls > 0
                    else 0.0
                ),
                estimated_tokens=total_llm_tokens,
                estimated_cost_usd=(total_llm_tokens / 1000.0) * self._cost_per_1k,
                model_names_used=sorted(all_model_names),
                prompt_versions_used=sorted(all_prompt_versions),
            ),
            overall_flow_counts=dict(flow_counter),
            overall_top_cards=dict(card_counter),
            overall_personalization_rate=(
                sum(d.personalization_rate * d.request_count for d in daily)
                / total_requests
                if total_requests > 0
                else 0.0
            ),
            daily_breakdown=daily,
        )
