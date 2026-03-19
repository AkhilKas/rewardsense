"""Latency benchmarking utilities for Story 4.4."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Any, Dict, List, Optional

from src.model_pipeline.llm.explanation_generator import ExplanationGenerator
from src.model_pipeline.llm.prompt_builder import ExplanationType
from src.model_pipeline.tracking import RewardSenseTracker


@dataclass(frozen=True)
class LatencyBenchmarkResult:
    """Aggregate latency metrics for explanation generation."""

    n_requests: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    latency_budget_ms: float
    passed: bool


class ExplanationLatencyBenchmark:
    """Run repeatable latency checks for the explainability layer."""

    def __init__(
        self,
        generator: ExplanationGenerator,
        latency_budget_ms: float = 2000.0,
        tracker: Optional[RewardSenseTracker] = None,
    ) -> None:
        self.generator = generator
        self.latency_budget_ms = latency_budget_ms
        self.tracker = tracker

    def run(
        self,
        scoring_output: Dict[str, Any],
        personalization_signals: Dict[str, Any],
        n_requests: int = 20,
        explanation_type: ExplanationType = ExplanationType.SINGLE_TRANSACTION,
    ) -> LatencyBenchmarkResult:
        """Benchmark end-to-end explanation generation latency."""
        latencies: List[float] = []

        for _ in range(n_requests):
            start = time.perf_counter()
            self.generator.generate(
                explanation_type=explanation_type,
                scoring_output=scoring_output,
                personalization_signals=personalization_signals,
            )
            latencies.append((time.perf_counter() - start) * 1000)

        ordered = sorted(latencies)
        p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))

        result = LatencyBenchmarkResult(
            n_requests=n_requests,
            mean_latency_ms=round(statistics.mean(ordered), 3),
            p50_latency_ms=round(statistics.median(ordered), 3),
            p95_latency_ms=round(ordered[p95_index], 3),
            max_latency_ms=round(max(ordered), 3),
            latency_budget_ms=self.latency_budget_ms,
            passed=ordered[p95_index] <= self.latency_budget_ms,
        )

        self._log_result(result, explanation_type.value)
        return result

    def _log_result(
        self, result: LatencyBenchmarkResult, explanation_type: str
    ) -> None:
        if self.tracker is None:
            return

        with self.tracker.start_run(run_name=f"llm-latency-{explanation_type}"):
            self.tracker.log_params(
                {
                    "benchmark_type": "llm_explanation_latency",
                    "n_requests": result.n_requests,
                    "latency_budget_ms": result.latency_budget_ms,
                    "explanation_type": explanation_type,
                }
            )
            self.tracker.log_metrics(
                {
                    "latency_mean_ms": result.mean_latency_ms,
                    "latency_p50_ms": result.p50_latency_ms,
                    "latency_p95_ms": result.p95_latency_ms,
                    "latency_max_ms": result.max_latency_ms,
                    "latency_budget_passed": 1.0 if result.passed else 0.0,
                }
            )
            self.tracker.log_dict(
                result.__dict__, f"llm_latency_{explanation_type}.json"
            )
