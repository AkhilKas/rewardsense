"""Run LLM explanation latency benchmark for Epic 4 Story 4.4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.model_pipeline.llm import ExplanationGenerator, ExplanationLatencyBenchmark
from src.model_pipeline.llm.prompt_builder import ExplanationType
from src.model_pipeline.llm.vertex_gemini_client import VertexGeminiClient
from src.model_pipeline.tracking import RewardSenseTracker


def sample_scoring_output() -> dict:
    return {
        "transaction": {"amount": 100.0, "category": "dining", "merchant": "Chipotle"},
        "best_card": {
            "card_id": "amex_gold",
            "card_name": "Amex Gold",
            "reward_rate": 4.0,
            "reward_amount": 4.0,
        },
        "alternatives": [
            {
                "card_id": "citi_double",
                "card_name": "Citi Double Cash",
                "reward_rate": 2.0,
                "reward_amount": 2.0,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--budget-ms", type=float, default=2000.0)
    parser.add_argument(
        "--model", type=str, default=os.getenv("LLM_MODEL", "gemini-2.5-flash")
    )
    parser.add_argument(
        "--location", type=str, default=os.getenv("VERTEX_LOCATION", "us-central1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/llm_latency_benchmark.json")
    )
    args = parser.parse_args()

    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise SystemExit("GCP_PROJECT_ID is required")

    client = VertexGeminiClient(
        project_id=project_id,
        location=args.location,
        model=args.model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        timeout_sec=float(os.getenv("LLM_TIMEOUT_SEC", "10")),
    )

    generator = ExplanationGenerator(llm_client=client, model_name=args.model)
    tracker = RewardSenseTracker(experiment="llm-explainability")

    bench = ExplanationLatencyBenchmark(
        generator=generator,
        latency_budget_ms=args.budget_ms,
        tracker=tracker,
    )
    result = bench.run(
        scoring_output=sample_scoring_output(),
        personalization_signals={"user_segment": "foodie"},
        n_requests=args.requests,
        explanation_type=ExplanationType.SINGLE_TRANSACTION,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
