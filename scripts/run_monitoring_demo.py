#!/usr/bin/env python3
"""
Monitoring Pipeline Local Demo.

Simulates the full monitoring loop without requiring a deployed API:
  1. Generates synthetic inference logs (mimics what inference_logger.py produces)
  2. Runs the data collector against those logs
  3. Creates a reference dataset and runs drift detection
  4. Computes performance metrics
  5. Evaluates thresholds and shows retrain decision
  6. Sends a dry-run Slack notification

Usage:
    # Normal run (no drift)
    PYTHONPATH=. python scripts/run_monitoring_demo.py

    # Simulate drift (shifts spending distributions)
    PYTHONPATH=. python scripts/run_monitoring_demo.py --simulate-drift

    # Simulate latency degradation
    PYTHONPATH=. python scripts/run_monitoring_demo.py --simulate-slow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitoring_demo")


def parse_args():
    p = argparse.ArgumentParser(description="RewardSense Monitoring Demo")
    p.add_argument("--n-logs", type=int, default=100,
                    help="Number of synthetic inference logs to generate")
    p.add_argument("--simulate-drift", action="store_true",
                    help="Shift spending distributions to simulate data drift")
    p.add_argument("--simulate-slow", action="store_true",
                    help="Inflate latency values to trigger performance alert")
    p.add_argument("--log-dir", default=None,
                    help="Directory for synthetic logs (default: tmp)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =====================================================================
# Step 1: Generate synthetic inference logs
# =====================================================================

def generate_synthetic_logs(
    log_dir: Path,
    n: int = 100,
    seed: int = 42,
    drift: bool = False,
    slow: bool = False,
) -> Path:
    """Write synthetic inference log JSON files to a date-partitioned dir."""
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    cards = [
        "Chase Sapphire Preferred",
        "Amex Gold Card",
        "Capital One Venture X",
        "Citi Double Cash",
        "Blue Cash Preferred",
    ]

    # Normal spending distributions
    spend_means = {"dining": 500, "travel": 800, "groceries": 400, "gas": 200, "online_shopping": 300}

    # Drift: shift spending patterns significantly
    if drift:
        spend_means = {"dining": 1500, "travel": 200, "groceries": 1200, "gas": 50, "online_shopping": 900}
        logger.info("DRIFT MODE: spending distributions shifted")

    # Slow: inflate latency
    latency_base = 200 if not slow else 12000
    if slow:
        logger.info("SLOW MODE: latency inflated to ~%.0fms", latency_base)

    records_written = 0
    for i in range(n):
        # Spread logs across last 3 days
        days_ago = rng.integers(0, 3)
        log_time = now - timedelta(days=int(days_ago), hours=int(rng.integers(0, 24)))
        prefix = f"{log_time.year:04d}/{log_time.month:02d}/{log_time.day:02d}"
        day_dir = log_dir / prefix
        day_dir.mkdir(parents=True, exist_ok=True)

        request_id = f"demo-{i:04d}"
        spending = {
            cat: max(0, float(rng.normal(mean, mean * 0.3)))
            for cat, mean in spend_means.items()
        }

        # Scores
        scores = []
        for rank, card in enumerate(rng.choice(cards, size=min(3, len(cards)), replace=False)):
            scores.append({
                "card_name": str(card),
                "rank": rank + 1,
                "deterministic_score": round(float(rng.normal(40, 15)), 4),
                "personalization_score": round(float(rng.normal(35, 12)), 4),
                "blended_score": round(float(rng.normal(38, 13)), 4),
            })

        total_ms = max(50.0, float(rng.normal(latency_base, latency_base * 0.2)))

        record = {
            "timestamp": log_time.isoformat(),
            "request_id": request_id,
            "user_hash": f"user_{rng.integers(1000, 9999)}",
            "input_features": {
                "spending_categories": spending,
                "monthly_spend": round(sum(spending.values()), 2),
                "preferred_rewards": list(rng.choice(
                    ["travel_points", "cashback", "hotel_points"], size=int(rng.integers(1, 3)), replace=False
                )),
                "transaction_history_count": int(rng.integers(5, 50)),
            },
            "predicted_scores": scores,
            "top_card": scores[0]["card_name"],
            "model_version": "3",
            "latency_breakdown_ms": {
                "normalize": round(total_ms * 0.02, 3),
                "deterministic": round(total_ms * 0.10, 3),
                "personalization": round(total_ms * 0.60, 3),
                "rank": round(total_ms * 0.03, 3),
                "total": round(total_ms, 3),
            },
            "is_personalized": bool(rng.random() > 0.15),
        }

        filepath = day_dir / f"{request_id}.json"
        filepath.write_text(json.dumps(record, indent=2), encoding="utf-8")
        records_written += 1

    logger.info("Generated %d synthetic inference logs in %s", records_written, log_dir)
    return log_dir


# =====================================================================
# Step 2: Generate reference dataset (simulates training data)
# =====================================================================

def generate_reference_dataset(output_path: Path, seed: int = 42) -> Path:
    """Create a reference CSV matching the input_features schema."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = 200

    ref = pd.DataFrame({
        "monthly_spend": rng.normal(2200, 800, n).clip(100),
        "transaction_history_count": rng.integers(5, 50, n),
        "spend_dining": rng.normal(500, 200, n).clip(0),
        "spend_travel": rng.normal(800, 300, n).clip(0),
        "spend_groceries": rng.normal(400, 150, n).clip(0),
        "spend_gas": rng.normal(200, 100, n).clip(0),
        "spend_online_shopping": rng.normal(300, 120, n).clip(0),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(output_path, index=False)
    logger.info("Generated reference dataset: %s (%d rows)", output_path, len(ref))
    return output_path


# =====================================================================
# Main demo
# =====================================================================

def run_demo(args):
    start = time.time()

    # Setup directories
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = PROJECT_ROOT / "data" / "demo_inference_logs"

    ref_path = PROJECT_ROOT / "data" / "reference" / "training_reference.csv"
    report_dir = PROJECT_ROOT / "data" / "monitoring" / "drift-reports"
    perf_dir = PROJECT_ROOT / "data" / "monitoring" / "performance"

    # --- Step 1: Generate synthetic inference logs ---
    print(f"\n{'='*60}")
    print("STEP 1: Generating synthetic inference logs")
    print(f"{'='*60}")
    generate_synthetic_logs(
        log_dir, n=args.n_logs, seed=args.seed,
        drift=args.simulate_drift, slow=args.simulate_slow,
    )

    # --- Step 2: Generate reference dataset ---
    print(f"\n{'='*60}")
    print("STEP 2: Creating reference (training) dataset")
    print(f"{'='*60}")
    generate_reference_dataset(ref_path, seed=args.seed)

    # --- Step 3: Run data collector ---
    print(f"\n{'='*60}")
    print("STEP 3: Collecting inference data")
    print(f"{'='*60}")
    from src.monitoring.data_collector import InferenceDataCollector

    collector = InferenceDataCollector(
        bucket="unused", local_dir=str(log_dir)
    )
    collector._gcs_client = None  # force local mode
    summary = collector.collect(days=7)

    print(f"  Records collected: {summary.total_records}")
    print(f"  Model versions: {summary.model_versions}")
    stats = summary.summary_stats
    if "latency" in stats:
        print(f"  Latency p50: {stats['latency']['p50_ms']:.0f}ms")
        print(f"  Latency p95: {stats['latency']['p95_ms']:.0f}ms")
    if "top_card_frequency" in stats:
        print(f"  Top cards: {stats['top_card_frequency']}")
    if "personalization_rate" in stats:
        print(f"  Personalization rate: {stats['personalization_rate']:.1%}")

    # --- Step 4: Run drift detection ---
    print(f"\n{'='*60}")
    print("STEP 4: Running drift detection (Evidently AI)")
    print(f"{'='*60}")
    from src.monitoring.drift_detector import DriftDetector, EVIDENTLY_AVAILABLE

    if EVIDENTLY_AVAILABLE:
        detector = DriftDetector(
            reference_path=str(ref_path),
            feature_drift_threshold=0.3,
            output_dir=str(report_dir),
        )
        drift_result = detector.detect(summary.input_features_df)

        print(f"  Drift detected: {drift_result.drift_detected}")
        print(f"  Dataset drift share: {drift_result.dataset_drift_share:.1%}")
        print(f"  Drifted features: {drift_result.drifted_features}")
        if drift_result.html_report_path:
            print(f"  HTML report: {drift_result.html_report_path}")
    else:
        print("  [SKIPPED] Evidently not installed. pip install evidently")
        # Create a mock result for downstream steps
        from src.monitoring.drift_detector import DriftResult
        drift_result = DriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            drift_detected=args.simulate_drift,
        )
        if args.simulate_drift:
            drift_result.drifted_features = ["spend_dining", "spend_travel"]
            drift_result.dataset_drift_share = 0.75

    # --- Step 5: Compute performance metrics ---
    print(f"\n{'='*60}")
    print("STEP 5: Computing performance metrics")
    print(f"{'='*60}")
    from src.monitoring.performance_tracker import PerformanceTracker

    perf_tracker = PerformanceTracker(
        output_dir=str(perf_dir),
        latency_threshold_ms=10000,
    )
    snapshot = perf_tracker.compute(summary)
    perf_tracker.save_snapshot(snapshot)

    print(f"  Total requests: {snapshot.total_requests}")
    print(f"  Latency p50: {snapshot.latency_p50_ms:.0f}ms")
    print(f"  Latency p95: {snapshot.latency_p95_ms:.0f}ms")
    print(f"  Latency p99: {snapshot.latency_p99_ms:.0f}ms")
    print(f"  Score mean: {snapshot.score_mean:.2f}")
    print(f"  Top card entropy: {snapshot.top_card_entropy:.2f}")
    print(f"  Personalization rate: {snapshot.personalization_rate:.1%}")
    if snapshot.alerts:
        print(f"  ALERTS: {snapshot.alerts}")
    else:
        print(f"  Alerts: None")

    # --- Step 6: Evaluate thresholds ---
    print(f"\n{'='*60}")
    print("STEP 6: Evaluating thresholds (retrain decision)")
    print(f"{'='*60}")

    should_retrain = False
    reasons = []

    if drift_result.drift_detected:
        reasons.append(f"data_drift (share={drift_result.dataset_drift_share:.1%})")
    if snapshot.has_alerts:
        for alert in snapshot.alerts:
            reasons.append(f"performance: {alert}")

    should_retrain = len(reasons) > 0

    if should_retrain:
        print(f"  RETRAIN RECOMMENDED")
        for r in reasons:
            print(f"    - {r}")
    else:
        print(f"  No retraining needed - all metrics healthy")

    # --- Step 7: Send notification (dry run) ---
    print(f"\n{'='*60}")
    print("STEP 7: Sending Slack notification (dry run)")
    print(f"{'='*60}")
    from src.monitoring.notifier import SlackNotifier

    notifier = SlackNotifier(dry_run=True)
    notifier.send_monitoring_summary(drift_result, snapshot)

    if should_retrain:
        notifier.send_retrain_trigger(
            reason="; ".join(reasons),
            drift_report_path=drift_result.html_report_path,
        )

    # --- Summary ---
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"MONITORING DEMO COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"\n  Inference logs:     {log_dir}")
    print(f"  Reference data:     {ref_path}")
    print(f"  Performance snap:   {perf_dir}")
    if drift_result.html_report_path:
        print(f"  Drift report:       file://{Path(drift_result.html_report_path).resolve()}")
    print(f"  Drift detected:     {drift_result.drift_detected}")
    print(f"  Retrain needed:     {should_retrain}")
    print(f"  Total time:         {elapsed:.1f}s")

    return {
        "total_records": summary.total_records,
        "drift_detected": drift_result.drift_detected,
        "retrain_needed": should_retrain,
        "reasons": reasons,
        "latency_p95": snapshot.latency_p95_ms,
        "alerts": snapshot.alerts,
    }


def main():
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()