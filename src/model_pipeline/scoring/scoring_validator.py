"""
Scoring engine validation runner for RewardSense.

Runs golden test cases, computes accuracy metrics, benchmarks throughput,
and logs everything to MLflow under the 'reward-scoring' experiment.

Usage:
    python -m src.model_pipeline.scoring.scoring_validator
"""

import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from src.model_pipeline.scoring.reward_calculator import RewardCalculator
from src.model_pipeline.scoring.transaction_scorer import TransactionScorer
from src.model_pipeline.scoring.card_ranker import CardRanker

logger = logging.getLogger(__name__)

# ── Golden Test Dataset ──────────────────────────────────────────────
# Duplicated from test file so validator can run independently.
# Format: (test_id, card, transaction, expected_reward)

GOLDEN_CASES: List[Tuple[str, Dict, Dict, float]] = [
    # Base rate cards
    (
        "base_1pct_100",
        {
            "card_id": "g01",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
        },
        1.0,
    ),
    (
        "base_1.5pct_80",
        {
            "card_id": "g02",
            "reward_rates": {"universal_base_rate": 1.5},
            "annual_fee": 0,
        },
        {"amount": 80.0, "category": "gas", "merchant": "Shell", "mcc_code": 5541},
        1.2,
    ),
    (
        "base_2pct_250",
        {
            "card_id": "g03",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {
            "amount": 250.0,
            "category": "groceries",
            "merchant": "Whole Foods",
            "mcc_code": 5411,
        },
        5.0,
    ),
    # Category bonuses
    (
        "cat_3x_dining",
        {
            "card_id": "g08",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "category": "dining", "merchant": "Nobu", "mcc_code": 5812},
        3.0,
    ),
    (
        "cat_4x_groceries",
        {
            "card_id": "g10",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 4.0, "groceries": 4.0},
            },
            "annual_fee": 250,
        },
        {
            "amount": 200.0,
            "category": "groceries",
            "merchant": "Trader Joes",
            "mcc_code": 5411,
        },
        8.0,
    ),
    (
        "cat_5x_travel",
        {
            "card_id": "g13",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"travel": 5.0},
            },
            "annual_fee": 0,
        },
        {
            "amount": 1000.0,
            "category": "travel",
            "merchant": "Marriott",
            "mcc_code": 7011,
        },
        50.0,
    ),
    (
        "cat_fallback_base",
        {
            "card_id": "g12",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "category": "gas", "merchant": "BP", "mcc_code": 5541},
        1.0,
    ),
    # Rotating bonuses
    (
        "rot_q3_active",
        {
            "card_id": "g21",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q3": {"categories": ["dining"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Chipotle",
            "mcc_code": 5812,
            "date": datetime(2025, 8, 20),
        },
        5.0,
    ),
    (
        "rot_q1_inactive",
        {
            "card_id": "g20",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q1": {"categories": ["gas"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 40.0,
            "category": "gas",
            "merchant": "Shell",
            "mcc_code": 5541,
            "date": datetime(2025, 5, 15),
        },
        0.4,
    ),
    # Foreign transactions
    (
        "ftf_net_negative",
        {
            "card_id": "g25",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 3.0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Foreign",
            "mcc_code": 5812,
            "is_foreign": True,
        },
        -1.0,
    ),
    (
        "ftf_no_fee",
        {
            "card_id": "g26",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 0.0,
        },
        {
            "amount": 200.0,
            "category": "travel",
            "merchant": "Hotel",
            "mcc_code": 7011,
            "is_foreign": True,
        },
        4.0,
    ),
    # Edge cases
    (
        "zero_amount",
        {
            "card_id": "g29",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"amount": 0.0, "category": "dining", "merchant": "Test", "mcc_code": 5812},
        0.0,
    ),
    (
        "missing_rates",
        {"card_id": "g34", "annual_fee": 0},
        {"amount": 100.0, "category": "dining", "merchant": "Test", "mcc_code": 5812},
        1.0,
    ),
    # Real-world cards
    (
        "csr_dining",
        {
            "card_id": "g38",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 3.0},
            },
            "annual_fee": 550,
        },
        {
            "amount": 50.0,
            "category": "dining",
            "merchant": "Restaurant",
            "mcc_code": 5812,
        },
        1.5,
    ),
    (
        "amex_gold_groceries",
        {
            "card_id": "g42",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 4.0, "groceries": 4.0},
            },
            "annual_fee": 250,
        },
        {
            "amount": 95.0,
            "category": "groceries",
            "merchant": "Whole Foods",
            "mcc_code": 5411,
        },
        3.8,
    ),
    (
        "venture_x_travel",
        {
            "card_id": "g43",
            "reward_rates": {
                "universal_base_rate": 2.0,
                "category_bonuses": {"travel": 5.0},
            },
            "annual_fee": 395,
        },
        {"amount": 300.0, "category": "travel", "merchant": "Hyatt", "mcc_code": 7011},
        15.0,
    ),
    (
        "double_cash_general",
        {
            "card_id": "g47",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"amount": 88.0, "category": "dining", "merchant": "Panera", "mcc_code": 5812},
        1.76,
    ),
    # Complex combos
    (
        "rotating_priority",
        {
            "card_id": "g49",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
                "rotating_bonuses": {"Q1": {"categories": ["dining"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Place",
            "mcc_code": 5812,
            "date": datetime(2025, 2, 14),
        },
        5.0,
    ),
    (
        "category_when_rot_inactive",
        {
            "card_id": "g50",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
                "rotating_bonuses": {"Q1": {"categories": ["dining"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Place",
            "mcc_code": 5812,
            "date": datetime(2025, 6, 14),
        },
        3.0,
    ),
    (
        "ftf_with_bonus",
        {
            "card_id": "g51",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"travel": 5.0},
            },
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 3.0,
        },
        {
            "amount": 200.0,
            "category": "travel",
            "merchant": "Tokyo Hotel",
            "mcc_code": 7011,
            "is_foreign": True,
        },
        4.0,
    ),
]

TOLERANCE = 1e-4


class ScoringValidator:
    """
    Validates the scoring engine against golden test cases and benchmarks.
    Logs all results to MLflow.
    """

    def __init__(self):
        self.calculator = RewardCalculator()
        self.scorer = TransactionScorer()
        self.ranker = CardRanker()

    def run_golden_tests(self) -> Dict[str, Any]:
        """
        Run all golden test cases and compute accuracy metrics.

        Returns:
            Dict with total, passed, failed, accuracy, and per-case details.
        """
        results = []
        passed = 0
        failed = 0

        for test_id, card, txn, expected in GOLDEN_CASES:
            actual = self.calculator.calculate_reward(card, txn)
            is_pass = abs(actual - expected) < TOLERANCE

            if is_pass:
                passed += 1
            else:
                failed += 1
                logger.warning(
                    f"FAIL [{test_id}]: expected={expected:.4f}, actual={actual:.4f}"
                )

            results.append(
                {
                    "test_id": test_id,
                    "expected": expected,
                    "actual": round(actual, 6),
                    "passed": is_pass,
                }
            )

        total = len(GOLDEN_CASES)
        accuracy = passed / total if total > 0 else 0.0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "accuracy": accuracy,
            "details": results,
        }

    def run_throughput_benchmark(
        self, n_transactions: int = 5000, n_cards: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark scoring throughput.

        Args:
            n_transactions: Number of transactions to score
            n_cards: Number of cards in test portfolio

        Returns:
            Dict with single_card_throughput, batch_throughput, latency_per_txn.
        """
        categories = ["dining", "travel", "gas", "groceries", "utilities"]

        card = {
            "card_id": "bench_card",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 5.0},
            },
            "annual_fee": 0,
        }

        transactions = [
            {
                "amount": 50.0 + i,
                "category": categories[i % len(categories)],
                "merchant": f"M_{i}",
                "mcc_code": 5812,
            }
            for i in range(n_transactions)
        ]

        # Single card throughput
        start = time.time()
        for txn in transactions:
            self.calculator.calculate_reward(card, txn)
        single_elapsed = time.time() - start
        single_throughput = n_transactions / single_elapsed

        # Batch throughput
        portfolio = [
            {
                "card_id": f"card_{i}",
                "card_name": f"Card {i}",
                "reward_rates": {
                    "universal_base_rate": 1.0 + i * 0.5,
                    "category_bonuses": {"dining": 2.0 + i},
                },
                "annual_fee": i * 100,
            }
            for i in range(n_cards)
        ]

        start = time.time()
        self.scorer.score_batch(portfolio, transactions)
        batch_elapsed = time.time() - start
        batch_throughput = n_transactions / batch_elapsed

        return {
            "n_transactions": n_transactions,
            "n_cards": n_cards,
            "single_card_throughput": round(single_throughput, 1),
            "batch_throughput": round(batch_throughput, 1),
            "single_latency_ms": round((single_elapsed / n_transactions) * 1000, 4),
            "batch_latency_ms": round((batch_elapsed / n_transactions) * 1000, 4),
        }

    def validate_and_log(self, log_to_mlflow: bool = True) -> Dict[str, Any]:
        """
        Run full validation and optionally log to MLflow.

        Args:
            log_to_mlflow: If True, log results to MLflow reward-scoring experiment

        Returns:
            Combined validation report dict
        """
        logger.info("Starting scoring engine validation...")

        golden_results = self.run_golden_tests()
        benchmark_results = self.run_throughput_benchmark()

        report = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "golden_tests": golden_results,
            "benchmarks": benchmark_results,
        }

        logger.info(
            f"Golden tests: {golden_results['passed']}/{golden_results['total']} passed "
            f"({golden_results['accuracy']:.1%} accuracy)"
        )
        logger.info(
            f"Throughput: {benchmark_results['single_card_throughput']:.0f} txn/s (single), "
            f"{benchmark_results['batch_throughput']:.0f} txn/s (batch)"
        )

        if log_to_mlflow:
            self._log_to_mlflow(report)

        return report

    def _log_to_mlflow(self, report: Dict[str, Any]) -> None:
        """Log validation report to MLflow reward-scoring experiment."""
        try:
            import mlflow

            mlflow.set_experiment("reward-scoring")

            with mlflow.start_run(run_name="scoring-validation"):
                # Golden test metrics
                mlflow.log_metric("golden_accuracy", report["golden_tests"]["accuracy"])
                mlflow.log_metric("golden_passed", report["golden_tests"]["passed"])
                mlflow.log_metric("golden_failed", report["golden_tests"]["failed"])
                mlflow.log_metric("golden_total", report["golden_tests"]["total"])

                # Benchmark metrics
                mlflow.log_metric(
                    "throughput_single", report["benchmarks"]["single_card_throughput"]
                )
                mlflow.log_metric(
                    "throughput_batch", report["benchmarks"]["batch_throughput"]
                )
                mlflow.log_metric(
                    "latency_single_ms", report["benchmarks"]["single_latency_ms"]
                )
                mlflow.log_metric(
                    "latency_batch_ms", report["benchmarks"]["batch_latency_ms"]
                )

                # Params
                mlflow.log_param("n_golden_cases", report["golden_tests"]["total"])
                mlflow.log_param(
                    "n_bench_transactions", report["benchmarks"]["n_transactions"]
                )
                mlflow.log_param("n_bench_cards", report["benchmarks"]["n_cards"])

                # Full report as artifact
                report_json = json.dumps(report, indent=2, default=str)
                with open("/tmp/scoring_validation_report.json", "w") as f:
                    f.write(report_json)
                mlflow.log_artifact("/tmp/scoring_validation_report.json")

                # Failed cases as separate artifact if any
                failed_cases = [
                    d for d in report["golden_tests"]["details"] if not d["passed"]
                ]
                if failed_cases:
                    with open("/tmp/scoring_failed_cases.json", "w") as f:
                        json.dump(failed_cases, f, indent=2)
                    mlflow.log_artifact("/tmp/scoring_failed_cases.json")

            logger.info(
                "Validation results logged to MLflow 'reward-scoring' experiment"
            )

        except ImportError:
            logger.warning("MLflow not installed, skipping logging")
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")


# ── CLI entrypoint ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validator = ScoringValidator()
    report = validator.validate_and_log(log_to_mlflow=True)
    print(f"\nValidation complete: {report['golden_tests']['accuracy']:.1%} accuracy")
    print(f"Throughput: {report['benchmarks']['batch_throughput']:.0f} txn/sec (batch)")
