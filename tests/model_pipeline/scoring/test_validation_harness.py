"""
Tests for the validation harness (ScoringValidator) - Story 2.3

Validates that the validation runner correctly executes golden tests,
computes metrics, benchmarks throughput, and logs to MLflow.

File: tests/model_pipeline/scoring/test_validation_harness.py
"""

from unittest.mock import patch, MagicMock


class TestScoringValidatorGolden:
    """Test the golden test runner."""

    def test_run_golden_tests_returns_results(self):
        from src.model_pipeline.scoring.scoring_validator import ScoringValidator

        validator = ScoringValidator()
        results = validator.run_golden_tests()

        assert "total" in results
        assert "passed" in results
        assert "failed" in results
        assert "accuracy" in results
        assert "details" in results
        assert results["total"] > 0

    def test_golden_tests_100_accuracy(self):
        """All golden cases pass with current implementation."""
        from src.model_pipeline.scoring.scoring_validator import ScoringValidator

        validator = ScoringValidator()
        results = validator.run_golden_tests()

        assert results["accuracy"] == 1.0, (
            f"{results['failed']} golden tests failed: "
            f"{[d['test_id'] for d in results['details'] if not d['passed']]}"
        )

    def test_golden_details_have_expected_fields(self):
        from src.model_pipeline.scoring.scoring_validator import ScoringValidator

        validator = ScoringValidator()
        results = validator.run_golden_tests()

        for detail in results["details"]:
            assert "test_id" in detail
            assert "expected" in detail
            assert "actual" in detail
            assert "passed" in detail


class TestScoringValidatorBenchmark:
    """Test the throughput benchmark."""

    def test_benchmark_returns_metrics(self):
        from src.model_pipeline.scoring.scoring_validator import ScoringValidator

        validator = ScoringValidator()
        bench = validator.run_throughput_benchmark(n_transactions=100, n_cards=2)

        assert "single_card_throughput" in bench
        assert "batch_throughput" in bench
        assert "single_latency_ms" in bench
        assert "batch_latency_ms" in bench
        assert bench["single_card_throughput"] > 0
        assert bench["batch_throughput"] > 0

    def test_benchmark_meets_threshold(self):
        """Throughput exceeds 1000 txn/sec requirement."""
        from src.model_pipeline.scoring.scoring_validator import ScoringValidator

        validator = ScoringValidator()
        bench = validator.run_throughput_benchmark(n_transactions=2000, n_cards=5)

        assert bench["single_card_throughput"] > 1000
        assert bench["batch_throughput"] > 1000


class TestScoringValidatorMLflow:
    """Test MLflow logging (mocked)."""

    def test_validate_and_log_calls_mlflow(self):
        """Verify MLflow is called with correct experiment and metrics."""
        import sys

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from src.model_pipeline.scoring.scoring_validator import ScoringValidator

            validator = ScoringValidator()
            report = validator.validate_and_log(log_to_mlflow=True)

            mock_mlflow.set_experiment.assert_called_with("reward-scoring")
            mock_mlflow.start_run.assert_called_once()
            assert mock_mlflow.log_metric.call_count >= 6
            assert report["golden_tests"]["accuracy"] == 1.0

    def test_validate_without_mlflow(self):
        """validate_and_log works even if MLflow not installed."""
        from src.model_pipeline.scoring.scoring_validator import ScoringValidator

        validator = ScoringValidator()
        # Should not raise even if mlflow import fails
        report = validator.validate_and_log(log_to_mlflow=False)

        assert report["golden_tests"]["total"] > 0
        assert report["benchmarks"]["single_card_throughput"] > 0
