"""
Unit tests for — MLflow Experiment Tracking Wrapper.

Tests:
  - Experiment creation and namespace management
  - Run lifecycle (start, log, end)
  - Parameter, metric, and artifact logging
  - Query helpers (best run, compare runs)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def mock_mlflow():
    """Patch mlflow and MlflowClient for isolated testing."""
    with patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.tracking": MagicMock()}):
        import importlib
        import src.model_pipeline.tracking as tracking_mod

        tracking_mod.MLFLOW_AVAILABLE = True
        tracking_mod.mlflow = MagicMock()
        tracking_mod.MlflowClient = MagicMock

        # Mock experiment lookup
        tracking_mod.mlflow.get_experiment_by_name.return_value = None
        tracking_mod.mlflow.create_experiment.return_value = "1"

        yield tracking_mod


@pytest.fixture
def tracker(mock_mlflow):
    """Create a RewardSenseTracker with mocked MLflow."""
    return mock_mlflow.RewardSenseTracker(
        experiment="personalization-model",
        tracking_uri="http://localhost:5000",
    )


# =====================================================================
# Experiment Management
# =====================================================================


class TestExperimentManagement:
    """Test experiment creation and namespace handling."""

    def test_creates_experiment_on_init(self, mock_mlflow, tracker):
        """Tracker should create experiment if it doesn't exist."""
        mock_mlflow.mlflow.create_experiment.assert_called_once()

    def test_reuses_existing_experiment(self, mock_mlflow):
        """If experiment exists, don't create a new one."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "42"
        mock_mlflow.mlflow.get_experiment_by_name.return_value = mock_exp

        t = mock_mlflow.RewardSenseTracker(experiment="personalization-model")
        # create_experiment should NOT be called again for existing exp
        # (it was called once during module-level fixture setup, so check
        # that the reuse path returns the existing ID)
        result = t._ensure_experiment_exists("personalization-model")
        assert result == "42"

    def test_create_all_namespaces(self, mock_mlflow, tracker):
        """create_all_namespaces should create all 3 predefined experiments."""
        mock_mlflow.mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.mlflow.create_experiment.return_value = "1"

        results = tracker.create_all_namespaces()
        assert len(results) == 3
        assert "reward-scoring" in results
        assert "personalization-model" in results
        assert "llm-explainability" in results

    def test_list_experiments(self, mock_mlflow, tracker):
        """list_experiments should return formatted experiment list."""
        mock_exp = MagicMock()
        mock_exp.name = "test-exp"
        mock_exp.experiment_id = "1"
        mock_exp.lifecycle_stage = "active"
        tracker._client = MagicMock()
        tracker._client.search_experiments.return_value = [mock_exp]

        exps = tracker.list_experiments()
        assert len(exps) == 1
        assert exps[0]["name"] == "test-exp"


# =====================================================================
# Run Lifecycle
# =====================================================================


class TestRunLifecycle:
    """Test run start/end context manager."""

    def test_start_run_context_manager(self, mock_mlflow, tracker):
        """Run should start and end cleanly via context manager."""
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_mlflow.mlflow.start_run.return_value = mock_run

        with tracker.start_run(run_name="test-run") as run:
            assert run is not None
            assert tracker._active_run is not None

        mock_mlflow.mlflow.end_run.assert_called_once()
        assert tracker._active_run is None

    def test_start_run_sets_experiment(self, mock_mlflow, tracker):
        """start_run should set the correct experiment."""
        mock_mlflow.mlflow.start_run.return_value = MagicMock()

        with tracker.start_run(run_name="test"):
            mock_mlflow.mlflow.set_experiment.assert_called_with(
                "personalization-model"
            )

    def test_active_run_id_returns_none_outside_run(self, tracker):
        """active_run_id should be None when no run is active."""
        assert tracker.active_run_id is None

    def test_run_ends_on_exception(self, mock_mlflow, tracker):
        """Run should still end if an exception occurs inside context."""
        mock_mlflow.mlflow.start_run.return_value = MagicMock()

        with pytest.raises(ValueError):
            with tracker.start_run(run_name="failing-run"):
                raise ValueError("intentional")

        mock_mlflow.mlflow.end_run.assert_called_once()


# =====================================================================
# Logging Helpers
# =====================================================================


class TestLogging:
    """Test parameter, metric, and artifact logging."""

    def test_log_params(self, mock_mlflow, tracker):
        """log_params should sanitize and forward to mlflow."""
        tracker.log_params({"lr": 0.01, "max_depth": 6})
        mock_mlflow.mlflow.log_params.assert_called_once()
        call_args = mock_mlflow.mlflow.log_params.call_args[0][0]
        assert call_args["lr"] == "0.01"
        assert call_args["max_depth"] == "6"

    def test_log_params_truncates_long_values(self, mock_mlflow, tracker):
        """Values longer than 500 chars should be truncated."""
        long_val = "x" * 1000
        tracker.log_params({"long": long_val})
        call_args = mock_mlflow.mlflow.log_params.call_args[0][0]
        assert len(call_args["long"]) == 500

    def test_log_metrics(self, mock_mlflow, tracker):
        """log_metrics should forward dict to mlflow."""
        tracker.log_metrics({"ndcg_5": 0.82, "map_5": 0.78})
        mock_mlflow.mlflow.log_metrics.assert_called_once_with(
            {"ndcg_5": 0.82, "map_5": 0.78}, step=None
        )

    def test_log_metric_single(self, mock_mlflow, tracker):
        """log_metric should log a single metric."""
        tracker.log_metric("accuracy", 0.95, step=10)
        mock_mlflow.mlflow.log_metric.assert_called_once_with("accuracy", 0.95, step=10)

    def test_log_artifact(self, mock_mlflow, tracker, tmp_path):
        """log_artifact should forward file path to mlflow."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        tracker.log_artifact(f, artifact_path="outputs")
        mock_mlflow.mlflow.log_artifact.assert_called_once_with(
            str(f), artifact_path="outputs"
        )

    def test_log_dict(self, mock_mlflow, tracker):
        """log_dict should serialize to JSON and log as artifact."""
        tracker.log_dict({"key": "value"}, "config.json")
        mock_mlflow.mlflow.log_artifact.assert_called_once()

    def test_set_tags(self, mock_mlflow, tracker):
        """set_tags should forward to mlflow."""
        tracker.set_tags({"model_type": "xgboost"})
        mock_mlflow.mlflow.set_tags.assert_called_once_with({"model_type": "xgboost"})


# =====================================================================
# Query Helpers
# =====================================================================


class TestQueryHelpers:
    """Test best run and comparison queries."""

    def test_get_best_run(self, mock_mlflow, tracker):
        """get_best_run should return formatted best run."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_mlflow.mlflow.get_experiment_by_name.return_value = mock_exp

        mock_run = MagicMock()
        mock_run.info.run_id = "best123"
        mock_run.info.status = "FINISHED"
        mock_run.data.params = {"lr": "0.01"}
        mock_run.data.metrics = {"ndcg_5": 0.85}
        mock_run.data.tags = {"mlflow.runName": "best-run"}

        tracker._client = MagicMock()
        tracker._client.search_runs.return_value = [mock_run]

        result = tracker.get_best_run(metric="ndcg_5")
        assert result is not None
        assert result["run_id"] == "best123"
        assert result["metrics"]["ndcg_5"] == 0.85

    def test_get_best_run_no_experiment(self, mock_mlflow, tracker):
        """get_best_run returns None if experiment doesn't exist."""
        mock_mlflow.mlflow.get_experiment_by_name.return_value = None
        assert tracker.get_best_run() is None

    def test_compare_runs(self, mock_mlflow, tracker):
        """compare_runs should return list of formatted runs."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_mlflow.mlflow.get_experiment_by_name.return_value = mock_exp

        mock_runs = []
        for i in range(3):
            r = MagicMock()
            r.info.run_id = f"run_{i}"
            r.data.tags = {"mlflow.runName": f"run-{i}"}
            r.data.params = {"lr": str(0.01 * (i + 1))}
            r.data.metrics = {"ndcg_5": 0.8 + i * 0.02}
            mock_runs.append(r)

        tracker._client = MagicMock()
        tracker._client.search_runs.return_value = mock_runs

        results = tracker.compare_runs(metric="ndcg_5", max_results=3)
        assert len(results) == 3
        assert results[0]["run_id"] == "run_0"


# =====================================================================
# Graceful Degradation
# =====================================================================


class TestGracefulDegradation:
    """Test behavior when MLflow is not installed."""

    def test_no_mlflow_no_crash(self, mock_mlflow):
        """Tracker should not crash when MLflow is unavailable."""
        mock_mlflow.MLFLOW_AVAILABLE = False
        t = mock_mlflow.RewardSenseTracker(experiment="test")
        # All methods should be no-ops
        t.log_params({"x": 1})
        t.log_metrics({"y": 2.0})
        t.log_metric("z", 3.0)
        assert t.list_experiments() == []
        assert t.get_best_run() is None
        assert t.compare_runs() == []

    def test_no_mlflow_run_context(self, mock_mlflow):
        """start_run should yield None when MLflow unavailable."""
        mock_mlflow.MLFLOW_AVAILABLE = False
        t = mock_mlflow.RewardSenseTracker(experiment="test")
        with t.start_run(run_name="ghost") as run:
            assert run is None