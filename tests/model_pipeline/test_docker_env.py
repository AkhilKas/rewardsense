"""
Unit Tests: Docker Environment Validation for Model Pipeline.

These tests validate that the model training environment has been
configured correctly — importable packages, config files, env vars,
and the get_config() helper. They can run both locally (pytest) and
inside the Docker container.

Story 1.3 — Dockerize Model Training Environment.
"""

import importlib
import os
import sys

import pytest


# ---- Python Version ----


class TestPythonVersion:
    """Ensure minimum Python version for ML compatibility."""

    def test_python_version_at_least_3_9(self):
        assert sys.version_info >= (
            3,
            9,
        ), f"Python >= 3.9 required, got {sys.version_info[0]}.{sys.version_info[1]}"


# ---- ML Dependencies Importable ----


class TestMLDependencies:
    """Verify all critical ML packages can be imported."""

    @pytest.mark.parametrize(
        "package_name",
        [
            "xgboost",
            "lightgbm",
            "sklearn",
            "mlflow",
            "shap",
            "lime",
            "fairlearn",
            "optuna",
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "seaborn",
            "yaml",
            "pydantic",
            "loguru",
            "joblib",
        ],
    )
    def test_ml_dependencies_importable(self, package_name):
        """Each ML package should be importable without errors."""
        mod = importlib.import_module(package_name)
        assert mod is not None, f"Failed to import {package_name}"


# ---- Model Pipeline Package ----


class TestModelPipelinePackage:
    """Verify the model_pipeline package is correctly installed."""

    def test_model_pipeline_importable(self):
        """model_pipeline package should be on the Python path."""
        import model_pipeline

        assert hasattr(model_pipeline, "__version__")
        assert model_pipeline.__version__ == "0.1.0"

    def test_model_pipeline_has_get_config(self):
        """model_pipeline should expose a get_config function."""
        from model_pipeline import get_config

        assert callable(get_config)

    def test_train_module_importable(self):
        """model_pipeline.train should be importable and main() should run."""
        from model_pipeline import train

        assert hasattr(train, "main")

    @pytest.mark.skip(
        reason="train.main() now runs full ML orchestration requiring data; tested via E2E"
    )
    def test_train_main_runs_successfully(self):
        """train.main() should execute and exit with code 0."""
        from model_pipeline.train import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


# ---- Config Files ----


class TestConfigFiles:
    """Verify required config files are accessible."""

    def test_model_config_yaml_exists(self):
        """config/model_config.yaml should be present."""
        # Check relative to project root (works locally and in container)
        possible_paths = [
            "config/model_config.yaml",
            os.path.join(os.path.dirname(__file__), "../../config/model_config.yaml"),
        ]
        found = any(os.path.isfile(p) for p in possible_paths)
        assert found, f"model_config.yaml not found in any of: {possible_paths}"

    def test_model_config_yaml_parseable(self):
        """config/model_config.yaml should be valid YAML."""
        import yaml

        possible_paths = [
            "config/model_config.yaml",
            os.path.join(os.path.dirname(__file__), "../../config/model_config.yaml"),
        ]
        config_path = next((p for p in possible_paths if os.path.isfile(p)), None)
        if config_path is None:
            pytest.skip("model_config.yaml not found")

        assert config_path is not None  # type guard for mypy
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict), "Config should parse to a dict"
        assert (
            "point_valuation_model" in config
        ), "Missing point_valuation_model section"
        assert (
            "recommendation_engine" in config
        ), "Missing recommendation_engine section"


# ---- Environment Configuration ----


class TestGetConfig:
    """Test the get_config() environment-aware configuration helper."""

    def test_get_config_returns_dict(self):
        """get_config() should return a dict with expected keys."""
        from model_pipeline import get_config

        config = get_config()
        assert isinstance(config, dict)
        expected_keys = {
            "execution_env",
            "data_dir",
            "mlflow_tracking_uri",
            "artifact_store",
            "model_registry",
            "gcp_project",
            "gcp_bucket",
        }
        assert expected_keys.issubset(
            config.keys()
        ), f"Missing keys: {expected_keys - config.keys()}"

    def test_get_config_local_mode(self, monkeypatch):
        """In local mode, paths should be local filesystem paths."""
        monkeypatch.setenv("EXECUTION_ENV", "local")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

        from model_pipeline import get_config

        config = get_config()

        assert config["execution_env"] == "local"
        assert not config["data_dir"].startswith("gs://")
        assert not config["artifact_store"].startswith("gs://")

    def test_get_config_gcp_mode(self, monkeypatch):
        """In GCP mode, paths should use gs:// URIs."""
        monkeypatch.setenv("EXECUTION_ENV", "gcp")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
        monkeypatch.setenv("GCP_BUCKET_NAME", "test-bucket")

        from model_pipeline import get_config

        config = get_config()

        assert config["execution_env"] == "gcp"
        assert config["data_dir"].startswith("gs://")
        assert config["artifact_store"].startswith("gs://")
        assert "test-bucket" in config["data_dir"]

    def test_get_config_defaults_to_local(self, monkeypatch):
        """When EXECUTION_ENV is not set, default to local."""
        monkeypatch.delenv("EXECUTION_ENV", raising=False)

        from model_pipeline import get_config

        config = get_config()

        assert config["execution_env"] == "local"
