"""
Unit tests for src/serving/model_loader.py (Story 1.3).

All MLflow and PersonalizedScorer calls are mocked so tests run
without a live MLflow server or trained model.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_version(version: str = "3", run_id: str = "abc123") -> SimpleNamespace:
    """Return a minimal mock ModelVersion object."""
    return SimpleNamespace(version=version, run_id=run_id)


def _reset_singletons() -> None:
    """Reset module-level singletons between tests."""
    import src.serving.model_loader as ml

    ml._scorer = None
    ml._model_version = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singletons():
    """Ensure each test starts with a clean singleton state."""
    _reset_singletons()
    yield
    _reset_singletons()


# ---------------------------------------------------------------------------
# load_model — happy path
# ---------------------------------------------------------------------------


class TestLoadModelSuccess:
    def test_sets_model_version(self):
        mock_raw_model = MagicMock()
        mock_scorer = MagicMock()

        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow"),
            patch("src.serving.model_loader.mlflow") as mock_mlflow,
            patch(
                "src.serving.model_loader.PersonalizedScorer",
                return_value=mock_scorer,
            ),
        ):
            mock_mlflow.sklearn.load_model.return_value = mock_raw_model
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = [_make_version("5")]

            import src.serving.model_loader as ml

            ml.load_model()

            assert ml._model_version == "5"

    def test_sets_scorer(self):
        mock_raw_model = MagicMock()
        mock_scorer = MagicMock()

        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow") as mock_mlflow,
            patch(
                "src.serving.model_loader.PersonalizedScorer",
                return_value=mock_scorer,
            ),
        ):
            mock_mlflow.sklearn.load_model.return_value = mock_raw_model
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = [_make_version("5")]

            import src.serving.model_loader as ml

            ml.load_model()

            assert ml._scorer is mock_scorer

    def test_scorer_initialised_with_raw_model(self):
        mock_raw_model = MagicMock()

        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow") as mock_mlflow,
            patch("src.serving.model_loader.PersonalizedScorer") as mock_scorer_cls,
        ):
            mock_mlflow.sklearn.load_model.return_value = mock_raw_model
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = [_make_version("2")]

            import src.serving.model_loader as ml

            ml.load_model()

            mock_scorer_cls.assert_called_once_with(model=mock_raw_model)

    def test_queries_correct_stage(self):
        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow") as mock_mlflow,
            patch("src.serving.model_loader.PersonalizedScorer"),
        ):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = [_make_version()]

            import src.serving.model_loader as ml

            ml.load_model()

            mock_client.get_latest_versions.assert_called_once_with(
                ml.REGISTERED_MODEL_NAME, stages=[ml.MODEL_STAGE]
            )


# ---------------------------------------------------------------------------
# load_model — failure paths (all must sys.exit(1))
# ---------------------------------------------------------------------------


class TestLoadModelFailures:
    def test_exits_when_no_production_version(self):
        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow"),
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = []

            import src.serving.model_loader as ml

            with pytest.raises(SystemExit) as exc_info:
                ml.load_model()

            assert exc_info.value.code == 1

    def test_exits_when_registry_query_raises(self):
        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow"),
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.side_effect = ConnectionError(
                "MLflow unreachable"
            )

            import src.serving.model_loader as ml

            with pytest.raises(SystemExit) as exc_info:
                ml.load_model()

            assert exc_info.value.code == 1

    def test_exits_when_client_init_raises(self):
        with (
            patch(
                "src.serving.model_loader.MlflowClient",
                side_effect=RuntimeError("bad URI"),
            ),
            patch("src.serving.model_loader.mlflow"),
        ):
            import src.serving.model_loader as ml

            with pytest.raises(SystemExit) as exc_info:
                ml.load_model()

            assert exc_info.value.code == 1

    def test_exits_when_artifact_load_raises(self):
        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow") as mock_mlflow,
        ):
            mock_mlflow.sklearn.load_model.side_effect = OSError("artifact not found")
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = [_make_version()]

            import src.serving.model_loader as ml

            with pytest.raises(SystemExit) as exc_info:
                ml.load_model()

            assert exc_info.value.code == 1

    def test_exits_when_scorer_init_raises(self):
        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow") as mock_mlflow,
            patch(
                "src.serving.model_loader.PersonalizedScorer",
                side_effect=ImportError("missing dependency"),
            ),
        ):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = [_make_version()]

            import src.serving.model_loader as ml

            with pytest.raises(SystemExit) as exc_info:
                ml.load_model()

            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_raises_before_load(self):
        import src.serving.model_loader as ml

        with pytest.raises(RuntimeError, match="load_model"):
            ml.get_model()

    def test_returns_scorer_after_load(self):
        mock_scorer = MagicMock()

        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow") as mock_mlflow,
            patch(
                "src.serving.model_loader.PersonalizedScorer",
                return_value=mock_scorer,
            ),
        ):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = [_make_version()]

            import src.serving.model_loader as ml

            ml.load_model()
            result = ml.get_model()

        assert result is mock_scorer


# ---------------------------------------------------------------------------
# get_model_version
# ---------------------------------------------------------------------------


class TestGetModelVersion:
    def test_returns_none_before_load(self):
        import src.serving.model_loader as ml

        assert ml.get_model_version() is None

    def test_returns_version_string_after_load(self):
        with (
            patch("src.serving.model_loader.MlflowClient") as mock_client_cls,
            patch("src.serving.model_loader.mlflow") as mock_mlflow,
            patch("src.serving.model_loader.PersonalizedScorer"),
        ):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_latest_versions.return_value = [_make_version("7")]

            import src.serving.model_loader as ml

            ml.load_model()

        assert ml.get_model_version() == "7"
