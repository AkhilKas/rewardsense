import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path

# Skip these tests in CI where Airflow is not installed
pytest.importorskip("airflow", reason="Airflow not installed in CI")

# Import the callables. We can import them directly since they don't execute logic at import time
from dags.rewardsense_data_pipeline import (  # noqa: E402
    _clean_data,
    _engineer_features,
    _run_transform_pipeline,
)

"""
Unit tests for the preprocessing tasks in rewardsense_data_pipeline.py.
These tests mock the inner TransformationPipeline logic to ensure the Airflow tasks
wrap the execution correctly without hitting actual data files.
"""


@pytest.fixture
def mock_pipeline_class():
    with patch(
        "data_pipeline.preprocessing.transform.TransformationPipeline"
    ) as MockClass:
        yield MockClass


@pytest.fixture
def mock_context():
    return {"task_instance": MagicMock(), "dag_run": MagicMock()}


def test_clean_data_task_success(mock_pipeline_class, mock_context):
    """Test that _clean_data properly instantiates the pipeline and calls load/clean."""
    mock_instance = mock_pipeline_class.return_value

    # Mock return values for step_load
    mock_cards = pd.DataFrame({"dummy": [1]})
    mock_txns = pd.DataFrame({"dummy": [2]})
    mock_users = pd.DataFrame({"dummy": [3]})
    mock_load_report = {"loaded": True}
    mock_instance._step_load.return_value = (
        mock_cards,
        mock_txns,
        mock_users,
        mock_load_report,
    )

    # Mock return values for step_clean
    mock_clean_cards = pd.DataFrame({"dummy_clean": [1]})
    mock_clean_txns = pd.DataFrame({"dummy_clean": [2]})
    mock_clean_users = pd.DataFrame({"dummy_clean": [3]})
    mock_clean_report = {"cleaned": True}
    mock_instance._step_clean.return_value = (
        mock_clean_cards,
        mock_clean_txns,
        mock_clean_users,
        mock_clean_report,
    )

    result = _clean_data(**mock_context)

    # Verify instantiations and method calls
    mock_pipeline_class.assert_called_once_with(
        config_path=Path("config/transform.yaml")
    )
    mock_instance._step_load.assert_called_once()
    mock_instance._step_clean.assert_called_once_with(mock_cards, mock_txns, mock_users)

    assert result == {"status": "success", "report": mock_clean_report}


def test_engineer_features_task_with_checkpoints(mock_pipeline_class, mock_context):
    """Test that _engineer_features uses checkpoints when available."""
    mock_instance = mock_pipeline_class.return_value
    mock_instance.checkpoints_enabled = True
    mock_instance._checkpoint_exists.return_value = True

    # Mock data loading
    mock_clean_df = pd.DataFrame({"clean": [1]})
    mock_instance._load_df.return_value = mock_clean_df

    # Mock feature generation
    mock_f_report = {"features": "ok"}
    mock_instance._step_features.return_value = (
        mock_clean_df,
        mock_clean_df,
        mock_clean_df,
        mock_f_report,
    )

    # Mock checkpoint dir
    mock_instance._step_ckpt_dir.return_value = Path("/tmp/mock_ckpt")

    # Needs to patch Path.exists since we are checking it in the callable
    with patch("pathlib.Path.exists", return_value=True):
        result = _engineer_features(**mock_context)

    # Verify we didn't call step_load or step_clean
    mock_instance._step_load.assert_not_called()
    mock_instance._step_clean.assert_not_called()

    # Verify load_df was used
    assert mock_instance._load_df.call_count == 3
    mock_instance._step_features.assert_called_once()
    assert result == {"status": "success", "report": mock_f_report}


def test_engineer_features_task_without_checkpoints(mock_pipeline_class, mock_context):
    """Test that _engineer_features falls back to load/clean if no checkpoints."""
    mock_instance = mock_pipeline_class.return_value
    mock_instance.checkpoints_enabled = False
    mock_instance._checkpoint_exists.return_value = False

    mock_instance._step_load.return_value = (None, None, None, {})
    mock_instance._step_clean.return_value = (None, None, None, {})
    mock_f_report = {"features": "ok_no_ckpt"}
    mock_instance._step_features.return_value = (None, None, None, mock_f_report)

    result = _engineer_features(**mock_context)

    # Verify fallbacks are used
    mock_instance._step_load.assert_called_once()
    mock_instance._step_clean.assert_called_once()
    mock_instance._step_features.assert_called_once()
    assert result == {"status": "success", "report": mock_f_report}


def test_run_transform_pipeline_with_checkpoints(mock_pipeline_class, mock_context):
    """Test that _run_transform_pipeline uses checkpoints for finalizing."""
    mock_instance = mock_pipeline_class.return_value
    mock_instance.checkpoints_enabled = True
    mock_instance._checkpoint_exists.return_value = True

    mock_df = pd.DataFrame({"f": [1]})
    mock_instance._load_df.return_value = mock_df
    mock_instance._step_ckpt_dir.return_value = Path("/tmp/mock_ckpt_f")

    mock_outputs = {"out": "final.csv"}
    mock_instance._write_final_outputs.return_value = mock_outputs

    with patch("pathlib.Path.exists", return_value=True):
        result = _run_transform_pipeline(**mock_context)

    mock_instance.run.assert_not_called()
    mock_instance._write_final_outputs.assert_called_once()
    assert result == {"status": "success", "outputs": mock_outputs}


def test_run_transform_pipeline_without_checkpoints(mock_pipeline_class, mock_context):
    """Test that _run_transform_pipeline runs the full pipeline if no checkpoints."""
    mock_instance = mock_pipeline_class.return_value
    mock_instance.checkpoints_enabled = False
    mock_instance._checkpoint_exists.return_value = False

    mock_run_out = {"full_run": True}
    mock_instance.run.return_value = mock_run_out

    result = _run_transform_pipeline(**mock_context)

    mock_instance.run.assert_called_once()
    mock_instance._write_final_outputs.assert_not_called()
    assert result == mock_run_out
