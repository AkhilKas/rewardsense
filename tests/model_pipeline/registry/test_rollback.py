from unittest.mock import patch, MagicMock
from src.model_pipeline.registry.rollback import ModelRollbackManager


@patch("src.model_pipeline.registry.rollback.RegistryClient")
@patch("src.model_pipeline.registry.rollback.mlflow.get_experiment_by_name")
@patch("src.model_pipeline.registry.rollback.mlflow.search_runs")
@patch("src.model_pipeline.registry.rollback.mlflow.set_experiment")
@patch("src.model_pipeline.registry.rollback.mlflow.start_run")
@patch("src.model_pipeline.registry.rollback.mlflow.log_param")
def test_rollback_success(
    mock_log, mock_start, mock_set, mock_search, mock_exp, mock_registry
):
    # Mock MLflow giving us 2 previous runs (current and previous)
    mock_df = MagicMock()
    mock_df.__len__.return_value = 2

    mock_row_0 = MagicMock()
    mock_row_0.run_id = "curr_id"
    mock_row_1 = MagicMock()
    mock_row_1.run_id = "prev_id"

    mock_df.iloc = [mock_row_0, mock_row_1]
    mock_search.return_value = mock_df

    manager = ModelRollbackManager("pid", "bucket", "personalization")

    with patch.object(manager.notifier, "notify") as mock_notify:
        result = manager.execute_rollback("Bad validation metrics")

        assert result is True
        mock_notify.assert_any_call(
            "Initiating rollback for personalization. Reason: Bad validation metrics",
            level="CRITICAL",
        )
        mock_notify.assert_any_call(
            "Rollback successful. Model personalization reverted to prev_id.",
            level="INFO",
        )


@patch("src.model_pipeline.registry.rollback.RegistryClient")
@patch("src.model_pipeline.registry.rollback.mlflow.get_experiment_by_name")
@patch("src.model_pipeline.registry.rollback.mlflow.search_runs")
def test_rollback_no_previous_version(mock_search, mock_exp, mock_registry):
    # Mock MLflow giving us only 1 run, so no previous version exists
    mock_df = MagicMock()
    mock_df.__len__.return_value = 1
    mock_search.return_value = mock_df

    manager = ModelRollbackManager("pid", "bucket", "personalization")

    with patch.object(manager.notifier, "notify") as mock_notify:
        result = manager.execute_rollback("Bad testing metric")

        assert result is False
        mock_notify.assert_any_call(
            "Rollback failed: Could not determine previous version.", level="CRITICAL"
        )
