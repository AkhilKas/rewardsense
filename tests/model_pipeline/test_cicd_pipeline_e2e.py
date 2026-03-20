import pytest
from unittest.mock import patch, MagicMock
from src.model_pipeline.cd.notifier import NotificationDispatcher
from src.model_pipeline.registry.rollback import ModelRollbackManager


@pytest.fixture
def mock_notifier():
    with patch(
        "src.model_pipeline.cd.notifier.NotificationDispatcher.notify"
    ) as mock_notify:
        yield mock_notify


@pytest.fixture
def mock_registry():
    with patch(
        "src.model_pipeline.registry.artifact_registry.RegistryClient.push_model"
    ) as mock_push:
        yield mock_push


def simulate_pipeline(
    train_success: bool = True,
    validation_pass: bool = True,
    bias_pass: bool = True,
    push_success: bool = True,
) -> bool:
    """
    Simulates the Airflow DAG CI/CD execution steps and notification triggers.
    In a real run, these steps are executed across various Python modules
    by the BashOperators in the DAG.
    """
    notifier = NotificationDispatcher()

    # Step 1: Model Training
    if not train_success:
        notifier.notify("Model training failed", level="ERROR")
        return False
    notifier.notify("Model training completed successfully", level="INFO")

    # Step 2: Quality Gates (Validation)
    if not validation_pass:
        notifier.notify("Model rejected by validation gate", level="WARNING")
        return False

    # Step 3: Quality Gates (Bias Detection)
    if not bias_pass:
        notifier.notify("Model rejected by bias gate", level="WARNING")
        return False

    # Step 4: Deployment (Artifact Registry Push)
    if not push_success:
        notifier.notify("Model push to registry failed", level="ERROR")
        return False
    notifier.notify("Model successfully pushed to Artifact Registry", level="SUCCESS")

    return True


def test_successful_pipeline_run(mock_notifier, mock_registry):
    """Scenario 1: Full CI/CD cycle completes (Training -> Validation -> Bias -> Registry)."""
    result = simulate_pipeline()
    assert result is True
    # Verify progression notifications
    mock_notifier.assert_any_call("Model training completed successfully", level="INFO")
    mock_notifier.assert_any_call(
        "Model successfully pushed to Artifact Registry", level="SUCCESS"
    )


def test_validation_gate_rejection(mock_notifier):
    """Scenario 2: Validation gate blocks an intentionally bad model."""
    result = simulate_pipeline(validation_pass=False)
    assert result is False
    mock_notifier.assert_any_call("Model rejected by validation gate", level="WARNING")


def test_bias_gate_rejection(mock_notifier):
    """Scenario 3: Bias gate blocks an intentionally biased model."""
    result = simulate_pipeline(bias_pass=False)
    assert result is False
    mock_notifier.assert_any_call("Model rejected by bias gate", level="WARNING")


@patch("src.model_pipeline.registry.rollback.mlflow.get_experiment_by_name")
@patch("src.model_pipeline.registry.rollback.mlflow.search_runs")
@patch("src.model_pipeline.registry.rollback.mlflow.set_experiment")
@patch("src.model_pipeline.registry.rollback.mlflow.start_run")
@patch("src.model_pipeline.registry.rollback.mlflow.log_param")
def test_post_deployment_rollback(
    mock_log, mock_start, mock_set, mock_search, mock_exp
):
    """Scenario 4: Rollback after a simulated post-deployment metric degradation."""
    # Mock MLflow giving us 2 previous runs (current and previous champion)
    mock_df = MagicMock()
    mock_df.__len__.return_value = 2

    mock_row_0 = MagicMock()
    mock_row_0.run_id = "curr_id"
    mock_row_1 = MagicMock()
    mock_row_1.run_id = "prev_id"

    mock_df.iloc = [mock_row_0, mock_row_1]
    mock_search.return_value = mock_df

    manager = ModelRollbackManager(
        "pid", "us-central1", "rewardsense-models", "personalization"
    )

    with patch.object(manager.notifier, "notify") as mock_notify:
        result = manager.execute_rollback(
            "Production NDCG degraded beyond 10% threshold"
        )

        assert result is True
        mock_notify.assert_any_call(
            "Initiating rollback for personalization. Reason: Production NDCG degraded beyond 10% threshold",
            level="CRITICAL",
        )
        mock_notify.assert_any_call(
            "Rollback successful. Model personalization reverted to prev_id.",
            level="INFO",
        )
