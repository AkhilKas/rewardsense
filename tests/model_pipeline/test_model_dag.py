import pytest
from airflow.models import DagBag

pytestmark = pytest.mark.skip(reason="Needs Airflow DB, fails in CI without it")

@pytest.fixture(scope="session")
def dagbag():
    return DagBag(dag_folder="dags/", include_examples=False)


def test_model_pipeline_dag_loaded(dagbag):
    """Test that the DAG loads without errors."""
    assert dagbag.import_errors == {}, f"DAG import failures: {dagbag.import_errors}"
    dag = dagbag.get_dag(dag_id="rewardsense_model_pipeline")
    assert dag is not None
    assert len(dag.tasks) > 0


def test_model_pipeline_dag_structure(dagbag):
    """Test task dependencies and acyclicity logic."""
    dag = dagbag.get_dag(dag_id="rewardsense_model_pipeline")

    # Check that it awaits data pipeline
    assert dag.has_task("wait_for_data_pipeline")

    # Check groups exist
    assert dag.has_task("data_preparation.data_loading")
    assert dag.has_task("model_development.model_training")
    assert dag.has_task("quality_gates.validation")
    assert dag.has_task("deployment.registry_push")

    # Ensure graph is acyclic
    # Airflow topological_sort throws AirflowException if cycle detected
    try:
        dag.topological_sort()
    except Exception as e:
        pytest.fail(f"DAG has cycles: {e}")
