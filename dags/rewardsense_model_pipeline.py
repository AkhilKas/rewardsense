from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.task_group import TaskGroup

# Default arguments for the DAG
default_args = {
    "owner": "rewardsense",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "rewardsense_model_pipeline",
    default_args=default_args,
    description="Model training pipeline for personalized credit card recommendations",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["model", "rewardsense", "ml"],
) as dag:

    wait_for_data_pipeline = ExternalTaskSensor(
        task_id="wait_for_data_pipeline",
        external_dag_id="rewardsense_data_pipeline",
        external_task_id=None,  # Wait for the whole DAG to complete
        mode="reschedule",
        timeout=3600,
    )

    with TaskGroup("data_preparation") as data_prep:
        data_loading = BashOperator(
            task_id="data_loading",
            bash_command="cd $DAGS_FOLDER && python -m src.model_pipeline.data_loader",
        )

        feature_engineering = BashOperator(
            task_id="feature_engineering",
            bash_command="cd $DAGS_FOLDER && python -m src.model_pipeline.personalization.features",
        )

        data_loading >> feature_engineering

    with TaskGroup("model_development") as model_dev:
        model_training = BashOperator(
            task_id="model_training",
            bash_command="cd $DAGS_FOLDER && python -m src.model_pipeline.train",
        )

    with TaskGroup("quality_gates") as quality_gates:
        validation = BashOperator(
            task_id="validation",
            bash_command="""cd $DAGS_FOLDER && python -c "
import json
from src.model_pipeline.cd.gates import ValidationGate
metrics = json.load(open('/tmp/model_pipeline/metrics.json'))
gate = ValidationGate({'ndcg@10': 0.7})
assert gate.evaluate(metrics), f'Validation Gate Failed: {metrics}'
" """,
        )

        bias_detection = BashOperator(
            task_id="bias_detection",
            bash_command="""cd $DAGS_FOLDER && python -c "
from src.model_pipeline.cd.gates import BiasGate
gate = BiasGate()
assert gate.evaluate('/tmp/model_pipeline/bias_report.json'), 'Bias Gate Failed'
" """,
        )

        validation >> bias_detection

    with TaskGroup("deployment") as deployment:
        registry_push = BashOperator(
            task_id="registry_push",
            bash_command="""cd $DAGS_FOLDER && python -c "
import json
from src.model_pipeline.cd.gates import RegistryGate
metrics = json.load(open('/tmp/model_pipeline/metrics.json'))
version = 'v' + metrics.get('run_id', '1.0')
gate = RegistryGate('rewardsense-prod', 'us-central1', 'rewardsense-models', 'personalization')
gate.push('/tmp/model_pipeline/model_artifact', version)
" """,
        )

    # DAG execution order
    wait_for_data_pipeline >> data_prep >> model_dev >> quality_gates >> deployment
