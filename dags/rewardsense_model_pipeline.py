import os
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
            bash_command="echo 'Python script for pulling DVC data...'",
        )
        
        feature_engineering = BashOperator(
            task_id="feature_engineering",
            bash_command="echo 'Python script for model feature generation...'",
        )
        
        data_loading >> feature_engineering

    with TaskGroup("model_development") as model_dev:
        model_training = BashOperator(
            task_id="model_training",
            bash_command="python src/model_pipeline/train.py",
        )

    with TaskGroup("quality_gates") as quality_gates:
        validation = BashOperator(
            task_id="validation",
            bash_command="pytest tests/model_pipeline/cd/test_gates.py::test_validation_gate_pass",
        )
        
        bias_detection = BashOperator(
            task_id="bias_detection",
            bash_command="echo 'Executing Bias Reports...'",
        )
        
        validation >> bias_detection

    with TaskGroup("deployment") as deployment:
        registry_push = BashOperator(
            task_id="registry_push",
            bash_command="echo 'Pushing passing champion model to Artifact Registry...'",
        )

    # DAG execution order
    wait_for_data_pipeline >> data_prep >> model_dev >> quality_gates >> deployment
