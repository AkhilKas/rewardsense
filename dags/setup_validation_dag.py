"""
RewardSense - Example DAG for verifying Airflow setup.

This DAG validates that:
1. Airflow scheduler discovers DAGs from the mounted volume
2. Task execution works with LocalExecutor
3. Python dependencies are available inside the container
4. GCP connection is configured (when Story 1.2 is complete)

Schedule: None (manual trigger only)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "rewardsense",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def check_dependencies():
    """Verify that key Python packages are importable."""
    import pandas as pd
    import numpy as np
    import great_expectations as ge
    import yaml

    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"great_expectations: {ge.__version__}")
    print(f"pyyaml: {yaml.__version__}")
    print("All core dependencies verified.")


def check_gcp_connection():
    """
    Test GCP connectivity. This will fail gracefully until
    Story 1.2 (GCP setup) is complete - that's expected.
    """
    try:
        from google.cloud import storage

        client = storage.Client()
        buckets = list(client.list_buckets(max_results=1))
        print(f"GCP connection successful. Found {len(buckets)} bucket(s).")
    except Exception as e:
        print(f"GCP connection not yet configured (expected until Story 1.2): {e}")
        # Don't raise - this is informational, not a hard failure
        # Remove this try/except once GCP is wired up


def check_src_mount():
    """Verify that the src/ directory is accessible from within the container."""
    import os

    src_path = "/opt/airflow/src"
    if os.path.isdir(src_path):
        contents = os.listdir(src_path)
        print(f"src/ mounted successfully. Contents: {contents}")
    else:
        print(f"WARNING: {src_path} not found. Check volume mount.")


with DAG(
    dag_id="setup_validation",
    default_args=default_args,
    description="Validates Airflow environment setup for RewardSense",
    schedule=None,  # Manual trigger only
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["setup", "validation", "rewardsense"],
) as dag:

    t1_echo = BashOperator(
        task_id="echo_environment",
        bash_command=(
            'echo "Airflow is running!" && '
            'echo "Executor: $AIRFLOW__CORE__EXECUTOR" && '
            'echo "DAGs folder: $AIRFLOW__CORE__DAGS_FOLDER" && '
            'echo "GCP Project: $GCP_PROJECT_ID" && '
            'echo "GCP Bucket: $GCP_BUCKET_NAME"'
        ),
    )

    t2_deps = PythonOperator(
        task_id="check_python_dependencies",
        python_callable=check_dependencies,
    )

    t3_gcp = PythonOperator(
        task_id="check_gcp_connection",
        python_callable=check_gcp_connection,
    )

    t4_src = PythonOperator(
        task_id="check_src_mount",
        python_callable=check_src_mount,
    )

    t1_echo >> t2_deps >> [t3_gcp, t4_src]
