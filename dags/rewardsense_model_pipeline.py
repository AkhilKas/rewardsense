import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger("airflow.task")

# Default arguments for the DAG
default_args = {
    "owner": "rewardsense",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def _trigger_serving_redeploy(**context):
    """Dispatch the serving_redeploy GitHub Actions workflow.

    Determines trigger_source from dag_run.conf so the audit log
    distinguishes between:
      - model_pipeline   : weekly scheduled training run
      - retrain_pipeline : monitoring-triggered retrain
    """
    import json as _json
    import urllib.error
    import urllib.request

    ti = context["ti"]
    dag_run = context["dag_run"]

    # Determine who originally triggered this pipeline run
    conf = dag_run.conf or {}
    triggered_by = conf.get("triggered_by", "")
    trigger_source = "retrain_pipeline" if "monitoring" in triggered_by else "model_pipeline"

    github_token = Variable.get("GITHUB_TOKEN")
    github_owner = Variable.get("GITHUB_OWNER", default_var="Raul2008NEU")
    github_repo = Variable.get("GITHUB_REPO", default_var="rewardsense")

    url = (
        f"https://api.github.com/repos/{github_owner}/{github_repo}"
        f"/actions/workflows/serving_redeploy.yml/dispatches"
    )
    payload = _json.dumps(
        {"ref": "main", "inputs": {"trigger_source": trigger_source}}
    ).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            http_status = resp.status
    except urllib.error.HTTPError as exc:
        logger.error("GitHub dispatch API error: %s %s", exc.code, exc.reason)
        raise

    logger.info(
        "Dispatched serving_redeploy workflow (trigger_source=%s, http_status=%d)",
        trigger_source,
        http_status,
    )

    ti.xcom_push(key="trigger_source", value=trigger_source)
    ti.xcom_push(key="github_dispatch_status", value=http_status)
    return {"status": "dispatched", "trigger_source": trigger_source, "http_status": http_status}


with DAG(
    "rewardsense_model_pipeline",
    default_args=default_args,
    description="Model training pipeline for personalized credit card recommendations",
    schedule="0 8 * * 0",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["model", "rewardsense", "ml"],
) as dag:
    base_task_env = {
        "SLACK_WEBHOOK_URL": Variable.get("SLACK_WEBHOOK_URL", default_var="")
    }

    wait_for_data_pipeline = ExternalTaskSensor(
        task_id="wait_for_data_pipeline",
        external_dag_id="rewardsense_data_pipeline",
        external_task_id=None,  # Wait for the whole DAG to complete
        execution_delta=timedelta(hours=2),
        mode="reschedule",
        timeout=3600,
        soft_fail=True,
    )

    # ── Data Preparation ──────────────────────────────────────────────
    with TaskGroup("data_preparation") as data_prep:
        data_loading = BashOperator(
            task_id="data_loading",
            bash_command='set -e; cd $DAGS_FOLDER; python -m src.model_pipeline.data_loader 1>&2; echo "data_loading: OK"',
            env=base_task_env,
            append_env=True,
            do_xcom_push=True,
        )

        feature_engineering = BashOperator(
            task_id="feature_engineering",
            bash_command='set -e; cd $DAGS_FOLDER; python -m src.model_pipeline.personalization.features 1>&2; echo "feature_engineering: OK"',
            env=base_task_env,
            append_env=True,
            do_xcom_push=True,
        )

        data_loading >> feature_engineering

    # ── Model Development ─────────────────────────────────────────────
    with TaskGroup("model_development") as model_dev:
        model_training = BashOperator(
            task_id="model_training",
            bash_command='set -e; cd $DAGS_FOLDER; python -m src.model_pipeline.train 1>&2; echo "model_training: OK"',
            env={
                "MLFLOW_TRACKING_URI": "https://mlflow-server-760934308287.us-central1.run.app",
                "SLACK_WEBHOOK_URL": Variable.get("SLACK_WEBHOOK_URL", default_var=""),
            },
            append_env=True,
            do_xcom_push=True,
        )

    # ── Quality Gates ─────────────────────────────────────────────────
    with TaskGroup("quality_gates") as quality_gates:
        validation = BashOperator(
            task_id="validation",
            bash_command="""set -e; cd $DAGS_FOLDER; python -c "
import json, sys
from src.model_pipeline.cd.gates import ValidationGate
metrics = json.load(open('/tmp/model_pipeline/metrics.json'))
gate = ValidationGate({'ndcg@10': 0.7})
passed = gate.evaluate(metrics)
assert passed, f'Validation Gate Failed: {metrics}'
print(json.dumps({'gate': 'validation', 'passed': True, 'ndcg_at_10': metrics.get('ndcg@10')}))
" """,
            env=base_task_env,
            append_env=True,
            do_xcom_push=True,
        )

        bias_detection = BashOperator(
            task_id="bias_detection",
            bash_command="""set -e; cd $DAGS_FOLDER; python -c "
import json
from src.model_pipeline.cd.gates import BiasGate
gate = BiasGate()
passed = gate.evaluate('/tmp/model_pipeline/bias_report.json')
assert passed, 'Bias Gate Failed'
print(json.dumps({'gate': 'bias_detection', 'passed': True}))
" """,
            env=base_task_env,
            append_env=True,
            do_xcom_push=True,
        )

        validation >> bias_detection

    # ── Deployment ────────────────────────────────────────────────────
    with TaskGroup("deployment") as deployment:
        registry_push = BashOperator(
            task_id="registry_push",
            bash_command="""set -e; cd $DAGS_FOLDER; python -c "
import json
from src.model_pipeline.cd.gates import RegistryGate
metrics = json.load(open('/tmp/model_pipeline/metrics.json'))
version = 'v' + str(metrics.get('run_id', '1.0'))
gate = RegistryGate('rewardsense-prod', 'us-central1', 'rewardsense-models', 'personalization')
result = gate.push('/tmp/model_pipeline/model_artifact', version)
print(json.dumps({'gate': 'registry_push', 'version': version, 'result': str(result)}))
" """,
            env=base_task_env,
            append_env=True,
            do_xcom_push=True,
        )

    # ── Serving Redeployment ──────────────────────────────────────────
    trigger_redeploy = PythonOperator(
        task_id="trigger_serving_redeploy",
        python_callable=_trigger_serving_redeploy,
        doc_md=(
            "Dispatch the serving_redeploy GitHub Actions workflow so the "
            "Cloud Run service restarts and loads the newly promoted MLflow model. "
            "Requires GITHUB_TOKEN Airflow Variable (PAT with `workflow` scope)."
        ),
    )

    # DAG execution order
    wait_for_data_pipeline >> data_prep >> model_dev >> quality_gates >> deployment >> trigger_redeploy
