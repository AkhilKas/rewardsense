# RewardSense Model Pipeline README & Reproduction Guide (Story 8.1)

This guide is the source of truth for reproducing the full Phase 2 model pipeline end-to-end.

## 1. What This Runs

The model pipeline covers three components:

1. Deterministic scoring engine (`src/model_pipeline/scoring`)
2. Personalization model (`src/model_pipeline/personalization`)
3. LLM explainability layer (`src/model_pipeline/llm` + `src/app/server.py`)

## 2. Prerequisites

- macOS/Linux with Python 3.11+
- Docker Desktop
- Google Cloud SDK (`gcloud`, `gsutil`)
- DVC CLI with GCS support (`dvc`, `dvc-gs`)
- Access to project buckets and MLflow Cloud Run endpoint

## 3. Environment Setup (Exact Commands)

```bash
git clone https://github.com/avadharj/rewardsense.git
cd rewardsense
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-ci.txt
pip install -r requirements-model.txt
pip install -e .
```

### 3.1 GCP Credentials

```bash
gcloud init
gcloud auth application-default login
gcloud config set project rewardsense
```

### 3.2 Required Environment Variables

```bash
cp .env.example .env
export EXECUTION_ENV=gcp
export GCP_PROJECT_ID=rewardsense
export GCP_BUCKET_NAME=rewardsense-dvc-store
export MLFLOW_TRACKING_URI=https://mlflow-server-760934308287.us-central1.run.app
export SLACK_WEBHOOK_URL='<set-from-secure-secret-store>'
export PYTHONPATH=.
```

## 4. Start Local Model Services

```bash
docker compose --profile model up -d mlflow-server
```

Local MLflow UI: `http://localhost:5001`

Cloud MLflow UI: <https://mlflow-server-760934308287.us-central1.run.app>

## 5. Reproduce Full Pipeline (DVC Pull -> Training -> Evaluation -> Bias -> Registry)

### Step A: Pull Phase 1 data artifacts

```bash
dvc pull
dvc status
```

### Step B: Run model pipeline training entrypoint

```bash
python -m src.model_pipeline.train
```

Expected outputs:

- `/tmp/model_pipeline/metrics.json`
- `/tmp/model_pipeline/bias_report.json`
- `/tmp/model_pipeline/model_artifact/model.joblib`

### Step C: Run validation + bias gates

```bash
python - <<'PY'
import json
from src.model_pipeline.cd.gates import ValidationGate, BiasGate

metrics = json.load(open('/tmp/model_pipeline/metrics.json'))
assert ValidationGate({'ndcg@10': 0.7}).evaluate(metrics)
assert BiasGate(max_disparity=0.10).evaluate('/tmp/model_pipeline/bias_report.json')
print('validation+bias gates: PASS')
PY
```

### Step D: Push model artifact to registry

```bash
python - <<'PY'
import json
from src.model_pipeline.cd.gates import RegistryGate

metrics = json.load(open('/tmp/model_pipeline/metrics.json'))
version = 'v' + str(metrics.get('run_id', 'manual'))
result = RegistryGate(
    project_id='rewardsense-prod',
    location='us-central1',
    repository='rewardsense-models',
    model_name='personalization',
).push('/tmp/model_pipeline/model_artifact', version)
print('registry push result:', result)
PY
```

## 6. Model Architecture Decisions (with rationale)

- Deterministic scoring for correctness and explainability of reward economics.
- Personalization model predicts user point valuation multiplier to adapt ranking by profile.
- LLM generates natural-language rationale while guardrails enforce factual consistency.
- Bias checks are separate gates so fairness regressions can block promotion.
- Registry + rollback logic separates experimentation from production promotion risk.

## 7. MLflow Dashboards and Key Run IDs

- Cloud dashboard: <https://mlflow-server-760934308287.us-central1.run.app>
- Main experiments:
  - `reward-scoring`
  - `personalization-point-valuation`
  - `llm-explainability`

Key run tracking table (update after every release):

| Date (UTC) | Experiment | Run Name | Run ID |
|---|---|---|---|
| 2026-03-23 | `Default` | `cloudrun-persistence-verify-20260323` | `ffda1800bb634b50885f65cec807b1c6` |
| 2026-03-24 | `personalization-point-valuation` | `xgboost` | `<add-latest-run-id>` |
| 2026-03-24 | `llm-explainability` | `llm-latency-single_transaction_recommendation` | `<add-latest-run-id>` |

To fetch latest run IDs:

```bash
python - <<'PY'
import mlflow
mlflow.set_tracking_uri('https://mlflow-server-760934308287.us-central1.run.app')
for exp_name in ['reward-scoring','personalization-point-valuation','llm-explainability']:
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        print(exp_name, 'missing')
        continue
    runs = mlflow.search_runs([exp.experiment_id], order_by=['start_time DESC'], max_results=3)
    print('\n', exp_name)
    print(runs[['run_id','tags.mlflow.runName','start_time']])
PY
```

## 8. Troubleshooting (Top 5)

1. GCP auth errors (`403`, ADC missing)

```bash
gcloud auth application-default login
gcloud auth list
gsutil ls gs://rewardsense-mlflow-artifacts/
```

2. Docker service not available

```bash
docker info
docker compose --profile model down
docker compose --profile model up -d mlflow-server
```

3. MLflow connection/timeouts

```bash
curl -I https://mlflow-server-760934308287.us-central1.run.app
export MLFLOW_TRACKING_URI=https://mlflow-server-760934308287.us-central1.run.app
```

4. Missing model output files under `/tmp/model_pipeline`

```bash
python -m src.model_pipeline.train
ls -lah /tmp/model_pipeline
```

5. Registry push fails (IAM or repo access)

```bash
gcloud auth application-default login
# verify service account and permissions in GCP console/IAM
```

## 9. Reproducibility Checklist

- [ ] Fresh clone + dependency install complete
- [ ] `dvc pull` succeeds
- [ ] `python -m src.model_pipeline.train` succeeds
- [ ] validation and bias gates pass
- [ ] artifact push to registry succeeds
- [ ] MLflow run IDs recorded in release notes
