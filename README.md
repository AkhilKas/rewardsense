# RewardSense

Cost-aware, explainable credit-card rewards data platform and MLOps pipeline.

RewardSense collects card-offer data from multiple sources, generates synthetic user/transaction data, preprocesses and engineers features, validates data quality, profiles datasets, detects anomalies, versions outputs with DVC, and publishes pipeline metrics/reports through Apache Airflow (local and Cloud Composer).

Repository: [https://github.com/avadharj/rewardsense](https://github.com/avadharj/rewardsense)

---

## 1. Project Overview

### What this repository contains

1. A production-style Airflow DAG (`rewardsense_data_pipeline`) for end-to-end orchestration.
2. Modular pipeline code under `src/data_pipeline/*` for ingestion, preprocessing, validation, profiling, anomaly detection, and monitoring.
3. DVC-based data versioning with a GCS remote.
4. Comprehensive testing (unit, DAG-structure, and integration tests).
5. Strict CI gates for linting, formatting, typing, and test coverage.

### Core pipeline stages

1. Ingestion
2. Preprocessing
3. Quality (schema validation + profiling)
4. Anomaly detection + quality gate
5. Versioning (DVC)
6. Reporting/monitoring

### High-level DAG flow

```text
pipeline_start
  -> ingestion
  -> preprocessing
  -> quality
  -> anomaly_detection
  -> versioning
  -> reporting
  -> pipeline_end
```

---

## 2. Epic/Story Delivery Model

The project is intentionally implemented in **epics and stories** (see `/Implementation_Phase_1.md` and `/Scoping_doc.md`).

### Delivery structure in this repo

1. Epic 1: Environment and infrastructure setup
2. Epic 2: Data acquisition (scrapers, API fetcher, synthetic generators)
3. Epic 3: Preprocessing and transformation pipeline
4. Epic 4: Testing framework
5. Epic 5: Airflow DAG orchestration and operational monitoring
6. Epic 6: Data quality
   - Story 6.2: Great Expectations-style schema validation
   - Story 6.3: Data profiling/statistics generation
7. Epic 7: Anomaly detection and quality gate enforcement
8. Epic 9: Pipeline performance monitoring and stage optimization

The implementation reflects this layering both in module structure and DAG task groups.

---

## 3. Repository Structure

```text
rewardsense/
  dags/
    rewardsense_data_pipeline.py         # Main production DAG
  src/
    data_pipeline/
      api_fetcher/                       # API clients + normalization
      scrapers/                          # NerdWallet + issuer scrapers
      generators/                        # Synthetic users + transactions
      preprocessing/                     # Cleaning, features, transform, normalization
      validation/                        # Great Expectations integration
      profiling/                         # Profiles, stats, viz helpers, history
      anomaly_detection/                 # Statistical + domain anomaly checks
      monitoring/                        # Perf instrumentation, alerts, reports
  tests/
    dags/                                # DAG contract/integration tests
    data_pipeline/                       # Unit tests by module
    integration/                         # End-to-end tests
  config/
    *.yaml                               # Runtime configs (transform, anomaly, alerting, etc.)
  data/
    processed/current/                   # Current pipeline outputs (DVC tracked)
  .github/workflows/
    ci.yml                               # Strict CI checks
    dvc-version-commit.yaml              # DVC metadata commit automation
  docs/
    gcp_setup.md
    data_card.md
```

---

## 4. Environment Setup (Reproducible)

These steps are designed so another engineer can clone and run without manual guesswork.

### 4.1 Prerequisites

1. Python 3.11 recommended (supported: 3.9+).
2. Git.
3. `pip` and `venv`.
4. Optional but recommended:
   - Docker Desktop (for local Airflow)
   - Google Cloud SDK (`gcloud`, `gsutil`)
   - DVC CLI

### 4.2 Clone and bootstrap

```bash
git clone https://github.com/avadharj/rewardsense.git
cd rewardsense
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.3 Install dependencies

For full local development + testing + quality checks:

```bash
pip install -r requirements-ci.txt
pip install -e .
```

If you specifically want Composer-compatible dependency parity:

```bash
pip install -r requirements_composer.txt
```

### 4.4 Configure environment variables

```bash
cp .env.example .env
```

Fill required values in `.env` (especially GCP/DVC/API-related values if you run cloud-backed workflows).

### 4.5 GCP authentication for DVC/GCS

If using GCP-backed storage:

```bash
gcloud init
gcloud auth application-default login
```

Validate bucket access:

```bash
gsutil ls gs://rewardsense-dvc-store
```

Detailed cloud setup: `docs/gcp_setup.md`.

---

## 5. Reproducibility and DVC (Critical)

RewardSense uses DVC to make data artifacts reproducible across machines.

### 5.1 Pull tracked data artifacts

After cloning:

```bash
dvc pull
```

This restores DVC-tracked data under `data/processed/current/*`.

### 5.2 Verify DVC state

```bash
dvc status
dvc list . --dvc-only -R
```

### 5.3 Regenerate and version new outputs

```bash
# run pipeline (local DAG or module flow)
# then:
dvc add data/processed/current/offers
dvc add data/processed/current/synthetic
dvc add data/processed/current/transformed

git add data/processed/current/*.dvc dvc.lock .gitignore
git commit -m "chore(data): update DVC tracking files"
dvc push
```

### 5.4 Why this guarantees reproducibility

1. `.dvc` files pin exact data content hashes.
2. DVC remote (`gs://rewardsense-dvc-store`) stores immutable content objects.
3. Teammates pull exact versions by checking out the same Git commit + running `dvc pull`.

---

## 6. Running the Pipeline

You can run RewardSense either locally (Docker Airflow) or in Cloud Composer.

## 6.1 Local Airflow (recommended for development)

### Start Airflow

```bash
docker compose up -d airflow-postgres airflow-init
docker compose up -d airflow-scheduler airflow-webserver
```

Airflow UI: `http://localhost:8080`

Default local credentials from compose init:

1. Username: `admin`
2. Password: `admin`

### Trigger DAG from CLI

```bash
docker compose exec -T airflow-scheduler \
  airflow dags trigger rewardsense_data_pipeline
```

### Inspect task states

```bash
docker compose exec -T airflow-scheduler \
  airflow tasks states-for-dag-run rewardsense_data_pipeline <run_id>
```

### Convenience script

```bash
bash scripts/test_airflow.sh
```

## 6.2 Cloud Composer run/deploy

### Deploy DAG code

```bash
gcloud composer environments storage dags import \
  --environment=rewardsense-composer-env \
  --location=us-central1 \
  --source=dags/rewardsense_data_pipeline.py
```

### Deploy module updates (example)

```bash
gcloud composer environments storage dags import \
  --environment=rewardsense-composer-env \
  --location=us-central1 \
  --source=src/data_pipeline \
  --destination=data_pipeline
```

### Trigger a Composer run

```bash
gcloud composer environments run rewardsense-composer-env \
  --location=us-central1 \
  dags trigger -- rewardsense_data_pipeline --run-id manual_verify_$(date +%Y%m%d_%H%M%S)
```

### Monitor run status

```bash
gcloud composer environments run rewardsense-composer-env \
  --location=us-central1 \
  tasks states-for-dag-run -- rewardsense_data_pipeline <run_id>
```

### Verify output artifacts in Composer bucket

```bash
gsutil ls -r 'gs://us-central1-rewardsense-com-8e7127ac-bucket/data/processed/current/**'
```

---

## 7. Quality, Profiling, Anomaly, and Performance Outputs

Pipeline emits operational artifacts into processed data paths.

### Key locations

1. Quality profiling artifacts:
   - `data/processed/current/profiling/`
2. Anomaly reports:
   - `data/processed/current/anomaly_reports/`
3. Transform outputs:
   - `data/processed/current/transformed/<run_id>/final/*.csv`
4. Performance metrics:
   - `data/metrics/performance/` (local) and equivalent Composer paths

### Expected anomaly files

1. `credit_cards_anomaly_report.json`
2. `transactions_anomaly_report.json`
3. `users_anomaly_report.json`

---

## 8. Testing Strategy (Comprehensive)

The repository includes:

1. Unit tests by module (`tests/data_pipeline/...`)
2. DAG contract/dependency tests (`tests/dags/...`)
3. Integration tests (`tests/integration/...`)
4. Schema/model tests (`tests/schemas/...`)

### Run all tests

```bash
pytest
```

### Run only DAG tests

```bash
pytest tests/dags -v
```

### Run integration tests

```bash
pytest -m integration -v
```

### Coverage standard

`pytest.ini` enforces:

1. strict markers
2. coverage reporting
3. `--cov-fail-under=75`

So the test suite fails if coverage drops below 75%.

---

## 9. CI/CD Rigor and Clean-Code Enforcement

CI workflow: `.github/workflows/ci.yml`

Every push/PR to `main` and `develop` is validated across Python 3.9/3.10/3.11 with:

1. Ruff lint checks
2. Black format checks (`--check`)
3. Mypy type checks (currently non-blocking)
4. Full pytest suite + coverage upload

This is intentionally strict to prevent drift in style, correctness, and reproducibility.

Data-versioning automation workflow: `.github/workflows/dvc-version-commit.yaml`

1. Uses GitHub OIDC + GCP Workload Identity Federation (no static GCP key in GitHub).
2. Authenticates as `rewardsense-pipeline@rewardsense.iam.gserviceaccount.com`.
3. Runs on:
   - `repository_dispatch` from Airflow (`event_type: dvc-commit`)
   - `workflow_dispatch` manual trigger
   - `push` to `main` when DVC/data paths change (`data/**`, `dvc.lock`, `*.dvc`)
4. Pulls/checks DVC state and commits `.dvc`/`dvc.lock` metadata updates when needed.

---

## 10. Data Validation and Profiling in Production Path

### Schema validation (Story 6.2)

`quality.validate_schema_expectations` runs Great Expectations-backed checks over:

1. merged card offers
2. synthetic transactions
3. synthetic users

Failures are logged with details; pipeline continues with warning semantics unless downstream gate policies block.

### Data profiling/statistics (Story 6.3)

`quality.generate_data_profiles` generates:

1. dataset row/column summaries
2. missingness statistics
3. numeric distribution summaries
4. historical snapshots for trend analysis

---

## 11. Anomaly Detection and Gate Behavior

Anomaly stage includes:

1. statistical anomaly checks
2. domain-rule checks
3. alert dispatch
4. critical gate

Alerting behavior (current configuration):

1. Slack: `WARNING` and `CRITICAL`
2. Email (SendGrid): `CRITICAL` only
3. Performance regression alerts are sent as `WARNING`

Alert runtime resolution in Composer:

1. Alert dispatcher reads config from `config/alerting_config.yaml` with Composer-safe path resolution.
2. Secrets are read from environment variables first, then Airflow Variables fallback.
3. Required keys:
   - `SLACK_WEBHOOK_URL`
   - `SLACK_CHANNEL`
   - `SENDGRID_API_KEY`
   - `ALERT_EMAIL`

Recent production verification confirmed:

1. `anomaly_detection.send_anomaly_alerts` sends Slack for warning/critical anomalies.
2. Critical anomaly alerts send both Slack and email.
3. Regression alerts are delivered via Slack.

Gate control variable:

1. `ANOMALY_GATE_ENFORCE=true`: blocks downstream versioning/reporting when critical anomalies are found.
2. `ANOMALY_GATE_ENFORCE=false`: logs critical anomalies but allows continuation (useful for verification/testing runs).

---

## 12. Performance Monitoring and Optimization

Stories 9.1/9.2 are integrated through timing instrumentation and regression analysis:

1. Python task timing decorators (`timed_python_task`)
2. Run snapshots with task spans/Gantt-compatible timing
3. Bottleneck identification and trend dashboards
4. Regression detection against historical run medians

Related module:

- `src/data_pipeline/monitoring/performance.py`

---

## 13. Important Composer Runtime Notes

1. Composer module imports should avoid `src.` prefixes in DAG runtime code paths.
2. Composer data root is `/home/airflow/gcs/data/processed/current`.
3. DAG bucket code root is under `/home/airflow/gcs/dags/`.
4. If you update modules under `src/data_pipeline/*`, re-import them to Composer DAG storage.
5. Composer DAG buckets may contain duplicate/stale module paths; always verify the active object path if behavior does not match local code.

---

## 14. Reproduce Pipeline on a New Machine (Checklist)

Use this exact sequence for a new developer machine:

1. Clone repo and create Python 3.11 venv.
2. Install `requirements-ci.txt` and editable package (`pip install -e .`).
3. Copy `.env.example` to `.env` and set required values.
4. Authenticate with GCP ADC if cloud-backed storage is required.
5. Run `dvc pull` to materialize tracked datasets.
6. Run `pytest` to verify local environment integrity.
7. Start local Airflow with Docker Compose.
8. Trigger `rewardsense_data_pipeline` and monitor task completion.
9. Verify outputs under `data/processed/current/*`.
10. If using cloud: deploy DAG/module updates to Composer and re-run verification.

If these steps pass, your local environment is functionally equivalent to the team baseline.

---

## 15. Common Troubleshooting

### `ModuleNotFoundError: src`

Run tests from repository root and ensure editable install:

```bash
pip install -e .
```

### `ModuleNotFoundError: pkg_resources`

Install setuptools in the active environment:

```bash
pip install setuptools
```

### Composer task can’t find transformed/profiling/anomaly outputs

1. Confirm code deployed to Composer DAG bucket.
2. Confirm paths point to `/home/airflow/gcs/data/processed/current`.
3. Verify with `gsutil ls -r` in the Composer bucket data prefix.

### Alerts are not sent even though tasks succeed

1. Confirm `config/alerting_config.yaml` exists in Composer DAG bucket.
2. Confirm Airflow Variables (or env vars) are set:
   - `SLACK_WEBHOOK_URL`, `SLACK_CHANNEL`, `SENDGRID_API_KEY`, `ALERT_EMAIL`
3. Check logs for:
   - `Alerting config not found ...` (config path issue)
   - `Slack enabled but SLACK_WEBHOOK_URL not set.`
   - `Email enabled but ALERT_EMAIL not set.`
4. If Composer has duplicate module objects, redeploy/overwrite the active `data_pipeline/monitoring/alerting.py` object path.

### DVC push says “Everything is up to date” but no Git history update

`dvc push` uploads data objects, but version history requires `.dvc` metadata commits. Ensure `.dvc` files and `dvc.lock` are committed to Git.

---

## 16. Documentation References

1. Phase implementation plan: `Implementation_Phase_1.md`
2. Product/system scope and architecture: `Scoping_doc.md`
3. Data card: `docs/data_card.md`
4. GCP setup details: `docs/gcp_setup.md`

---

## 17. License and Team

Team:

1. Aditya Shenoy
2. Akhilesh Kasturi
3. Arjun Vinay Avadhani
4. Rahul Suresh
5. Vidya Kalyandurg

Repository owner and coordination: [avadharj/rewardsense](https://github.com/avadharj/rewardsense)
