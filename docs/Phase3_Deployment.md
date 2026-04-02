# RewardSense — Phase 3: Model Deployment Implementation Plan

## What Already Exists (from Phases 1 & 2)

- GCP project `rewardsense` with Cloud Composer (`rewardsense-composer-env`, `us-central1`)
- Service account `rewardsense-pipeline-sa@rewardsense.iam.gserviceaccount.com`
- Data pipeline DAG on Composer (weekly schedule)
- Model pipeline DAG on Composer (training → validation → bias detection → registry push)
- MLflow on Cloud Run (`https://mlflow-server-760934308287.us-central1.run.app`)
- LLM explainability layer via Gemini 2.0 Flash on Vertex AI
- GitHub Actions CI pipeline (lint, format, type checks, unit tests)
- Docker Compose for local development
- Deterministic scoring engine + ML personalization model + LLM explainer

## What Phase 3 Delivers

- A Cloud Run inference API serving card recommendations in real time
- Automated CD pipeline that deploys new model versions on push
- Monitoring dashboard with drift detection (Evidently AI)
- Automatic retraining trigger when drift/decay is detected
- Slack/email notifications for retraining and redeployment events
- React frontend for the Expo demo
- Full replication documentation and video demo

## Team Structure & Parallel Tracks

| Member | Track | Epics |
|--------|-------|-------|
| A | Serving Infrastructure + CD Pipeline | Epic 1, Epic 3 |
| B | Monitoring & Drift Detection | Epic 4, Epic 5 |
| C | Inference API Application Code | Epic 2 |
| D | React Frontend | Epic 6 |
| E | Documentation, Replication | Epic 7, Epic 8 |

**Key dependencies:**

- C needs A's Cloud Run service + Artifact Registry to deploy
- D needs C's API running to integrate
- B needs C's API to emit inference logs before drift detection works
- E needs everything working to record the video

---

## Epic 1: Serving Infrastructure Setup

**Owner:** Member A | **Total Points:** 16

### Story 1.1: Artifact Registry & Container Setup

**Story Points:** 3

**Description:** Create a GCP Artifact Registry Docker repository and configure authentication so that CI/CD and local builds can push/pull container images.

**Tasks:**

- Create Artifact Registry repository (`rewardsense-docker` in `us-central1`)
- Grant `rewardsense-pipeline-sa` the Artifact Registry Writer role
- Configure Docker credential helper for local development (`gcloud auth configure-docker us-central1-docker.pkg.dev`)
- Create base `Dockerfile.serving` for the inference service (Python 3.11, slim base)
- Test push/pull of a hello-world image to verify setup

**Acceptance Criteria:**

- `docker push` to `us-central1-docker.pkg.dev/rewardsense/rewardsense-docker/serving:test` succeeds
- Service account can pull images from Artifact Registry
- Dockerfile builds locally without errors

### Story 1.2: Cloud Run Serving Service

**Story Points:** 5

**Description:** Deploy a Cloud Run service that will host the inference API. Configure networking, autoscaling, and IAM so the service can access MLflow, GCS, and Vertex AI.

**Tasks:**

- Create Cloud Run service `rewardsense-serving` in `us-central1`
- Attach `rewardsense-pipeline-sa` as the service account
- Configure environment variables: `MLFLOW_TRACKING_URI`, `GCP_PROJECT`, `GCP_REGION`, `MODEL_STAGE=Production`
- Set autoscaling: min 0, max 5, concurrency 80, 2 GiB memory, 2 vCPU
- Configure startup probe on `/health` endpoint
- Set up VPC connector if MLflow Cloud Run service is on internal-only networking (check current config)
- Grant `roles/run.invoker` for unauthenticated access (or use IAP for the Expo — decide with team)

**Acceptance Criteria:**

- Cloud Run service deployed and accessible via HTTPS URL
- `/health` returns 200 with `{"status": "healthy"}`
- Service can reach MLflow tracking URI and GCS buckets
- Service can call Vertex AI Gemini endpoint

### Story 1.3: Model Loading from MLflow Registry

**Story Points:** 5

**Description:** Implement the model loading logic that pulls the latest Production-stage model from the MLflow registry at container startup and caches it in memory.

**Tasks:**

- Write `src/serving/model_loader.py`:
  - On startup, query MLflow for the latest model version in Production stage
  - Download model artifacts from GCS-backed MLflow artifact store
  - Load the scikit-learn personalization model into memory
  - Load the scoring engine configuration/weights
  - Expose a `get_model()` singleton for request handlers
- Implement model version caching (store version number, check on health endpoint)
- Add fallback: if MLflow is unreachable at startup, fail fast with clear error logs
- Write unit tests for model loading with mocked MLflow client

**Acceptance Criteria:**

- Container starts and loads the correct Production model within 30 seconds
- `/health` response includes `model_version` field
- If no Production model exists, container exits with non-zero code and descriptive log
- Unit tests pass with >90% coverage on `model_loader.py`

### Story 1.4: Integration Test — Deployed Service Smoke Test

**Story Points:** 3

**Description:** Verify that the Cloud Run service is fully operational end-to-end after initial deployment.

**Tasks:**

- Write `tests/integration/test_serving_deployment.py`:
  - Hit `/health` and verify model version matches MLflow Production stage
  - Hit `/predict` with a sample user profile and verify response schema
  - Verify response includes `recommended_cards`, `scores`, and `explanation` fields
  - Verify latency is under 10 seconds (includes LLM call)
- Run the integration test from GitHub Actions against the deployed URL
- Add the test to the CI workflow as a post-deploy step

**Acceptance Criteria:**

- Smoke test passes against live Cloud Run URL
- Test is automated in CI (runs after every deployment)
- Failures trigger a Slack notification (or GitHub Actions failure email)

---

## Epic 2: Inference API Application

**Owner:** Member C | **Total Points:** 21

### Story 2.1: FastAPI Application Scaffold

**Story Points:** 3

**Description:** Create the FastAPI application with request/response schemas, health check, and CORS configuration.

**Tasks:**

- Create `src/serving/app.py` with FastAPI instance
- Define Pydantic request schema: `PredictionRequest` (`user_id`, `spending_categories`, `monthly_spend`, `preferred_rewards`, `transaction_history`)
- Define Pydantic response schema: `PredictionResponse` (`recommended_cards` list with `card_name`, `score`, `rank`, `explanation`)
- Add `/health` endpoint returning service status, model version, uptime
- Add `/predict` endpoint stub
- Configure CORS (allow the React frontend origin)
- Add request logging middleware (log `request_id`, latency, status code)
- Write unit tests for schema validation (valid/invalid inputs)

**Acceptance Criteria:**

- `uvicorn src.serving.app:app` starts without errors
- `/health` returns valid JSON
- Invalid `/predict` payloads return 422 with descriptive errors
- CORS headers present in responses

### Story 2.2: Scoring Engine Integration

**Story Points:** 5

**Description:** Wire the deterministic scoring engine from Phase 2 into the `/predict` endpoint so that incoming user profiles get scored against all credit cards.

**Tasks:**

- Import the scoring engine module from `src/model_pipeline/scoring/`
- On `/predict`, run the user profile through the scoring engine
- Return ranked card list with deterministic scores
- Handle edge cases: unknown spending categories, missing fields (use defaults)
- Add request-level logging: log input features (anonymized), scores, and latency per stage
- Write unit tests with various user profiles (high spender, travel-focused, cashback-focused, etc.)

**Acceptance Criteria:**

- `/predict` returns ranked cards with scores for any valid input
- Scoring latency < 100ms (deterministic, no ML or LLM here)
- Edge cases handled gracefully with sensible defaults
- Unit tests cover at least 5 distinct user personas

### Story 2.3: Personalization Model Integration

**Story Points:** 5

**Description:** Layer the ML personalization model on top of the deterministic scores to re-rank cards based on learned user spending patterns.

**Tasks:**

- Load the personalization model via `model_loader.get_model()`
- On `/predict`, after deterministic scoring, run personalization model inference
- Blend deterministic score and personalization score (configurable weight, e.g., 0.6 deterministic + 0.4 ML)
- Re-rank cards based on blended score
- Log both score components for each card (for monitoring later)
- Write unit tests with mocked model that returns known outputs

**Acceptance Criteria:**

- Personalization model inference runs without errors
- Blended scores differ from pure deterministic scores (proves personalization has effect)
- Score blending weight is configurable via environment variable
- Total scoring latency (deterministic + ML) < 500ms

### Story 2.4: LLM Explanation Integration

**Story Points:** 5

**Description:** Call the Gemini-based explanation generator for the top-N recommended cards and include plain-language explanations in the API response.

**Tasks:**

- Import `ExplanationGenerator` from `src/model_pipeline/explainability/`
- After scoring and ranking, call Gemini for the top 3 cards
- Pass card name, score breakdown, and user profile context into the prompt
- Parse LLM response and attach to the response schema
- Implement timeout handling (5-second timeout per explanation, fall back to template-based explanation)
- Implement async calls to Gemini (call all 3 explanations concurrently with `asyncio.gather`)
- Log LLM latency per explanation
- Write unit tests with mocked LLM client

**Acceptance Criteria:**

- Top 3 cards in the response include human-readable `explanation` field
- LLM failures don't crash the endpoint (graceful fallback to template)
- Concurrent LLM calls reduce total latency compared to sequential
- Total `/predict` latency < 10 seconds end-to-end (including LLM)

### Story 2.5: Inference Logging for Monitoring

**Story Points:** 3

**Description:** Log every inference request's input features, output scores, and metadata to GCS so the monitoring system can consume them.

**Tasks:**

- Create `src/serving/inference_logger.py`
- On every `/predict` request, asynchronously write a JSON log record to a GCS bucket (`gs://rewardsense-inference-logs/YYYY/MM/DD/`)
- Log record includes: timestamp, request_id, input features (hashed user_id), predicted scores, top card, model version, latency breakdown
- Use background task (`FastAPI BackgroundTasks`) so logging doesn't add latency to the response
- Write unit tests for log record schema and GCS write logic (mocked)

**Acceptance Criteria:**

- Every `/predict` call produces a log record in GCS
- Logging adds < 5ms to response time (async)
- Log records are queryable by date partition
- Unit tests verify log schema completeness

---

## Epic 3: Automated CD Pipeline

**Owner:** Member A | **Total Points:** 13

### Story 3.1: GitHub Actions — Build & Push Container

**Story Points:** 5

**Description:** Extend the existing CI workflow with a CD stage that builds the serving Docker image and pushes it to Artifact Registry on merge to main.

**Tasks:**

- Add a `deploy` job to `.github/workflows/ci.yml` (runs after `test` job passes)
- Authenticate to GCP using Workload Identity Federation (already set up) or service account key
- Build the serving Docker image with the commit SHA as the tag
- Push to `us-central1-docker.pkg.dev/rewardsense/rewardsense-docker/serving:<sha>`
- Also tag as `latest`
- Cache Docker layers for faster builds

**Acceptance Criteria:**

- Every merge to main triggers a container build
- Image appears in Artifact Registry within 5 minutes of merge
- Build fails fast if tests fail (deploy job depends on test job)

### Story 3.2: GitHub Actions — Deploy to Cloud Run

**Story Points:** 5

**Description:** After pushing the container, automatically deploy the new image to the Cloud Run serving service.

**Tasks:**

- Add a `gcloud run deploy` step after the container push
- Deploy with `--image` pointing to the newly pushed SHA tag
- Set `--no-traffic` initially, then switch traffic after health check passes
- Implement rollback: if the new revision's health check fails within 60 seconds, route traffic back to the previous revision
- Add a post-deploy step that runs the integration smoke test (Story 1.4)
- On success, send a Slack notification with the new model version and commit SHA

**Acceptance Criteria:**

- New Cloud Run revision deployed automatically on merge to main
- Traffic only switches to new revision after health check passes
- Failed deployments automatically roll back
- Slack notification sent on successful deployment

### Story 3.3: Model-Triggered Redeployment

**Story Points:** 3

**Description:** When the model pipeline pushes a new model to the MLflow registry (Production stage), trigger a redeployment of the serving service to pick up the new model.

**Tasks:**

- **Option A:** Add a final task to the Composer model pipeline DAG that calls the GitHub Actions API to trigger the deploy workflow
- **Option B:** Set up a Cloud Function triggered by a Pub/Sub message from the model pipeline DAG's last task
- The redeployment rebuilds the container (model is loaded at startup from MLflow, so a restart is sufficient)
- Log the redeployment trigger source (manual push vs. model pipeline vs. retrain pipeline)

**Acceptance Criteria:**

- New Production model in MLflow → serving service restarts within 10 minutes
- Redeployment uses the same rollback safety as Story 3.2
- Audit log shows which event triggered each redeployment

---

## Epic 4: Model Monitoring & Drift Detection

**Owner:** Member B | **Total Points:** 18

### Story 4.1: Monitoring Data Collection

**Story Points:** 3

**Description:** Build a module that reads inference logs from GCS and computes summary statistics for the monitoring pipeline.

**Tasks:**

- Create `src/monitoring/data_collector.py`
- Read inference logs from `gs://rewardsense-inference-logs/` for a configurable time window (default: last 7 days)
- Aggregate: input feature distributions, prediction score distributions, top-card frequency, latency percentiles
- Output a structured summary as a Pandas DataFrame
- Write unit tests with sample log data

**Acceptance Criteria:**

- Collector reads and aggregates 7 days of logs in < 30 seconds
- Output DataFrame has consistent schema regardless of log volume
- Unit tests cover empty logs, single day, and multi-day windows

### Story 4.2: Evidently AI Drift Detection

**Story Points:** 5

**Description:** Implement data drift and prediction drift detection using Evidently AI, comparing recent inference data against the training data distribution.

**Tasks:**

- Create `src/monitoring/drift_detector.py`
- Load the reference dataset (training data distribution — store a reference profile in GCS during model training)
- Load the current dataset (from Story 4.1's collector output)
- Run Evidently `DataDriftPreset` for input feature drift
- Run Evidently `TargetDriftPreset` for prediction distribution drift
- Generate drift report as HTML (for dashboard) and JSON (for programmatic threshold check)
- Define configurable thresholds:
  - Feature drift: if > 30% of features have statistically significant drift → flag
  - Prediction drift: if KL divergence > 0.1 → flag
- Write unit tests with synthetic drift scenarios (no drift, mild drift, severe drift)

**Acceptance Criteria:**

- Drift detection runs against real inference logs and produces a valid Evidently report
- HTML report is human-readable and stored in GCS (`gs://rewardsense-monitoring/drift-reports/`)
- JSON output includes a boolean `drift_detected` and per-feature drift scores
- Unit tests cover all three drift scenarios

### Story 4.3: Performance Metrics Monitoring

**Story Points:** 3

**Description:** Track model performance metrics over time (if ground truth labels become available) and serving health metrics.

**Tasks:**

- Create `src/monitoring/performance_tracker.py`
- Track serving metrics from inference logs: p50/p95/p99 latency, error rate, throughput
- Track model metrics: prediction confidence distribution, score variance
- If user feedback data is available (e.g., user clicked recommended card → positive signal), compute proxy accuracy
- Store time-series metrics in GCS as daily JSON snapshots
- Write unit tests

**Acceptance Criteria:**

- Daily performance snapshot generated and stored in GCS
- Latency degradation detected if p95 > 10 seconds
- Metrics queryable by date range

### Story 4.4: Monitoring Airflow DAG

**Story Points:** 5

**Description:** Create a Composer DAG that runs the monitoring pipeline on a daily schedule.

**Tasks:**

- Create `dags/rewardsense_monitoring_pipeline.py`
- Task 1: `collect_inference_data` — calls Story 4.1's collector
- Task 2: `run_drift_detection` — calls Story 4.2's detector
- Task 3: `compute_performance_metrics` — calls Story 4.3's tracker
- Task 4: `evaluate_thresholds` — checks drift and performance against thresholds
- Task 5: `trigger_retrain_if_needed` — if thresholds breached, trigger the model pipeline DAG (see Epic 5)
- Task 6: `send_notification` — send summary to Slack/email regardless of drift status
- Schedule: daily at 6 AM UTC
- Deploy DAG and source modules to Composer bucket

**Acceptance Criteria:**

- DAG appears in Composer UI and runs on schedule
- All tasks complete successfully when inference logs exist
- DAG handles gracefully when insufficient data exists (< 1 day of logs)
- Threshold breach correctly triggers downstream retrain

### Story 4.5: Integration Test — Monitoring Pipeline End-to-End

**Story Points:** 2

**Description:** Verify the monitoring pipeline works end-to-end with real inference logs.

**Tasks:**

- Hit the serving API with 50 synthetic requests to generate inference logs
- Manually trigger the monitoring DAG
- Verify drift report is generated in GCS
- Verify performance metrics snapshot is stored
- Verify notification is sent (Slack or email)

**Acceptance Criteria:**

- Full pipeline runs without errors on real data
- All artifacts (drift report, metrics, notification) produced

---

## Epic 5: Automatic Retraining Trigger

**Owner:** Member B | **Total Points:** 10

### Story 5.1: Retrain Trigger Mechanism

**Story Points:** 5

**Description:** When the monitoring DAG detects drift or performance decay, automatically trigger the existing model pipeline DAG to retrain.

**Tasks:**

- In the monitoring DAG's `trigger_retrain_if_needed` task:
  - Use Airflow's `TriggerDagRunOperator` to trigger `rewardsense_model_pipeline`
  - Pass context: `trigger_reason` (drift/decay), `drift_report_path`, `threshold_values`
- The model pipeline DAG already handles: retrain → validate → if better → push to registry → deploy
- Add a guard: don't trigger retrain if one is already running (check DAG run state)
- Add a guard: max 1 retrain per 24 hours to prevent thrashing
- Log all trigger decisions (triggered, skipped due to guard, no drift detected)

**Acceptance Criteria:**

- Drift detection → model pipeline DAG triggered within 5 minutes
- Guard prevents concurrent or excessive retrains
- Trigger reason logged and passed to the model pipeline run
- If retrained model is worse, existing model stays in Production (this is already handled by Phase 2's validation gate)

### Story 5.2: Notification System

**Story Points:** 3

**Description:** Send notifications via Slack webhook (and optional email) for monitoring events, retraining triggers, and redeployment completions.

**Tasks:**

- Create `src/monitoring/notifier.py`
- Implement Slack webhook notification with structured message:
  - Monitoring summary: drift detected (yes/no), top drifted features, latency metrics
  - Retrain trigger: reason, timestamp, link to drift report
  - Redeployment: new model version, old model version, performance comparison
- Store Slack webhook URL as a Composer environment variable (not in code)
- Add optional email notification via SendGrid or GCP's built-in email
- Write unit tests with mocked webhook

**Acceptance Criteria:**

- Slack messages sent for: daily monitoring summary, retrain trigger, redeployment
- Messages include actionable information (links to reports, model versions)
- Webhook failure doesn't crash the monitoring DAG (fire-and-forget with retry)

### Story 5.3: Integration Test — Full Retrain Loop

**Story Points:** 2

**Description:** Verify the complete closed loop: drift detected → retrain triggered → new model validated → deployed to serving.

**Tasks:**

- Inject synthetic drifted data into inference logs (simulate a distribution shift)
- Trigger the monitoring DAG
- Verify it detects drift and triggers the model pipeline
- Verify the model pipeline runs, trains, validates, and pushes a new model
- Verify the serving service picks up the new model (check `/health` `model_version`)
- Verify Slack notifications sent at each stage

**Acceptance Criteria:**

- Full loop completes within 30 minutes (training is the bottleneck)
- Each stage produces expected artifacts and notifications
- Serving service is running the newly trained model after the loop completes

---

## Epic 6: React Frontend

**Owner:** Member D | **Total Points:** 16

### Story 6.1: Frontend Scaffold & Design System

**Story Points:** 3

**Description:** Set up a React application with Tailwind CSS, routing, and a clean design system for the Expo demo.

**Tasks:**

- Initialize React app with Vite + TypeScript
- Configure Tailwind CSS
- Set up project structure: `src/components/`, `src/pages/`, `src/api/`, `src/types/`
- Create design tokens: color palette (match RewardSense branding), typography, spacing
- Create reusable components: `Card`, `Button`, `Input`, `LoadingSpinner`, `Badge`
- Set up React Router with pages: Home, Recommend, Results, Dashboard
- Create responsive layout shell (header, content area, footer)

**Acceptance Criteria:**

- `npm run dev` starts the app without errors
- All reusable components render correctly in isolation
- Responsive layout works on desktop and tablet (Expo will be on a laptop screen)

### Story 6.2: Recommendation Input Form

**Story Points:** 5

**Description:** Build the main user-facing page where a user enters their spending profile and receives credit card recommendations.

**Tasks:**

- Create `RecommendPage` with input form:
  - Monthly spending by category (groceries, dining, travel, gas, online shopping, etc.) — slider or number input
  - Preferred reward types (cashback, travel points, hotel points, airline miles) — multi-select
  - Annual income range — dropdown
  - Current cards (optional) — multi-select from known card list
- Form validation with helpful error messages
- "Get Recommendations" button triggers API call to `/predict`
- Loading state with animated spinner while waiting for response
- Error handling: API timeout, network error, server error — user-friendly messages
- Write component tests (React Testing Library)

**Acceptance Criteria:**

- Form is intuitive and completable in < 30 seconds (Expo demo constraint)
- Submitting the form calls the inference API and navigates to results
- Invalid inputs are caught client-side before API call
- Loading state is visually clear and engaging

### Story 6.3: Results Display & Explanation Cards

**Story Points:** 5

**Description:** Display the ranked credit card recommendations with scores, visual breakdowns, and LLM-generated explanations.

**Tasks:**

- Create `ResultsPage` that receives API response
- For each recommended card (top 3):
  - Card name and issuer with a styled card component
  - Overall match score as a percentage bar or radial gauge
  - Score breakdown: deterministic score vs. personalization score (stacked bar)
  - LLM explanation displayed in a collapsible "Why this card?" section
  - Key benefits highlighted (cashback %, travel perks, sign-up bonus)
- Add "Try different profile" button to go back to input form
- Add comparison view: side-by-side card comparison
- Animate cards appearing (staggered fade-in for Expo wow factor)
- Write component tests

**Acceptance Criteria:**

- Top 3 cards displayed with all score components and explanations
- Comparison view works for 2-3 cards
- Animations are smooth and not distracting
- Page renders correctly even if LLM explanation is missing (fallback text)

### Story 6.4: Monitoring Dashboard Page

**Story Points:** 3

**Description:** Build a simple dashboard page that displays the latest monitoring metrics and drift status — useful for the Expo demo to show the MLOps story.

**Tasks:**

- Create `DashboardPage` that fetches monitoring data from GCS (via a lightweight API endpoint or directly from a public GCS bucket)
- Display:
  - Current model version and last deployment time
  - Last drift check result: drift detected (yes/no), per-feature drift heatmap
  - Serving metrics: request count, average latency, error rate (last 24h)
  - Retraining history: last 5 retrain events with trigger reason
- Use Recharts for visualizations (already available in the artifact environment)
- Auto-refresh every 60 seconds

**Acceptance Criteria:**

- Dashboard loads and displays real monitoring data
- Drift heatmap clearly shows which features drifted
- Metrics are current (within last 24 hours)
- Page is visually polished for the Expo

---

## Epic 7: Documentation & Replication

**Owner:** Member E | **Total Points:** 10

### Story 7.1: Deployment README

**Story Points:** 5

**Description:** Write comprehensive step-by-step instructions for replicating the entire deployment from a clean machine.

**Tasks:**

- Document prerequisites: GCP account, gcloud CLI, Docker, Node.js, Python 3.11
- **Section 1: GCP Project Setup**
  - Enable APIs (Cloud Run, Artifact Registry, Vertex AI, Cloud Composer, Cloud Storage)
  - Create service account with exact IAM roles listed
  - Configure authentication
- **Section 2: MLflow Server Deployment**
  - Cloud Run deployment command with exact env vars
- **Section 3: Model Training**
  - How to trigger the data pipeline and model pipeline DAGs
  - How to verify a Production model exists in MLflow
- **Section 4: Serving Deployment**
  - Build and push Docker image
  - Deploy to Cloud Run
  - Verify with health check and sample prediction
- **Section 5: Frontend Deployment**
  - Build React app
  - Deploy to Cloud Run (or Firebase Hosting)
- **Section 6: Monitoring Setup**
  - Deploy monitoring DAG
  - Configure Slack webhook
  - Verify monitoring pipeline runs
- Include exact `gcloud` commands, not just descriptions
- Include troubleshooting section for common issues

**Acceptance Criteria:**

- A teammate with GCP access can replicate the full deployment in < 1 hour
- All commands copy-pasteable and tested on a clean machine
- No assumed pre-installations beyond the stated prerequisites

### Story 7.2: Architecture Diagram & Model Card

**Story Points:** 3

**Description:** Create a system architecture diagram and model card for the deployed system.

**Tasks:**

- Architecture diagram showing:
  - User → React Frontend → Cloud Run Serving API → Scoring Engine + ML Model + Gemini LLM
  - Cloud Composer: Data Pipeline → Model Pipeline → Monitoring Pipeline
  - MLflow Registry ↔ Cloud Run (model loading)
  - GCS: inference logs, drift reports, data versions
  - GitHub Actions: CI/CD → Artifact Registry → Cloud Run deploy
  - Monitoring loop: inference logs → drift detection → retrain trigger
- Model card documenting:
  - Model type, training data, feature set, performance metrics
  - Known limitations, ethical considerations, fairness metrics (from Phase 2 bias detection)
  - Intended use case, out-of-scope use cases

**Acceptance Criteria:**

- Architecture diagram is clear, accurate, and presentation-ready
- Model card follows standard format (Google's Model Card template)
- Both included in the README and the Expo presentation

### Story 7.3: Environment Configuration Files

**Story Points:** 2

**Description:** Provide all configuration files needed to replicate the environment.

**Tasks:**

- `Dockerfile.serving` — production serving container
- `docker-compose.serving.yaml` — local development of the full serving stack
- k8s/ or Cloud Run YAML configs (if using declarative deployment)
- `.env.example` — all required environment variables with descriptions
- `requirements-serving.txt` — pinned dependencies for the serving service
- `frontend/package.json` — pinned frontend dependencies
- Terraform/CDK scripts if infrastructure is codified (optional but ideal)

**Acceptance Criteria:**

- All config files are in the repo and documented
- `docker compose up` starts the full local stack (API + frontend + MLflow)
- `.env.example` has every variable with clear descriptions

---

## Epic 8: Video Demo

**Owner:** Member E (with all members for their respective sections) | **Total Points:** 5

### Story 8.1: Record Deployment Demo Video

**Story Points:** 5

**Description:** Record a 5-10 minute video demonstrating the entire deployment process on a fresh environment.

**Tasks:**

- Start from a clean GCP project (or demonstrate setup steps)
- Show: environment setup → deploy infrastructure → deploy serving API → deploy frontend
- Demonstrate hitting the `/predict` endpoint with curl
- Show the React frontend making a live recommendation
- Show the monitoring dashboard with real drift data
- Show a retrain trigger (or simulate one) and the subsequent redeployment
- Show Slack notifications arriving
- Ensure clear audio narration explaining each step
- Keep video between 5-10 minutes

**Acceptance Criteria:**

- Video shows the complete flow from setup to inference to monitoring
- All steps are clearly narrated
- Video quality is high (1080p screen recording, clear audio)
- No pre-existing installations visible (fresh environment demonstrated)

---

## Summary: Story Points by Epic

| Epic | Description | Owner | Stories | Total Points |
|------|-------------|-------|---------|--------------|
| 1 | Serving Infrastructure Setup | A | 1.1–1.4 | 16 |
| 2 | Inference API Application | C | 2.1–2.5 | 21 |
| 3 | Automated CD Pipeline | A | 3.1–3.3 | 13 |
| 4 | Model Monitoring & Drift Detection | B | 4.1–4.5 | 18 |
| 5 | Automatic Retraining Trigger | B | 5.1–5.3 | 10 |
| 6 | React Frontend | D | 6.1–6.4 | 16 |
| 7 | Documentation & Replication | E | 7.1–7.3 | 10 |
| 8 | Video Demo | E | 8.1 | 5 |
| **Total** | | | **27 Stories** | **109 Points** |

---

## Definition of Done (Phase 3)

- Inference API deployed on Cloud Run, serving recommendations in < 10 seconds
- CI/CD pipeline automatically builds, tests, and deploys on merge to main
- New model versions automatically picked up by the serving service
- Monitoring DAG runs daily on Composer, produces drift reports
- Drift detection triggers automatic retraining via the model pipeline DAG
- Retrained models go through the existing validation gate before deployment
- Slack notifications sent for monitoring summaries, retraining, and redeployments
- React frontend allows users to input a profile and view ranked recommendations with explanations
- Monitoring dashboard displays drift status and serving metrics
- Step-by-step README enables full replication on a clean machine in < 1 hour
- Video demo shows the complete flow in 5-10 minutes
- All unit tests passing with > 85% coverage on new code
- Integration tests verify: API end-to-end, monitoring pipeline, retrain loop
