RewardSense: Phase 1 Data Pipeline Implementation Plan
Project Overview
Project Name: RewardSense - Credit Card Recommendation System
Phase: 1 - Data Pipeline Implementation
Cloud Platform: Google Cloud Platform (GCP)
Total Estimated Story Points: 89
Epic 1: Project Environment & Infrastructure Setup
Story 1.1: Initialize Development Environment - done
Story Points: 2
Description: Set up the foundational development environment with all required dependencies, ensuring reproducibility across team members' machines.
Tasks:
Create comprehensive requirements.txt with pinned versions for all Python dependencies (Airflow, DVC, pytest, pandas, Great Expectations, etc.)
Create environment.yml for conda users as an alternative
Set up .env.example template for environment variables
Configure .gitignore for Python, Airflow, DVC, and IDE-specific files
Document Python version requirements (3.9+)
Acceptance Criteria:
Any team member can replicate the environment using provided files
All dependencies install without conflicts
Environment setup documented in README

Story 1.2: Configure GCP Project & Services - done
Story Points: 8
Description: Set up GCP infrastructure required for the data pipeline including Cloud Storage buckets, service accounts, and IAM permissions.
Tasks:
Create GCP project for RewardSense 
Set up Cloud Storage bucket for DVC remote storage (gs://rewardsense-dvc-store)
Create service account with appropriate IAM roles (Storage Object Admin, BigQuery Data Editor)
Generate and securely store service account key
Configure gcloud CLI authentication
Set up Cloud SQL PostgreSQL instance (for production metadata)
Document GCP setup steps for team replication
Acceptance Criteria:
GCP project accessible by all team members
Service account credentials working for DVC operations
Cloud Storage bucket accessible for data versioning

Story 1.3: Initialize DVC with GCP Remote - done
Story Points: 3
Description: Initialize Data Version Control (DVC) in the repository and configure GCP Cloud Storage as the remote storage backend.
Tasks:
Run dvc init in repository root
Configure GCP remote: dvc remote add -d gcs-remote gs://rewardsense-dvc-store
Set up remote authentication with service account
Create .dvc/.gitignore entries
Add initial .dvc files to git tracking
Create dvc.yaml pipeline definition file (skeleton)
Acceptance Criteria:
DVC initialized and configured
Remote storage connection verified with test push/pull
Configuration documented in README

Story 1.4: Set Up Local Airflow Environment - done
Story Points: 5
Description: Configure Apache Airflow for local development using Docker Compose to ensure consistent environments.
Tasks:
Create docker-compose.yaml for Airflow (webserver, scheduler, worker, postgres, redis)
Configure Airflow with LocalExecutor for development
Set up Airflow connections for GCP services
Create Airflow variables for environment-specific configs
Set up volume mounts for DAGs, logs, and plugins directories
Configure Airflow logging to local logs/ directory
Acceptance Criteria:
Airflow accessible at localhost:8080
DAGs auto-discovered from dags/ folder
GCP connections working from Airflow tasks


EPIC 1 DONE:


Epic 2: Data Acquisition Pipeline 
Story 2.1: Build Credit Card Data Scraper Module - done
Story Points: 8
Description: Develop a modular web scraper to collect credit card rewards data from issuer websites and aggregator platforms.
Tasks:
Create src/data_pipeline/scrapers/ module structure
Implement base BaseScraper class with common methods
Build NerdWalletScraper for aggregator data
Build IssuerScraper for direct issuer sites (Chase, Amex, Citi)
Implement rate limiting and retry logic with exponential backoff
Add user-agent rotation and request headers
Create scraper configuration in configs/scraper_config.yaml
Handle pagination and dynamic content loading - the pagination is not for our project but its enable the web scraper to be able to click through next buttons and dynamic content is for fetching data after page loads. 
Acceptance Criteria:
Scrapers collect card data without hitting rate limits
At least 100 credit cards with complete reward structures
Scraper respects robots.txt and site ToS

Story 2.2: Implement Credit Card API Fetcher - done
Story Points: 5
Description: Create API integration module for fetching credit card data from available APIs (Credit Card Bonuses API, etc.).
Tasks:
Create src/data_pipeline/api_fetcher/ module
Implement CreditCardBonusesAPI client
Add API key management via environment variables
Build response parsing and normalization functions
Implement caching layer to reduce API calls
Add error handling for API failures and timeouts
Acceptance Criteria:
API fetcher retrieves current card offers
Response data normalized to common schema
API errors logged and handled gracefully

Story 2.3: Generate Synthetic User Transaction Data - done
Story Points: 8
Description: Build a synthetic data generator for user transactions and profiles that mirrors realistic spending patterns.
Tasks:
Create src/data_pipeline/generators/ module
Implement UserProfileGenerator for ~100 synthetic users
Build TransactionGenerator with realistic spending patterns
Define spending categories aligned with MCC codes
Add temporal patterns (weekly, monthly, seasonal)
Include edge cases (high spenders, minimal use, category specialists)
Ensure reproducibility with seed parameters
Generate redemption_preference data per user
Acceptance Criteria:
100 synthetic users with diverse profiles
Transactions span 12+ months of simulated history
Data generation reproducible with fixed seed
Note: No of users is subject to discussion

Story 2.4: Create Data Download Orchestration Script - done
Story Points: 3
Description: Build a unified script that orchestrates all data acquisition sources with logging and error handling.
Tasks:
Create scripts/download_data.py as main entry point
Implement CLI arguments for selecting data sources
Add progress logging and status reporting
Create data manifest file after successful download
Implement atomic writes to prevent partial data states
Acceptance Criteria:
Single command downloads all required data
Failed downloads don't corrupt existing data
Download status clearly logged

Epic 3: Data Preprocessing Pipeline
Story 3.1: Implement Data Cleaning Module - done 
Story Points: 8
Description: Build comprehensive data cleaning functions for handling missing values, duplicates, and invalid entries.
Tasks:
Create src/data_pipeline/preprocessing/cleaning.py
Implement clean_credit_card_data() function
Handle missing reward rates (impute or flag)
Remove duplicate card entries
Standardize issuer names
Validate annual fee ranges
Implement clean_transaction_data() function
Remove invalid transactions (negative amounts, future dates)
Handle missing merchant categories
Flag suspicious patterns
Add cleaning report generation
Acceptance Criteria:
All cleaning functions are idempotent
Cleaning steps logged with before/after metrics
No data loss without explicit logging

Story 3.2: Build Feature Engineering Module - done
Story Points: 8
Description: Create feature engineering transformations for both credit card and transaction data.
Tasks:
Create src/data_pipeline/preprocessing/feature_engineering.py
Implement credit card features:
Normalized reward rates per category
Net value calculation (rewards - annual fee)
Bonus category flags
Cap utilization potential
Implement transaction features:
Spending by category aggregations
Temporal spending patterns
Merchant-to-category mapping
Implement user profile features:
Point valuation estimates
Redemption preference encoding
Acceptance Criteria:
Feature engineering is deterministic and reproducible
All features documented with descriptions
Features stored in standardized format

Story 3.3: Create Data Transformation Pipeline -Vidya
Story Points: 5
Description: Build the transformation pipeline that applies cleaning and feature engineering in correct order.
Tasks:
Create src/data_pipeline/preprocessing/transform.py
Implement TransformationPipeline class
Define transformation order and dependencies
Add checkpoint saving between major transformations
Implement transformation configuration via YAML
Create transformation logs and audit trail
Acceptance Criteria:
Pipeline runs end-to-end without manual intervention
Intermediate checkpoints allow partial reruns
Transformation config is version controlled

Story 3.4: Implement Data Normalization & Encoding - Akhilesh
Story Points: 5
Description: Create normalization and encoding utilities for preparing data for ML consumption.
Tasks:
Create src/data_pipeline/preprocessing/normalization.py
Implement numerical normalization (StandardScaler, MinMaxScaler)
Build categorical encoding (OneHot, Label, Target encoding)
Store fitted encoders for inference consistency
Create encoding configuration management
Acceptance Criteria:
Encoders persist and reload correctly
Normalization parameters versioned with data
Encoding handles unseen categories gracefully

Epic 4: Testing Framework
Story 4.1: Set Up Testing Infrastructure  - done
Story Points: 3
Description: Configure pytest framework with fixtures, markers, and coverage reporting.
Tasks:
Create tests/ directory structure mirroring src/
Configure pytest.ini with settings and markers
Set up conftest.py with shared fixtures
Configure coverage reporting with pytest-cov
Add test data fixtures in tests/fixtures/
Set up GitHub Actions for CI testing
Acceptance Criteria:
pytest runs all tests with single command
Coverage report generated automatically
CI pipeline runs tests on every PR

Story 4.2: Write Unit Tests for Data Acquisition - done 
Story Points: 5
Description: Create comprehensive unit tests for all data acquisition modules.
Tasks:
Create tests/test_scrapers.py
Test scraper initialization
Test rate limiting logic
Mock HTTP responses for consistent testing
Create tests/test_api_fetcher.py
Test API client methods
Test error handling scenarios
Mock API responses
Create tests/test_generators.py
Test synthetic data reproducibility
Test data distribution properties
Validate generated schema
Acceptance Criteria:
80% code coverage for acquisition modules



All edge cases tested
Tests run in <30 seconds total

Story 4.3: Write Unit Tests for Preprocessing - tests for 3.3 and 3.4 left , pending these two stories
Story Points: 8
Description: Create unit tests for all preprocessing and transformation functions.
Tasks:
Create tests/test_cleaning.py
Test missing value handling
Test duplicate removal
Test invalid data filtering
Test edge cases (empty data, all nulls)
Create tests/test_feature_engineering.py
Test each feature calculation
Test with boundary values
Verify feature determinism
Create tests/test_transformation.py
Test pipeline ordering
Test checkpoint recovery
Test configuration loading
Acceptance Criteria:
80% code coverage for preprocessing



Edge cases documented and tested
Tests are isolated and independent

Story 4.4: Create Integration Tests for Pipeline - Vidya
Story Points: 5
Description: Build integration tests that verify the complete pipeline workflow.
Tasks:
Create tests/integration/test_pipeline_e2e.py
Test full pipeline with sample data
Verify data flows correctly between stages
Test DVC tracking integration
Test Airflow DAG execution
Create integration test fixtures
Acceptance Criteria:
Integration tests run in isolated environment
Pipeline produces expected outputs
Tests clean up after execution

Epic 5: Airflow DAG Implementation
Story 5.1: Create Core DAG Structure 
Story Points: 8
Description: Build the main Airflow DAG that orchestrates the entire data pipeline.
Tasks:
Create dags/rewardsense_data_pipeline.py
Define DAG with appropriate schedule (daily/weekly)
Set up task dependencies reflecting pipeline flow
Configure retries, timeouts, and SLAs
Add DAG documentation and tags
Implement task groups for logical organization
Acceptance Criteria:
DAG visible and parseable in Airflow UI
Task dependencies correctly represent pipeline flow
DAG documentation visible in UI

Story 5.2: Implement Data Acquisition Tasks
Story Points: 5
Description: Create Airflow tasks for all data acquisition operations.
Tasks:
Create scrape_credit_cards task using PythonOperator
Create fetch_api_data task
Create generate_synthetic_data task
Implement task-level error handling
Add XCom for passing metadata between tasks
Configure task-specific timeouts
Acceptance Criteria:
Tasks execute acquisition modules correctly
Failures trigger appropriate alerts
Task logs capture detailed execution info

Story 5.3: Implement Preprocessing Tasks
Story Points: 5
Description: Create Airflow tasks for data preprocessing stages.
Tasks:
Create clean_data task
Create engineer_features task
Create transform_data task
Implement data validation between tasks
Add sensors for data availability checks
Acceptance Criteria:
Preprocessing tasks maintain data integrity
Tasks are idempotent on reruns
Intermediate data accessible for debugging

Story 5.4: Implement DVC Integration in DAG
Story Points: 5
Description: Integrate DVC operations into Airflow for automated data versioning.
Tasks:
Create version_data task using BashOperator/PythonOperator
Implement dvc add for new data artifacts
Implement dvc push to remote storage
Create task for generating DVC lock file
Add git commit for .dvc file changes
Acceptance Criteria:
Data automatically versioned after successful pipeline run
DVC files synced to Git
Remote storage contains versioned data

Story 5.5: Add Monitoring & Alerting Tasks
Story Points: 3
Description: Implement tasks for pipeline monitoring, logging, and alert generation.
Tasks:
Create generate_pipeline_report task
Implement Slack/Email alerting on failures
Add task for logging pipeline metrics
Configure Airflow callbacks for task state changes
Create dashboard task for status updates
Acceptance Criteria:
Alerts sent on task failures
Pipeline report generated on completion
Metrics logged for performance tracking

Epic 6: Data Schema & Validation
Story 6.1: Define Data Schemas
Story Points: 5
Description: Create formal schema definitions for all data artifacts in the pipeline.
Tasks:
Create schemas/ directory structure
Define credit card data schema (JSON Schema/Pydantic)
Define transaction data schema
Define user profile schema
Define feature output schemas
Document schema evolution policy
Acceptance Criteria:
All data artifacts have defined schemas
Schemas are version controlled
Schema documentation auto-generated

Story 6.2: Implement Schema Validation with Great Expectations
Story Points: 8
Description: Set up Great Expectations for automated data quality validation.
Tasks:
Initialize Great Expectations in project
Create expectation suites for each data source:
Credit card expectations (valid rates, required fields)
Transaction expectations (positive amounts, valid dates)
User expectations (valid IDs, preference ranges)
Set up data context and checkpoints
Configure validation results storage
Create data docs generation
Acceptance Criteria:
Validation runs automatically in pipeline
Failed expectations logged with details
Data docs accessible for review

Story 6.3: Generate Data Statistics & Profiles
Story Points: 5
Description: Implement automated data profiling and statistics generation.
Tasks:
Create src/data_pipeline/profiling/ module
Implement pandas-profiling integration
Create custom statistics for domain-specific metrics
Generate distribution plots for key features
Store historical statistics for comparison
Integrate with Airflow for scheduled profiling
Acceptance Criteria:
Data profiles generated after each pipeline run
Statistics stored for trend analysis
Profiles accessible via reports

Epic 7: Anomaly Detection & Alerts
Story 7.1: Implement Data Anomaly Detection
Story Points: 8
Description: Build anomaly detection mechanisms for identifying data quality issues.
Tasks:
Create src/data_pipeline/anomaly_detection/ module
Implement missing value ratio detection
Build outlier detection for numerical fields (IQR, Z-score)
Create schema violation detection
Implement data drift detection for distributions
Add custom anomaly rules for domain logic
Acceptance Criteria:
Anomalies detected and logged automatically
Configurable thresholds for each check
Historical anomaly tracking

Story 7.2: Build Alert Generation System
Story Points: 5
Description: Create alerting infrastructure for notifying team of detected issues.
Tasks:
Create src/data_pipeline/alerts/ module
Implement Slack webhook integration
Implement email alerting (SendGrid/SMTP)
Create alert severity levels (INFO, WARNING, CRITICAL)
Build alert aggregation to prevent spam
Add alert configuration via YAML
Acceptance Criteria:
Alerts sent within 5 minutes of detection
Alert channels configurable
Alert history logged for review

Story 7.3: Integrate Anomaly Detection into DAG
Story Points: 3
Description: Add anomaly detection tasks to the Airflow DAG workflow.
Tasks:
Create detect_anomalies task
Create send_alerts task
Configure task dependencies (run after data load)
Add short-circuit logic for critical anomalies
Implement anomaly-aware downstream processing
Acceptance Criteria:
Anomaly detection runs on each pipeline execution
Critical anomalies halt pipeline progression
All anomalies logged regardless of severity

Epic 8: Data Bias Detection & Mitigation
Story 8.1: Implement Data Slicing Module
Story Points: 8
Description: Build data slicing capabilities for bias analysis across demographic and categorical features.
Tasks:
Create src/data_pipeline/bias_detection/ module
Implement DataSlicer class for flexible slicing
Define slicing dimensions:
User demographics (age groups, location)
Spending categories
Card types
Transaction amounts (quintiles)
Create slice performance metrics calculator
Implement fairness metrics (demographic parity, equalized odds)
Acceptance Criteria:
Data can be sliced by any categorical feature
Slice statistics computed efficiently
Fairness metrics calculated per slice

Story 8.2: Integrate Bias Detection Tools
Story Points: 5
Description: Integrate established bias detection libraries into the pipeline.
Tasks:
Integrate Fairlearn for fairness assessment
Set up TensorFlow Model Analysis (TFMA) for slice analysis
Create unified bias detection interface
Implement bias detection reports
Add configurable bias thresholds
Acceptance Criteria:
Multiple bias detection tools available
Reports identify biased slices clearly
Thresholds configurable per use case

Story 8.3: Implement Bias Mitigation Strategies
Story Points: 8
Description: Create mitigation mechanisms for detected biases in the data.
Tasks:
Implement resampling strategies (oversampling, undersampling)
Build SMOTE integration for synthetic oversampling
Create fairness-aware feature weighting
Implement threshold adjustment recommendations
Document mitigation trade-offs
Create before/after bias comparison reports
Acceptance Criteria:
At least 3 mitigation strategies available
Mitigation impact measurable
Trade-offs documented for each strategy

Story 8.4: Document Bias Analysis Process
Story Points: 3
Description: Create comprehensive documentation for bias detection and mitigation processes.
Tasks:
Document bias detection methodology
Create bias mitigation decision flowchart
Write guide for interpreting bias reports
Document trade-offs and recommendations
Add bias analysis section to README
Acceptance Criteria:
Documentation enables independent bias analysis
Decision process clearly documented
Trade-offs explicitly stated

Epic 9: Pipeline Optimization & Performance
Story 9.1: Implement Pipeline Performance Monitoring
Story Points: 5
Description: Add instrumentation to track pipeline performance and identify bottlenecks.
Tasks:
Add timing instrumentation to all tasks
Create performance logging module
Integrate with Airflow Gantt chart analysis
Build performance dashboard
Set up performance regression alerts
Acceptance Criteria:
Task execution times tracked
Gantt chart available for analysis
Performance trends visible

Story 9.2: Optimize Slow Pipeline Stages
Story Points: 5
Description: Identify and optimize the slowest stages of the pipeline.
Tasks:
Analyze Gantt chart for bottlenecks
Implement parallel processing where applicable
Add data chunking for large datasets
Optimize database queries and I/O
Implement caching for repeated computations
Acceptance Criteria:
Pipeline execution time reduced by 20%+
Parallelization documented
No functionality regressions

Epic 10: Documentation & Reproducibility
Story 10.1: Create Comprehensive README
Story Points: 5
Description: Write detailed README with setup instructions, architecture overview, and usage guides.
Tasks:
Write project overview and objectives
Document folder structure
Create environment setup instructions
Write step-by-step pipeline execution guide
Add troubleshooting section
Include contribution guidelines
Add architecture diagram
Acceptance Criteria:
New team member can set up in <30 minutes
All commands documented
Architecture clearly explained

Story 10.2: Document Data Pipeline Architecture
Story Points: 3
Description: Create technical documentation for the data pipeline architecture.
Tasks:
Create pipeline flow diagram
Document each component's responsibility
Write data flow documentation
Document error handling strategies
Create API/module documentation
Acceptance Criteria:
Architecture diagrams in docs/
Component interactions documented
Technical decisions explained

Story 10.3: Create Reproducibility Guide
Story Points: 3
Description: Write guide ensuring anyone can reproduce the pipeline on their machine.
Tasks:
Document exact version requirements
Create step-by-step reproduction checklist
Document known platform-specific issues
Create verification steps
Add seed/random state documentation
Acceptance Criteria:
Pipeline reproducible on fresh machine
All dependencies explicitly versioned
Verification steps pass consistently

Story 10.4: Generate Code Documentation
Story Points: 3
Description: Ensure all code is properly documented with docstrings and type hints.
Tasks:
Add docstrings to all public functions
Add type hints throughout codebase
Set up Sphinx for API documentation
Generate HTML documentation
Add inline comments for complex logic
Acceptance Criteria:
All public APIs documented
Type hints pass mypy checks
Generated docs hosted/accessible

