"""
RewardSense - Main Data Pipeline DAG

Orchestrates the end-to-end data pipeline for the RewardSense
credit card recommendation system.

Schedule: Weekly (Sunday 6:00 AM UTC)
Owner: rewardsense

Pipeline stages:
    1. Ingestion   — Scrape card data, fetch API, generate synthetic data
    2. Preprocessing — Clean, feature-engineer, and transform datasets
    3. Versioning  — Version artifacts with DVC
    4. Reporting   — Generate pipeline execution report

Task Groups:
    ingestion/      Parallel data acquisition from multiple sources
    preprocessing/  Sequential cleaning → features → transform
    versioning/     DVC add + push (placeholder for Story 5.4)
    reporting/      Summary report generation (placeholder for Story 5.5)

Notes:
    - Task callables use deferred imports (import inside function body)
      to keep DAG parsing fast and avoid import-time failures.
    - Story 5.1 defines the DAG structure with placeholder task bodies.
      Stories 5.2 and 5.3 will wire in real implementations.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup


# =============================================================================
# DAG documentation (rendered as markdown in the Airflow UI)
# =============================================================================

DAG_DOC_MD = """
## RewardSense Data Pipeline

### Overview
Weekly pipeline that ingests credit card data from multiple sources,
generates synthetic user/transaction data, cleans and transforms
everything, then versions the output with DVC.

### Pipeline Flow
```
┌─────────────────── Ingestion ───────────────────┐
│                                                  │
│  scrape_nerdwallet ──┐                           │
│  scrape_issuers ─────┼──► merge_card_data        │
│  fetch_api_data ─────┘                           │
│  generate_synthetic_data (parallel)              │
│                                                  │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────── Preprocessing ───────────────────┐
│                                                  │
│  clean_data ──► engineer_features                │
│                      │                           │
│                      ▼                           │
│              run_transform_pipeline              │
│                                                  │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────── Versioning ─────────────────────┐
│  version_with_dvc                                │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────── Reporting ─────────────────────┐
│  generate_pipeline_report                        │
└──────────────────────────────────────────────────┘
```

### Data Sources
| Source | Type | Module |
|--------|------|--------|
| NerdWallet | Web scrape | `scrapers.NerdWalletScraper` |
| Chase, Amex, Citi, Capital One, Discover | Web scrape | `scrapers.issuer_scrapers` |
| CreditCardBonuses API | REST API | `api_fetcher.CreditCardBonusesClient` |
| Synthetic users & transactions | Generator | `generators.*` |

### Configuration
- Scraper config: `config/scraper_config.yaml`
- Transform config: `config/transform.yaml`
- Generator config: `config/generator_config.yaml`

### Contacts
- **Owner**: RewardSense Team
"""

# =============================================================================
# Default args
# =============================================================================

default_args = {
    "owner": "rewardsense",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
    "sla": timedelta(hours=3),
}


# =============================================================================
# Task callables (placeholder implementations for Story 5.1)
#
# Each callable uses deferred imports to keep DAG parsing lightweight.
# Stories 5.2 and 5.3 will replace these with real logic.
# =============================================================================


def _scrape_nerdwallet(**context):
    """Scrape credit card data from NerdWallet."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("🔍 [PLACEHOLDER] Scraping NerdWallet for credit card data...")

    # Story 5.2 will implement:
    #   from data_pipeline.scrapers import NerdWalletScraper
    #   scraper = NerdWalletScraper()
    #   cards = scraper.scrape_all_cards()

    return {"source": "nerdwallet", "cards_found": 0, "status": "placeholder"}


def _scrape_issuers(**context):
    """Scrape credit card data from issuer websites."""
    import logging

    logger = logging.getLogger("airflow.task")
    issuers = ["chase", "amex", "citi", "capital_one", "discover"]
    logger.info(f"🔍 [PLACEHOLDER] Scraping {len(issuers)} issuers: {issuers}")

    # Story 5.2 will implement:
    #   from data_pipeline.scrapers import ChaseScraper, AmexScraper, ...
    #   results = {}
    #   for scraper_cls in [ChaseScraper, AmexScraper, ...]:
    #       scraper = scraper_cls()
    #       results[scraper.get_source_name()] = scraper.scrape_all_cards()

    return {
        "source": "issuers",
        "issuers_scraped": issuers,
        "total_cards": 0,
        "status": "placeholder",
    }


def _fetch_api_data(**context):
    """Fetch credit card data from the CreditCardBonuses API."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("🌐 [PLACEHOLDER] Fetching data from CreditCardBonuses API...")

    # Story 5.2 will implement:
    #   from data_pipeline.api_fetcher import CreditCardBonusesClient
    #   client = CreditCardBonusesClient()
    #   offers = client.fetch_normalized_offers()

    return {"source": "creditcardbonuses_api", "offers_found": 0, "status": "placeholder"}


def _generate_synthetic_data(**context):
    """Generate synthetic user profiles and transaction data."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("🏭 [PLACEHOLDER] Generating synthetic user & transaction data...")

    # Story 5.2 will implement:
    #   from data_pipeline.generators import UserProfileGenerator, TransactionGenerator
    #   user_gen = UserProfileGenerator(seed=42)
    #   users = user_gen.generate(n=1000)
    #   txn_gen = TransactionGenerator(seed=42)
    #   transactions = txn_gen.generate(users)

    return {"users_generated": 0, "transactions_generated": 0, "status": "placeholder"}


def _merge_card_data(**context):
    """Merge and deduplicate card data from all ingestion sources."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("🔀 [PLACEHOLDER] Merging card data from all ingestion sources...")

    # Story 5.2 will pull XCom from upstream tasks and merge

    return {"total_merged_cards": 0, "duplicates_removed": 0, "status": "placeholder"}


def _clean_data(**context):
    """Run data cleaning on all datasets."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("🧹 [PLACEHOLDER] Cleaning credit card, transaction, and user data...")

    # Story 5.3 will implement:
    #   from data_pipeline.preprocessing.cleaning import clean_all_data
    #   results = clean_all_data(credit_cards_df, transactions_df, users_df)

    return {"datasets_cleaned": 3, "status": "placeholder"}


def _engineer_features(**context):
    """Run feature engineering on cleaned datasets."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("⚙️ [PLACEHOLDER] Engineering features for cards and transactions...")

    # Story 5.3 will implement:
    #   from data_pipeline.preprocessing.feature_engineering import (
    #       CreditCardFeatureEngineer, TransactionFeatureEngineer
    #   )

    return {"features_engineered": 0, "status": "placeholder"}


def _run_transform_pipeline(**context):
    """Run the full transformation pipeline."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("🔄 [PLACEHOLDER] Running TransformationPipeline...")

    # Story 5.3 will implement:
    #   from data_pipeline.preprocessing.transform import TransformationPipeline
    #   pipeline = TransformationPipeline(config_path=Path("config/transform.yaml"))
    #   pipeline.run()

    return {"transform_status": "placeholder"}


def _generate_pipeline_report(**context):
    """Generate a summary report of the pipeline run."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("📊 [PLACEHOLDER] Generating pipeline execution report...")

    # Story 5.5 will implement full reporting

    dag_run = context.get("dag_run")
    return {
        "dag_run_id": str(dag_run.run_id) if dag_run else "unknown",
        "status": "placeholder",
    }


# =============================================================================
# DAG definition
# =============================================================================

with DAG(
    dag_id="rewardsense_data_pipeline",
    default_args=default_args,
    description="Weekly pipeline: ingest → preprocess → version → report for credit card recommendation data",
    doc_md=DAG_DOC_MD,
    schedule="0 6 * * 0",  # Every Sunday at 06:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rewardsense", "data-pipeline", "weekly"],
) as dag:

    # ── Start sentinel ──────────────────────────────────────────────────
    pipeline_start = EmptyOperator(task_id="pipeline_start")

    # ── Task Group: Ingestion ───────────────────────────────────────────
    with TaskGroup("ingestion", tooltip="Data acquisition from all sources") as ingestion_group:

        scrape_nerdwallet = PythonOperator(
            task_id="scrape_nerdwallet",
            python_callable=_scrape_nerdwallet,
            doc_md="Scrape credit card listings from NerdWallet.",
        )

        scrape_issuers = PythonOperator(
            task_id="scrape_issuers",
            python_callable=_scrape_issuers,
            doc_md="Scrape credit card data from Chase, Amex, Citi, Capital One, and Discover.",
        )

        fetch_api = PythonOperator(
            task_id="fetch_api_data",
            python_callable=_fetch_api_data,
            doc_md="Fetch normalized credit card offers from CreditCardBonuses API.",
        )

        generate_synthetic = PythonOperator(
            task_id="generate_synthetic_data",
            python_callable=_generate_synthetic_data,
            doc_md="Generate synthetic user profiles and transaction histories.",
        )

        merge_cards = PythonOperator(
            task_id="merge_card_data",
            python_callable=_merge_card_data,
            doc_md="Merge and deduplicate card data from scrapers and API.",
        )

        # Scraping and API run in parallel, then converge at merge
        [scrape_nerdwallet, scrape_issuers, fetch_api] >> merge_cards

        # Synthetic data generation runs in parallel (no merge dependency)
        # Both merge_cards and generate_synthetic feed into preprocessing

    # ── Task Group: Preprocessing ───────────────────────────────────────
    with TaskGroup("preprocessing", tooltip="Data cleaning, feature engineering, and transformation") as preprocessing_group:

        clean = PythonOperator(
            task_id="clean_data",
            python_callable=_clean_data,
            doc_md="Clean and validate credit card, transaction, and user profile data.",
        )

        features = PythonOperator(
            task_id="engineer_features",
            python_callable=_engineer_features,
            doc_md="Engineer ML features: reward rates, spending patterns, net card value.",
        )

        transform = PythonOperator(
            task_id="run_transform_pipeline",
            python_callable=_run_transform_pipeline,
            doc_md="Run end-to-end TransformationPipeline with checkpointing and audit.",
        )

        clean >> features >> transform

    # ── Task Group: Versioning ──────────────────────────────────────────
    with TaskGroup("versioning", tooltip="Data versioning with DVC") as versioning_group:

        version_dvc = BashOperator(
            task_id="version_with_dvc",
            bash_command=(
                'echo "[PLACEHOLDER] DVC versioning — Story 5.4 will implement:" && '
                'echo "  dvc add data/processed/current/transformed/" && '
                'echo "  dvc push"'
            ),
            doc_md="Version processed data artifacts with DVC and push to remote.",
        )

    # ── Task Group: Reporting ───────────────────────────────────────────
    with TaskGroup("reporting", tooltip="Pipeline execution summary") as reporting_group:

        report = PythonOperator(
            task_id="generate_pipeline_report",
            python_callable=_generate_pipeline_report,
            doc_md="Generate summary report with card counts, cleaning stats, and timing.",
        )

    # ── End sentinel ────────────────────────────────────────────────────
    pipeline_end = EmptyOperator(
        task_id="pipeline_end",
        trigger_rule="none_failed",
    )

    # ── Cross-group dependencies ────────────────────────────────────────
    # start → ingestion → preprocessing → versioning → reporting → end
    pipeline_start >> ingestion_group >> preprocessing_group
    preprocessing_group >> versioning_group >> reporting_group >> pipeline_end
