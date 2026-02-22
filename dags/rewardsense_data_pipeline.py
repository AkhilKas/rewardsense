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
    4. Reporting   — Generate pipeline report, log metrics, send alerts

Task Groups:
    ingestion/      Parallel data acquisition from multiple sources
    preprocessing/  Sequential cleaning → features → transform
    versioning/     DVC add + push (placeholder for Story 5.4)
    reporting/      Report generation, metrics logging, and alerting

Notes:
    - Task callables use deferred imports (import inside function body)
      to keep DAG parsing fast and avoid import-time failures.
    - Story 5.1 defines the DAG structure with placeholder task bodies.
      Stories 5.2 and 5.3 will wire in real implementations.
    - Story 5.5 implements monitoring, alerting, and callbacks.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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
│                                                  │
│  generate_pipeline_report                        │
│         │                                        │
│         ├──► log_pipeline_metrics                │
│         └──► send_pipeline_alerts                │
│                                                  │
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
- Alerting config: `config/alerting_config.yaml`

### Contacts
- **Owner**: RewardSense Team
"""


# =============================================================================
# Callbacks (deferred imports to keep DAG parsing lightweight)
# =============================================================================


def _on_task_failure(context):
    """Route task failures to AlertDispatcher (CRITICAL)."""
    from data_pipeline.monitoring.callbacks import on_task_failure_callback

    on_task_failure_callback(context)


def _on_dag_success(context):
    """Send a summary alert when the full DAG succeeds."""
    from data_pipeline.monitoring.callbacks import on_dag_success_callback

    on_dag_success_callback(context)


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
    "on_failure_callback": _on_task_failure,
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
    import os
    from data_pipeline.scrapers.nerdwallet_scraper import NerdWalletScraper

    logger = logging.getLogger("airflow.task")
    logger.info("🔍 Scraping NerdWallet for credit card data...")

    # Initialize scraper (using default categories for full coverage)
    scraper = NerdWalletScraper()
    
    try:
        cards = scraper.scrape_all_cards()
        logger.info(f"Successfully scraped {len(cards)} cards from NerdWallet.")
        
        # Save output to GCS processed directory
        import json
        output_dir = REPO_ROOT / "data" / "processed" / "current" / "offers"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "nerdwallet.json"
        with open(output_path, "w") as f:
            json.dump(cards, f, indent=2)
        logger.info(f"Saved NerdWallet output to {output_path}")
        
        return {"source": "nerdwallet", "cards_found": len(cards), "status": "success"}
    except Exception as e:
        logger.error(f"Failed to scrape NerdWallet: {e}")
        raise


def _scrape_issuers(**context):
    """Scrape credit card data from issuer websites."""
    import logging
    from data_pipeline.scrapers.issuer_scrapers import ChaseScraper, AmexScraper

    logger = logging.getLogger("airflow.task")
    logger.info("🔍 Scraping issuers (Chase, Amex)...")

    results = {}
    total_cards = 0
    scrapers = [ChaseScraper(), AmexScraper()]
    
    for scraper in scrapers:
        try:
            cards = scraper.scrape_all_cards()
            source_name = scraper.get_source_name()
            results[source_name] = len(cards)
            total_cards += len(cards)
            logger.info(f"Scraped {len(cards)} cards from {source_name}")
        except Exception as e:
            logger.error(f"Failed to scrape {scraper.get_source_name()}: {e}")
            # Continue to next issuer even if one fails
            continue

    # Save to GCS if we found anything
    if total_cards > 0:
        import json
        output_dir = REPO_ROOT / "data" / "processed" / "current" / "offers"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "issuers.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2) # simplified for now
        logger.info(f"Saved Issuers output to {output_path}")

    return {
        "source": "issuers",
        "issuers_scraped": list(results.keys()),
        "total_cards": total_cards,
        "results": results,
        "status": "success",
    }


def _fetch_api_data(**context):
    """Fetch credit card data from the CreditCardBonuses API."""
    import logging
    from data_pipeline.api_fetcher import CreditCardBonusesClient

    logger = logging.getLogger("airflow.task")
    logger.info("🌐 Fetching data from CreditCardBonuses API...")

    try:
        client = CreditCardBonusesClient()
        # Log mode for debugging
        logger.info(f"API Client mode: {client.mode}")
        
        offers = client.fetch_normalized_offers()
        logger.info(f"Successfully fetched {len(offers)} offers from API.")
        
        # Save output to GCS (matches transform.yaml expected path)
        import json
        output_dir = REPO_ROOT / "data" / "processed" / "current" / "offers"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "creditcardbonuses_offers.json"
        
        # Convert Pydantic models to dicts for JSON serialization
        offers_dict = [o.model_dump() for o in offers]
        with open(output_path, "w") as f:
            json.dump(offers_dict, f, indent=2)
        logger.info(f"Saved API output to {output_path}")
        
        return {
            "source": "creditcardbonuses_api",
            "offers_found": len(offers),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Failed to fetch API data: {type(e).__name__}: {e}")
        # Log more details if it's an upstream error
        from data_pipeline.api_fetcher import CreditCardBonusesUpstreamError
        if isinstance(e, CreditCardBonusesUpstreamError):
            logger.error("Context: This is an upstream error from the API/Export source.")
        raise


def _generate_synthetic_data(**context):
    """Generate synthetic user profiles and transaction data."""
    import logging
    from data_pipeline.generators import UserProfileGenerator, TransactionGenerator

    logger = logging.getLogger("airflow.task")
    logger.info("🏭 Generating synthetic user & transaction data...")

    try:
        import gc
        # Reduce count to 100 to stay within worker memory limits for this story
        user_gen = UserProfileGenerator(num_users=100, seed=42)
        users = user_gen.generate()
        logger.info(f"Generated {len(users)} synthetic users.")
        
        # Clean up UserProfileGenerator overhead before large transaction gen
        gc.collect()
        
        txn_gen = TransactionGenerator(seed=42)
        transactions = txn_gen.generate(users)
        logger.info(f"Generated {len(transactions)} synthetic transactions.")
        
        # Save to GCS (matches transform.yaml expected paths)
        output_dir = REPO_ROOT / "data" / "processed" / "current" / "synthetic"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        user_path = output_dir / "user_profiles.csv"
        txn_path = output_dir / "transactions.csv"
        
        users.to_csv(user_path, index=False)
        transactions.to_csv(txn_path, index=False)
        logger.info(f"Saved synthetic data to {output_dir}")
        
        # Capture counts before clean up
        u_count = len(users)
        t_count = len(transactions)
        
        # Free memory immediately
        del transactions
        del users
        gc.collect()
        
        return {
            "users_generated": u_count, 
            "transactions_generated": t_count, 
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {type(e).__name__}: {e}")
        raise


def _merge_card_data(**context):
    """Merge and deduplicate card data from all ingestion sources."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("🔀 Merging card data from all ingestion sources...")
    
    # Extract metrics from upstream tasks using XCom
    ti = context['ti']
    nerdwallet_metrics = ti.xcom_pull(task_ids='ingestion.scrape_nerdwallet')
    issuers_metrics = ti.xcom_pull(task_ids='ingestion.scrape_issuers')
    api_metrics = ti.xcom_pull(task_ids='ingestion.fetch_api_data')
    
    nw_count = nerdwallet_metrics.get("cards_found", 0) if nerdwallet_metrics else 0
    issuer_count = issuers_metrics.get("total_cards", 0) if issuers_metrics else 0
    api_count = api_metrics.get("offers_found", 0) if api_metrics else 0
    
    total_found = nw_count + issuer_count + api_count
    
    # Story 5.2 implementation connects the scraper outputs to the prep layer
    # For now, we return the counts merged to confirm the pipeline execution logic flowed properly
    logger.info(f"Merge metrics: NW={nw_count}, Issuers={issuer_count}, API={api_count}")

    # Write manifest file to signal ingestion completion to the preprocessing phase
    import json
    manifest_path = REPO_ROOT / "data" / "processed" / "current" / "manifest_latest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_merged_cards": total_found,
        "sources": {
            "nerdwallet": nw_count,
            "issuers": issuer_count,
            "api": api_count
        }
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    return {"total_merged_cards": total_found, "duplicates_removed": 0, "status": "success"}


def _clean_data(**context):
    """Run data cleaning on all datasets."""
    import logging
    from data_pipeline.preprocessing.transform import TransformationPipeline

    logger = logging.getLogger("airflow.task")
    logger.info("🧹 Cleaning credit card, transaction, and user data...")

    # Run the load and clean steps of the transformation pipeline
    pipeline = TransformationPipeline(config_path=Path("config/transform.yaml"))
    # The clean step will load from raw and write clean checkpoints
    cards_df, txns_df, users_df, load_report = pipeline._step_load()
    clean_cards, clean_txns, clean_users, clean_report = pipeline._step_clean(
        cards_df, txns_df, users_df
    )

    return {"status": "success", "report": clean_report}


def _engineer_features(**context):
    """Run feature engineering on cleaned datasets."""
    import logging
    from data_pipeline.preprocessing.transform import TransformationPipeline

    logger = logging.getLogger("airflow.task")
    logger.info("⚙️ Engineering features for cards and transactions...")

    pipeline = TransformationPipeline(config_path=Path("config/transform.yaml"))

    # Instead of recalculating, we load the checkpoints from the previous step
    # if checkpoints are enabled and available or continue from step_clean
    step_name_clean = "02_cleaned"
    ckpt_dir = pipeline._step_ckpt_dir(step_name_clean)
    if pipeline.checkpoints_enabled and pipeline._checkpoint_exists(step_name_clean):
        clean_cards = (
            pipeline._load_df(ckpt_dir / "credit_cards_clean.csv")
            if (ckpt_dir / "credit_cards_clean.csv").exists()
            else None
        )
        clean_txns = (
            pipeline._load_df(ckpt_dir / "transactions_clean.csv")
            if (ckpt_dir / "transactions_clean.csv").exists()
            else None
        )
        clean_users = (
            pipeline._load_df(ckpt_dir / "users_clean.csv")
            if (ckpt_dir / "users_clean.csv").exists()
            else None
        )
    else:
        # Fallback to run previous steps if checkpoints aren't found
        cards_df, txns_df, users_df, _ = pipeline._step_load()
        clean_cards, clean_txns, clean_users, _ = pipeline._step_clean(
            cards_df, txns_df, users_df
        )

    cards_f, txns_f, users_f, features_report = pipeline._step_features(
        clean_cards, clean_txns, clean_users
    )

    return {"status": "success", "report": features_report}


def _run_transform_pipeline(**context):
    """Run the full transformation pipeline."""
    import logging
    from data_pipeline.preprocessing.transform import TransformationPipeline

    logger = logging.getLogger("airflow.task")
    logger.info("🔄 Running TransformationPipeline write outputs...")

    pipeline = TransformationPipeline(config_path=Path("config/transform.yaml"))

    step_name_features = "03_features"
    ckpt_dir = pipeline._step_ckpt_dir(step_name_features)
    if pipeline.checkpoints_enabled and pipeline._checkpoint_exists(step_name_features):
        cards_f = (
            pipeline._load_df(ckpt_dir / "credit_cards_features.csv")
            if (ckpt_dir / "credit_cards_features.csv").exists()
            else None
        )
        txns_f = (
            pipeline._load_df(ckpt_dir / "transactions_features.csv")
            if (ckpt_dir / "transactions_features.csv").exists()
            else None
        )
        users_f = (
            pipeline._load_df(ckpt_dir / "users_features.csv")
            if (ckpt_dir / "users_features.csv").exists()
            else None
        )
    else:
        # Just run the whole pipeline
        return pipeline.run()

    outputs = pipeline._write_final_outputs(cards_f, txns_f, users_f)

    return {"status": "success", "outputs": outputs}


# =============================================================================
# Reporting / Monitoring task callables  (Story 5.5)
# =============================================================================


def _generate_pipeline_report(**context):
    """Generate a summary report of the pipeline run."""
    from data_pipeline.monitoring.pipeline_report import PipelineReportGenerator

    generator = PipelineReportGenerator()
    return generator.generate(context)


def _log_pipeline_metrics(**context):
    """Log timing, record counts, and error metrics for the pipeline run."""
    from data_pipeline.monitoring.metrics import PipelineMetricsLogger

    logger = PipelineMetricsLogger()
    return logger.log_metrics(context)


def _send_pipeline_alerts(**context):
    """Send end-of-pipeline alerts via configured channels."""
    import logging

    from data_pipeline.monitoring.alerting import AlertDispatcher, Severity

    log = logging.getLogger("airflow.task")
    ti = context.get("ti")
    dag_run = context.get("dag_run")

    # Pull the report summary from upstream task
    report_summary = ti.xcom_pull(task_ids="reporting.generate_pipeline_report") or {}

    dag_id = dag_run.dag_id if dag_run else "unknown"
    run_id = str(dag_run.run_id) if dag_run else "unknown"
    duration = report_summary.get("total_duration_sec", "N/A")

    message = (
        f"Pipeline *{dag_id}* run completed.\n"
        f"Run ID: {run_id}\n"
        f"Duration: {duration}s\n"
        f"Report: {report_summary.get('report_path', 'N/A')}"
    )

    dispatcher = AlertDispatcher()
    results = dispatcher.dispatch(
        message=message,
        severity=Severity.INFO,
        subject=f"Pipeline Summary: {dag_id}",
    )
    log.info("Alert dispatch results: %s", results)

    return {"alerts_sent": results, "status": "completed"}


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
    on_success_callback=_on_dag_success,
) as dag:

    # ── Start sentinel ──────────────────────────────────────────────────
    pipeline_start = EmptyOperator(task_id="pipeline_start")

    # ── Task Group: Ingestion ───────────────────────────────────────────
    with TaskGroup(
        "ingestion", tooltip="Data acquisition from all sources"
    ) as ingestion_group:

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
    with TaskGroup(
        "preprocessing",
        tooltip="Data cleaning, feature engineering, and transformation",
    ) as preprocessing_group:

        # Let's ensure the manifest file which signifies completion of ingestion is written
        from airflow.sensors.python import PythonSensor
        import os

        def _check_manifest_exists():
            manifest_path = (
                REPO_ROOT / "data" / "processed" / "current" / "manifest_latest.json"
            )
            return os.path.exists(manifest_path)

        check_raw_data_ready = PythonSensor(
            task_id="check_raw_data_ready",
            python_callable=_check_manifest_exists,
            poke_interval=60,
            timeout=60 * 30,  # 30 mins
            mode="reschedule",
            doc_md="Wait for the data ingestion manifest file to appear before proceeding.",
        )

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

        check_raw_data_ready >> clean >> features >> transform

    # ── Task Group: Versioning ──────────────────────────────────────────
    with TaskGroup(
        "versioning", tooltip="Data versioning with DVC"
    ) as versioning_group:

        version_raw_data = BashOperator(
            task_id="version_raw_data",
            bash_command="cd {{ var.value.get('repo_root', '/opt/airflow') }} && dvc add data/processed/current/synthetic data/processed/current/offers || true",
            doc_md="Version raw ingestion data using DVC.",
        )

        version_processed_data = BashOperator(
            task_id="version_processed_data",
            bash_command="cd {{ var.value.get('repo_root', '/opt/airflow') }} && dvc add data/processed/current/transformed || true",
            doc_md="Version transformed and feature-engineered data using DVC.",
        )

        push_to_remote = BashOperator(
            task_id="push_to_remote",
            bash_command="cd {{ var.value.get('repo_root', '/opt/airflow') }} && dvc push",
            doc_md="Push DVC tracked files to the configured Google Cloud Storage remote.",
        )

        commit_dvc_files = BashOperator(
            task_id="commit_dvc_files",
            bash_command=(
                "cd {{ var.value.get('repo_root', '/opt/airflow') }} && "
                "git add data/processed/current/*.dvc dvc.lock .gitignore && "
                "git diff-index --quiet HEAD || git commit -m 'chore(data): auto-update DVC tracking files [skip ci]'"
            ),
            doc_md="Commit updated DVC tracking files to Git to maintain version history.",
        )

        version_raw_data >> version_processed_data >> push_to_remote >> commit_dvc_files

    # ── Task Group: Reporting & Monitoring ──────────────────────────────
    with TaskGroup(
        "reporting", tooltip="Pipeline report, metrics, and alerting"
    ) as reporting_group:

        report = PythonOperator(
            task_id="generate_pipeline_report",
            python_callable=_generate_pipeline_report,
            doc_md="Generate summary report with card counts, cleaning stats, and timing.",
        )

        metrics = PythonOperator(
            task_id="log_pipeline_metrics",
            python_callable=_log_pipeline_metrics,
            doc_md="Log pipeline metrics (durations, record counts, errors) to data/metrics/.",
        )

        alerts = PythonOperator(
            task_id="send_pipeline_alerts",
            python_callable=_send_pipeline_alerts,
            doc_md="Dispatch end-of-pipeline alerts to Slack and/or Email.",
        )

        # Report first, then metrics and alerts run in parallel
        report >> [metrics, alerts]

    # ── End sentinel ────────────────────────────────────────────────────
    pipeline_end = EmptyOperator(
        task_id="pipeline_end",
        trigger_rule="none_failed",
    )

    # ── Cross-group dependencies ────────────────────────────────────────
    # start → ingestion → preprocessing → versioning → reporting → end
    pipeline_start >> ingestion_group >> preprocessing_group
    preprocessing_group >> versioning_group >> reporting_group >> pipeline_end
