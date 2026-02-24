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
import os
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

from data_pipeline.monitoring.performance import (
    PipelinePerformanceMonitor,
    timed_python_task,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def get_data_root() -> Path:
    """Return stable data root for Composer or local runs."""
    if Path("/home/airflow/gcs").exists():
        # Composer persisted data path (bucket root mount).
        return Path("/home/airflow/gcs/data/processed/current")
    return REPO_ROOT / "data" / "processed" / "current"


def _write_csv_chunked(df, output_path: Path, chunk_size: int) -> None:
    """Write a DataFrame in chunks to reduce peak memory pressure."""
    total_rows = len(df)
    if total_rows == 0:
        df.to_csv(output_path, index=False)
        return
    for start in range(0, total_rows, chunk_size):
        stop = min(start + chunk_size, total_rows)
        mode = "w" if start == 0 else "a"
        header = start == 0
        df.iloc[start:stop].to_csv(output_path, mode=mode, header=header, index=False)


def _is_synthetic_cache_valid(
    meta_path: Path, users_path: Path, txns_path: Path
) -> bool:
    """Check whether synthetic outputs can be reused for this run."""
    if not (meta_path.exists() and users_path.exists() and txns_path.exists()):
        return False
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return False

    expected_users = int(os.getenv("SYNTHETIC_USER_COUNT", "100"))
    expected_seed = int(os.getenv("SYNTHETIC_SEED", "42"))
    return (
        payload.get("num_users") == expected_users
        and payload.get("seed") == expected_seed
        and payload.get("status") == "success"
    )


def _resolve_transform_config_path() -> Path:
    candidates = [
        REPO_ROOT / "config" / "transform_config.yaml",
        REPO_ROOT / "dags" / "config" / "transform_config.yaml",
        REPO_ROOT / "config" / "transform.yaml",
        REPO_ROOT / "dags" / "config" / "transform.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


def _on_task_success(context):
    """Log task-level success timing for all operators."""
    from data_pipeline.monitoring.callbacks import on_task_success_callback

    on_task_success_callback(context)


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
    "on_success_callback": _on_task_success,
}


# =============================================================================
# Task callables (placeholder implementations for Story 5.1)
#
# Each callable uses deferred imports to keep DAG parsing lightweight.
# Stories 5.2 and 5.3 will replace these with real logic.
# =============================================================================


@timed_python_task("ingestion.scrape_nerdwallet")
def _scrape_nerdwallet(**context):
    """Scrape credit card data from NerdWallet."""
    import logging
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

        output_dir = get_data_root() / "offers"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to: {output_dir}")
        logger.info(f"Path exists after mkdir: {output_dir.exists()}")

        output_path = output_dir / "nerdwallet.json"
        with open(output_path, "w") as f:
            json.dump(cards, f, indent=2)
        logger.info(f"Saved NerdWallet output to {output_path}")

        return {"source": "nerdwallet", "cards_found": len(cards), "status": "success"}
    except Exception as e:
        logger.error(f"Failed to scrape NerdWallet: {e}")
        raise


@timed_python_task("ingestion.scrape_issuers")
def _scrape_issuers(**context):
    """Scrape credit card data from issuer websites."""
    import logging
    from data_pipeline.scrapers.issuer_scrapers import ChaseScraper, AmexScraper

    logger = logging.getLogger("airflow.task")
    logger.info("🔍 Scraping issuer sites...")

    results = {}
    all_cards = []
    total_cards = 0
    skipped_scrapers = []

    # Cloud Composer workers do not provide a stable Chrome runtime for Selenium.
    # Run Selenium scrapers only in non-Composer environments.
    is_composer = os.path.exists("/home/airflow/gcs")

    scrapers = [ChaseScraper()]
    if is_composer:
        skipped_scrapers.append("American Express (Selenium disabled on Composer)")
        logger.warning(
            "Skipping AmexScraper in Composer environment "
            "(Selenium/Chrome not available)."
        )
    else:
        scrapers.append(AmexScraper())

    # Run issuer scrapers concurrently to reduce end-to-end ingestion latency.
    with ThreadPoolExecutor(max_workers=min(4, len(scrapers))) as executor:
        future_map = {
            executor.submit(scraper.scrape_all_cards): scraper for scraper in scrapers
        }
        for future in as_completed(future_map):
            scraper = future_map[future]
            source_name = scraper.get_source_name()
            try:
                cards = future.result()
                results[source_name] = len(cards)
                total_cards += len(cards)
                for card in cards:
                    if not isinstance(card, dict):
                        continue
                    enriched = dict(card)
                    # Preserve origin for downstream traceability.
                    enriched.setdefault("source", source_name)
                    enriched["issuer_source"] = source_name
                    all_cards.append(enriched)
                logger.info(f"Scraped {len(cards)} cards from {source_name}")
            except Exception as e:
                logger.error(f"Failed to scrape {source_name}: {e}")
                continue

    # Save to GCS if we found anything
    if total_cards > 0:
        import json

        output_dir = get_data_root() / "offers"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to: {output_dir}")
        logger.info(f"Path exists after mkdir: {output_dir.exists()}")
        output_path = output_dir / "issuers.json"
        with open(output_path, "w") as f:
            json.dump(all_cards, f, indent=2)
        logger.info(f"Saved Issuers output to {output_path}")

    return {
        "source": "issuers",
        "issuers_scraped": list(results.keys()),
        "total_cards": total_cards,
        "issuer_counts": results,
        "cards_written": len(all_cards),
        "skipped_scrapers": skipped_scrapers,
        "status": "success",
    }


@timed_python_task("ingestion.fetch_api_data")
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

        output_dir = get_data_root() / "offers"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to: {output_dir}")
        logger.info(f"Path exists after mkdir: {output_dir.exists()}")
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
            logger.error(
                "Context: This is an upstream error from the API/Export source."
            )
        raise


@timed_python_task("ingestion.generate_synthetic_data")
def _generate_synthetic_data(**context):
    """Generate synthetic user profiles and transaction data."""
    import logging
    from data_pipeline.generators.user_profile_generator import UserProfileGenerator
    from data_pipeline.generators.transaction_generator import TransactionGenerator

    logger = logging.getLogger("airflow.task")
    logger.info("🏭 Generating synthetic user & transaction data...")

    try:
        user_count = int(os.getenv("SYNTHETIC_USER_COUNT", "100"))
        seed = int(os.getenv("SYNTHETIC_SEED", "42"))
        chunk_size = int(os.getenv("SYNTHETIC_TXN_CHUNK_SIZE", "25000"))

        output_dir = get_data_root() / "synthetic"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to: {output_dir}")
        logger.info(f"Path exists after mkdir: {output_dir.exists()}")

        user_path = output_dir / "user_profiles.csv"
        txn_path = output_dir / "transactions.csv"
        meta_path = output_dir / "synthetic_meta.json"

        if _is_synthetic_cache_valid(meta_path, user_path, txn_path):
            cache_meta = json.loads(meta_path.read_text())
            logger.info("Using cached synthetic data from %s", output_dir)
            return {
                "users_generated": cache_meta.get("users_generated", 0),
                "transactions_generated": cache_meta.get("transactions_generated", 0),
                "status": "success",
                "cache_hit": True,
            }

        user_gen = UserProfileGenerator(num_users=user_count, seed=seed)
        users = user_gen.generate()
        logger.info(f"Generated {len(users)} synthetic users.")

        # Clean up UserProfileGenerator overhead before large transaction gen
        gc.collect()

        txn_gen = TransactionGenerator(seed=seed)
        transactions = txn_gen.generate(users)
        logger.info(f"Generated {len(transactions)} synthetic transactions.")

        users.to_csv(user_path, index=False)
        _write_csv_chunked(transactions, txn_path, chunk_size=chunk_size)
        meta_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "generated_at": datetime.utcnow().isoformat(),
                    "num_users": user_count,
                    "seed": seed,
                    "chunk_size": chunk_size,
                    "users_generated": int(len(users)),
                    "transactions_generated": int(len(transactions)),
                },
                indent=2,
            )
        )
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
            "status": "success",
            "cache_hit": False,
        }
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {type(e).__name__}: {e}")
        raise


@timed_python_task("ingestion.merge_card_data")
def _merge_card_data(**context):
    """Merge and deduplicate card data from all ingestion sources."""
    import logging

    logger = logging.getLogger("airflow.task")
    logger.info("🔀 Merging card data from all ingestion sources...")
    data_root = get_data_root()
    offers_dir = data_root / "offers"
    offers_dir.mkdir(parents=True, exist_ok=True)

    def _load_cards(path: Path, source_name: str) -> list[dict]:
        if not path.exists():
            logger.warning("Source file missing for merge: %s", path)
            return []
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("offers", [])
        if not isinstance(payload, list):
            logger.warning(
                "Unexpected payload shape in %s (%s). Skipping.",
                path,
                type(payload).__name__,
            )
            return []

        cards = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            enriched = dict(item)
            enriched.setdefault("source", source_name)
            cards.append(enriched)
        return cards

    nerdwallet_cards = _load_cards(offers_dir / "nerdwallet.json", "nerdwallet")
    issuer_cards = _load_cards(offers_dir / "issuers.json", "issuers")
    api_cards = _load_cards(
        offers_dir / "creditcardbonuses_offers.json", "creditcardbonuses_api"
    )

    def _norm(value):
        return " ".join(str(value or "").strip().lower().split())

    def _dedupe_key(card: dict) -> tuple[str, str]:
        card_name = card.get("name") or card.get("card_name") or card.get("title") or ""
        issuer = (
            card.get("issuer")
            or card.get("issuer_name")
            or card.get("source")
            or card.get("issuer_source")
            or ""
        )
        return (_norm(card_name), _norm(issuer))

    merged_cards = []
    seen_keys = set()
    for card in nerdwallet_cards + issuer_cards + api_cards:
        key = _dedupe_key(card)
        if not key[0]:
            # Keep cards missing a name, but avoid duplicate empty-name payloads.
            key = (f"unnamed:{len(merged_cards)}", key[1])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_cards.append(card)

    merged_path = offers_dir / "merged_cards.json"
    merged_path.write_text(json.dumps(merged_cards, indent=2))
    logger.info("Merged %s cards into %s", len(merged_cards), merged_path)

    # Write manifest file to signal ingestion completion to the preprocessing phase
    manifest_path = data_root / "manifest_latest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing manifest to: {manifest_path.parent}")
    logger.info(f"Path exists after mkdir: {manifest_path.parent.exists()}")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_merged_cards": len(merged_cards),
        "sources": {
            "nerdwallet": len(nerdwallet_cards),
            "issuers": len(issuer_cards),
            "api": len(api_cards),
        },
        "files": {
            "nerdwallet": "offers/nerdwallet.json",
            "issuers": "offers/issuers.json",
            "api": "offers/creditcardbonuses_offers.json",
            "merged_cards": "offers/merged_cards.json",
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    return {
        "total_merged_cards": len(merged_cards),
        "duplicates_removed": (
            len(nerdwallet_cards) + len(issuer_cards) + len(api_cards)
        )
        - len(merged_cards),
        "merged_file": "offers/merged_cards.json",
        "status": "success",
    }


@timed_python_task("preprocessing.clean_data")
def _clean_data(**context):
    """Run data cleaning on all datasets."""
    import logging
    import traceback

    logger = logging.getLogger("airflow.task")
    logger.info("🎬 Starting _clean_data task...")

    try:
        from data_pipeline.preprocessing.transform import TransformationPipeline

        config_path = _resolve_transform_config_path()
        logger.info(f"Using config at: {config_path}")

        # Run the load and clean steps of the transformation pipeline
        pipeline = TransformationPipeline(config_path=config_path)
        # The clean step will load from raw and write clean checkpoints
        cards_df, txns_df, users_df, load_report = pipeline._step_load()
        clean_cards, clean_txns, clean_users, clean_report = pipeline._step_clean(
            cards_df, txns_df, users_df
        )

        return {"status": "success", "report": clean_report}
    except Exception as e:
        logger.error(f"❌ FATAL ERROR in _clean_data: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise


@timed_python_task("preprocessing.engineer_features")
def _engineer_features(**context):
    """Run feature engineering on cleaned datasets."""
    import logging
    from data_pipeline.preprocessing.transform import TransformationPipeline

    logger = logging.getLogger("airflow.task")
    logger.info("⚙️ Engineering features for cards and transactions...")

    config_path = _resolve_transform_config_path()
    logger.info(f"Using config at: {config_path}")
    pipeline = TransformationPipeline(config_path=config_path)

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


@timed_python_task("preprocessing.run_transform_pipeline")
def _run_transform_pipeline(**context):
    """Run the full transformation pipeline."""
    import logging
    from data_pipeline.preprocessing.transform import TransformationPipeline

    logger = logging.getLogger("airflow.task")
    logger.info("🔄 Running TransformationPipeline write outputs...")

    config_path = _resolve_transform_config_path()
    logger.info(f"Using config at: {config_path}")
    pipeline = TransformationPipeline(config_path=config_path)

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


@timed_python_task("reporting.generate_pipeline_report")
def _generate_pipeline_report(**context):
    """Generate a summary report of the pipeline run."""
    from data_pipeline.monitoring.pipeline_report import PipelineReportGenerator

    generator = PipelineReportGenerator()
    return generator.generate(context)


@timed_python_task("reporting.log_pipeline_metrics")
def _log_pipeline_metrics(**context):
    """Log timing, record counts, and error metrics for the pipeline run."""
    from data_pipeline.monitoring.metrics import PipelineMetricsLogger

    logger = PipelineMetricsLogger()
    return logger.log_metrics(context)


@timed_python_task("reporting.send_pipeline_alerts")
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


@timed_python_task("reporting.generate_performance_dashboard")
def _generate_performance_dashboard(**context):
    """Build a performance dashboard with bottlenecks and task trends."""
    monitor = PipelinePerformanceMonitor()
    return monitor.generate_dashboard(context)


@timed_python_task("reporting.check_performance_regression")
def _check_performance_regression(**context):
    """Detect performance regressions and emit warnings via alerting channels."""
    import logging

    from data_pipeline.monitoring.alerting import AlertDispatcher, Severity

    log = logging.getLogger("airflow.task")
    monitor = PipelinePerformanceMonitor()
    result = monitor.detect_regression(context)

    if not result.get("regression_detected"):
        log.info("[PERF] No regressions detected for run %s", result.get("run_id"))
        return {"status": "ok", **result}

    task_rows = result.get("task_regressions", [])
    bottleneck_line = (
        ", ".join(
            f"{row['task_id']} (+{round(row['regression_ratio'] * 100, 1)}%)"
            for row in task_rows[:5]
        )
        or "none"
    )
    message = (
        "Performance regression detected.\n"
        f"DAG: {result.get('dag_id')}\n"
        f"Run: {result.get('run_id')}\n"
        f"Run regression ratio: {result.get('run_regression_ratio')}\n"
        f"Task regressions: {bottleneck_line}"
    )
    dispatch = AlertDispatcher().dispatch(
        message=message,
        severity=Severity.WARNING,
        subject=f"Performance regression: {result.get('dag_id')}",
    )
    log.warning("[PERF] Regression alert dispatched: %s", dispatch)
    return {"status": "warning", "alerts_sent": dispatch, **result}


@timed_python_task("versioning.trigger_github_dvc_commit")
def _trigger_github_dvc_commit(**context):
    """Trigger GitHub Actions workflow to commit DVC tracking files."""
    import logging

    import requests
    from airflow.models import Variable

    logger = logging.getLogger("airflow.task")
    logger.info("🚀 Triggering GitHub Actions DVC commit workflow...")

    github_token = os.getenv("GITHUB_TOKEN") or Variable.get(
        "github_token", default_var=None
    )
    github_repo = os.getenv("GITHUB_REPO") or Variable.get(
        "github_repo", default_var="avadharj/rewardsense"
    )

    if not github_token:
        logger.warning("GITHUB_TOKEN not configured; skipping trigger.")
        return {"status": "skipped", "reason": "GITHUB_TOKEN not configured"}

    dag_run = context.get("dag_run")
    run_id = str(dag_run.run_id) if dag_run else "unknown"

    url = f"https://api.github.com/repos/{github_repo}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "event_type": "dvc-commit",
        "client_payload": {
            "dag_run_id": run_id,
            "triggered_at": datetime.utcnow().isoformat(),
        },
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 204:
            logger.info("GitHub dispatch accepted for %s", github_repo)
            return {
                "status": "triggered",
                "dag_run_id": run_id,
                "github_repo": github_repo,
            }
        logger.error("GitHub API error %s: %s", response.status_code, response.text)
        return {
            "status": "failed",
            "error": f"HTTP {response.status_code}: {response.text}",
            "github_repo": github_repo,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to trigger GitHub workflow: %s", exc)
        return {
            "status": "failed",
            "error": str(exc),
            "github_repo": github_repo,
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

        def _check_raw_data_ready():
            """Verify all ingestion outputs exist before preprocessing."""
            data_root = get_data_root()
            required_files = [
                data_root / "manifest_latest.json",
                data_root / "synthetic" / "user_profiles.csv",
                data_root / "synthetic" / "transactions.csv",
                data_root / "offers" / "merged_cards.json",
            ]
            return all(path.exists() for path in required_files)

        check_raw_data_ready = PythonSensor(
            task_id="check_raw_data_ready",
            python_callable=_check_raw_data_ready,
            poke_interval=60,
            timeout=60 * 30,  # 30 mins
            mode="reschedule",
            doc_md="Wait for manifest, merged cards, and synthetic files before preprocessing.",
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
            bash_command=(
                "cd {{ var.value.get('repo_root', dag.folder) }} && "
                "printf 'stages: {}\\n' > dvc.yaml && "
                "dvc config core.no_scm true --local && "
                "if [ -d /home/airflow/gcs ]; then "
                "  DATA_ROOT=/home/airflow/gcs/data/processed/current; "
                "else "
                "  DATA_ROOT=data/processed/current; "
                "fi && "
                "dvc add ${DATA_ROOT}/synthetic ${DATA_ROOT}/offers || true"
            ),
            doc_md="Version raw ingestion data using DVC.",
        )

        version_processed_data = BashOperator(
            task_id="version_processed_data",
            bash_command=(
                "cd {{ var.value.get('repo_root', dag.folder) }} && "
                "printf 'stages: {}\\n' > dvc.yaml && "
                "dvc config core.no_scm true --local && "
                "if [ -d /home/airflow/gcs ]; then "
                "  DATA_ROOT=/home/airflow/gcs/data/processed/current; "
                "else "
                "  DATA_ROOT=data/processed/current; "
                "fi && "
                "dvc add ${DATA_ROOT}/transformed || true"
            ),
            doc_md="Version transformed and feature-engineered data using DVC.",
        )

        push_to_remote = BashOperator(
            task_id="push_to_remote",
            bash_command=(
                "cd {{ var.value.get('repo_root', dag.folder) }} && "
                "dvc push || echo 'DVC push completed or nothing to push'"
            ),
            doc_md="Push DVC tracked files to the configured Google Cloud Storage remote.",
        )

        trigger_dvc_commit = PythonOperator(
            task_id="trigger_github_dvc_commit",
            python_callable=_trigger_github_dvc_commit,
            doc_md="Trigger GitHub Actions workflow to commit .dvc files to Git.",
        )

        (
            version_raw_data
            >> version_processed_data
            >> push_to_remote
            >> trigger_dvc_commit
        )

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

        performance_dashboard = PythonOperator(
            task_id="generate_performance_dashboard",
            python_callable=_generate_performance_dashboard,
            doc_md="Build performance dashboard with bottlenecks, trends, and Gantt spans.",
        )

        performance_regression = PythonOperator(
            task_id="check_performance_regression",
            python_callable=_check_performance_regression,
            doc_md="Detect performance regressions against recent run history.",
        )

        # Report first, then metrics + alerts in parallel.
        # Dashboard/regression analysis happens after metrics are persisted.
        report >> [metrics, alerts]
        metrics >> performance_dashboard >> performance_regression

    # ── End sentinel ────────────────────────────────────────────────────
    pipeline_end = EmptyOperator(
        task_id="pipeline_end",
        trigger_rule="none_failed",
    )

    # ── Cross-group dependencies ────────────────────────────────────────
    # start → ingestion → preprocessing → versioning → reporting → end
    pipeline_start >> ingestion_group >> preprocessing_group
    preprocessing_group >> versioning_group >> reporting_group >> pipeline_end
