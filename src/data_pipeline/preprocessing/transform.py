# src/data_pipeline/preprocessing/transform.py
"""
RewardSense - Data Transformation Pipeline

Story 3.3: Create Data Transformation Pipeline

Builds an end-to-end transformation pipeline that:
- Loads raw/staged artifacts from data/processed/current (created by scripts/download_data.py)
- Applies cleaning (Story 3.1) in the correct order
- Applies feature engineering deterministically
- Saves intermediate checkpoints for partial reruns
- Uses YAML-driven configuration (version controlled)
- Writes transformation logs + audit trail JSON

Design goals:
- Idempotent: safe to re-run; checkpoints enable fast incremental runs
- Auditable: every run produces an audit trail (config hash + input/output hashes + counts)
- Minimal assumptions: works even if some datasets are missing (e.g., no transactions)

Expected input layout (from download_data.py):
data/processed/current/
  offers/creditcardbonuses_offers.json   (normalized offers)
  offers/issuer_<issuer>_offers.json     (scraped dict offers)
  offers/nerdwallet_offers.json          (scraped dict offers)
  synthetic/user_profiles.csv
  synthetic/user_cards.csv
  synthetic/transactions.csv
  manifest_latest.json

Outputs (default):
data/processed/current/transformed/<run_id>/
  checkpoints/
    01_loaded/...
    02_cleaned/...
    03_features/...
  final/
    credit_cards_features.csv
    transactions_features.csv
    users_features.csv
  audit/
    audit.json
    step_reports.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from data_pipeline.preprocessing.cleaning import CleaningConfig, clean_all_data
from data_pipeline.preprocessing.feature_engineering import engineer_all_features

logger = logging.getLogger(__name__)


# =============================================================================
# Utilities
# =============================================================================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp_{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(dest))


def atomic_write_text(dest: Path, text: str) -> None:
    atomic_write_bytes(dest, (text + "\n").encode("utf-8"))


def json_sanitize(obj: Any) -> Any:
    """
    Convert objects that json.dumps can't handle into serializable equivalents.
    - set -> sorted list
    - Path -> str
    - numpy scalars -> python scalars
    - pandas Timestamp/Timedelta -> str
    - dict/list/tuple recurse
    """
    if obj is None:
        return None

    # pandas NA
    if obj is pd.NA:
        return None

    # sets
    if isinstance(obj, set):
        return sorted(list(obj))

    # tuples/lists
    if isinstance(obj, tuple):
        return [json_sanitize(x) for x in obj]
    if isinstance(obj, list):
        return [json_sanitize(x) for x in obj]

    # dicts
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}

    # filesystem paths
    if isinstance(obj, Path):
        return str(obj)

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # pandas time types
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)

    # numpy arrays -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def atomic_write_json(dest: Path, obj: Any) -> None:
    safe_obj = json_sanitize(obj)
    atomic_write_bytes(
        dest,
        (json.dumps(safe_obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def safe_read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(json_sanitize(obj), sort_keys=True, indent=2, ensure_ascii=False)


def df_hash(df: pd.DataFrame) -> str:
    """
    Robust dataframe hash for audit purposes.

    Handles unhashable object cells (lists/dicts/sets/ndarrays) by converting them to
    stable JSON strings before hashing.

    Notes:
    - This is for audit/change detection, not cryptographic integrity.
    - Uses stable column order + row order + JSON normalization for objects.
    """
    if df is None:
        return None  # type: ignore

    df2 = df.copy()
    df2 = df2.sort_index(axis=1).reset_index(drop=True)

    def _is_missing_scalar(x: Any) -> bool:
        # Avoid pd.isna on non-scalars (can return boolean arrays)
        if x is None:
            return True
        if x is pd.NA:
            return True
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return True
        if isinstance(x, np.generic):
            try:
                return bool(np.isnan(x))
            except Exception:
                return False
        return False

    def _canon_obj(x: Any) -> Any:
        # Safe missing check (never ambiguous)
        if _is_missing_scalar(x):
            return None

        # normalize numpy arrays -> lists
        if isinstance(x, np.ndarray):
            x = x.tolist()

        # normalize pandas Timestamp/Timedelta to strings
        if isinstance(x, (pd.Timestamp, pd.Timedelta)):
            return str(x)

        # normalize iterables / mappings to stable JSON strings
        if isinstance(x, (dict, list, tuple, set)):
            try:
                if isinstance(x, set):
                    x = sorted(list(x))
                elif isinstance(x, tuple):
                    x = list(x)
                return json.dumps(
                    json_sanitize(x),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except Exception:
                return str(x)

        return x

    # Convert object columns safely
    obj_cols = df2.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df2[c] = df2[c].map(_canon_obj)

    # Hash rows using pandas and then sha256 the bytes
    h = pd.util.hash_pandas_object(df2, index=True).to_numpy().tobytes()
    return sha256_bytes(h)


# =============================================================================
# Config
# =============================================================================


DEFAULT_CONFIG_YAML = """\
version: 1
pipeline:
  # Where download_data.py committed artifacts
  input_root: data/processed/current

  # Where transform outputs go (under input_root/transformed/<run_id>)
  output_subdir: transformed

  # If true, resume from existing checkpoints when available
  resume: true

  # If true, always recompute steps even if checkpoints exist
  force_recompute: false

datasets:
  credit_cards:
    enabled: true
    # Sources to load. "api" reads creditcardbonuses_offers.json.
    # Scraped issuer/nerdwallet files are optional and currently loaded
    # as raw dicts; by default we DO NOT merge them into cards until Story 3.4+.
    load_api_offers: true
    load_issuer_offers: false
    load_nerdwallet_offers: false

    # Which offer file to load for API offers
    api_offers_file: offers/creditcardbonuses_offers.json

    # If true, attempt to flatten offers into a tabular DataFrame for cleaning/FE.
    # For CreditCardBonuses offers, this should be true.
    flatten_api_offers: true

    annual_spending: 25000

  transactions:
    enabled: true
    file: synthetic/transactions.csv

  users:
    enabled: true
    file: synthetic/user_profiles.csv

cleaning:
  # thresholds consistent with cleaning.CleaningConfig
  max_annual_fee: 1000.0
  min_annual_fee: 0.0
  min_transaction_amount: 0.0
  suspicious_amount_threshold: 10000.0

  validate_mcc: true

checkpoints:
  enabled: true
  # Save as CSV for dataframes and JSON for reports/audit
  format: csv

logging:
  level: INFO
"""


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cleaning_config(cfg: Dict[str, Any]) -> CleaningConfig:
    c = cfg.get("cleaning", {}) or {}
    return CleaningConfig(
        max_annual_fee=float(c.get("max_annual_fee", 1000.0)),
        min_annual_fee=float(c.get("min_annual_fee", 0.0)),
        min_transaction_amount=float(c.get("min_transaction_amount", 0.0)),
        suspicious_amount_threshold=float(
            c.get("suspicious_amount_threshold", 10000.0)
        ),
        # valid_mcc_codes + issuer_aliases use defaults from dataclass
    )


# =============================================================================
# Audit structures
# =============================================================================


@dataclass
class StepAudit:
    name: str
    started_at: str
    finished_at: str
    duration_s: float
    used_checkpoint: bool
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    report_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None


@dataclass
class RunAudit:
    run_id: str
    started_at: str
    finished_at: str
    duration_s: float
    config_path: str
    config_sha256: str
    input_root: str
    output_root: str
    steps: Dict[str, StepAudit]


# =============================================================================
# Pipeline implementation
# =============================================================================


class TransformationPipeline:
    """
    Orchestrates:
    1) Load raw data artifacts
    2) Clean datasets (credit cards, transactions, users)
    3) Feature engineering
    4) Checkpointing + audit logs

    Usage:
        pipeline = TransformationPipeline(config_path=Path("configs/transform.yaml"))
        outputs = pipeline.run()
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cfg = load_yaml_config(config_path)
        self._setup_logging()

        p = self.cfg.get("pipeline", {}) or {}
        raw_input = p.get("input_root", "data/processed/current")
        if Path(raw_input).is_absolute():
            self.input_root = Path(raw_input).resolve()
        else:
            # Resolve relative to config_path's parent's parent (standard sibling structure)
            candidate = (self.config_path.parent.parent / raw_input).resolve()
            
            # If not found and in a 'dags' folder (Composer-like), try one level above dags
            if not candidate.exists() and "dags" in self.config_path.parts:
                try:
                    parts = list(self.config_path.parts)
                    # Find the last occurrence of 'dags'
                    dags_idx = len(parts) - 1 - parts[::-1].index("dags")
                    root = Path(*parts[:dags_idx])
                    if root.parts: # ensure not empty
                         alt_candidate = (root / raw_input).resolve()
                         if alt_candidate.exists():
                             candidate = alt_candidate
                except (ValueError, IndexError):
                    pass
            
            self.input_root = candidate
        self.output_subdir = p.get("output_subdir", "transformed")
        self.resume = bool(p.get("resume", True))
        self.force_recompute = bool(p.get("force_recompute", False))

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = (
            self.input_root / self.output_subdir / self.run_id
        ).resolve()

        self.checkpoints_enabled = bool(
            (self.cfg.get("checkpoints", {}) or {}).get("enabled", True)
        )
        self.checkpoint_format = (self.cfg.get("checkpoints", {}) or {}).get(
            "format", "csv"
        )

        # audit dirs
        self.audit_dir = self.output_root / "audit"
        self.final_dir = self.output_root / "final"
        self.ckpt_dir = self.output_root / "checkpoints"

        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoints_enabled:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # cleaning config object
        self.cleaning_config = build_cleaning_config(self.cfg)

        # config hash for audit/versioning
        self.config_sha256 = sha256_bytes(stable_json_dumps(self.cfg).encode("utf-8"))

        logger.info("TransformationPipeline initialized")
        logger.info("Input root : %s", self.input_root)
        logger.info("Output root: %s", self.output_root)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _setup_logging(self) -> None:
        log_cfg = self.cfg.get("logging", {}) or {}
        level_str = str(log_cfg.get("level", "INFO")).upper()
        level = getattr(logging, level_str, logging.INFO)

        root = logging.getLogger()
        root.setLevel(level)

        # Avoid duplicate handlers if called multiple times
        if not root.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            ch.setLevel(level)
            root.addHandler(ch)

        # File handler for transform logs
        log_path = self.input_root / "logs" / "transform.log"
        log_path = log_path.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # attach file handler to module logger
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        # prevent duplicate file handlers
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == str(log_path)
            for h in logger.handlers
        ):
            logger.addHandler(fh)

    # -------------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------------

    def _step_ckpt_dir(self, step_name: str) -> Path:
        return self.ckpt_dir / step_name

    def _checkpoint_exists(self, step_name: str, sentinel: str = "_DONE") -> bool:
        d = self._step_ckpt_dir(step_name)
        return d.exists() and (d / sentinel).exists()

    def _mark_checkpoint_done(self, step_name: str) -> None:
        d = self._step_ckpt_dir(step_name)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(d / "_DONE", f"done_at={utc_now_iso()}")

    def _save_df(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_format.lower() == "csv":
            atomic_write_text(path, df.to_csv(index=False))
        else:
            df.to_parquet(path, index=False)

    def _load_df(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported dataframe checkpoint format: {path}")

    # -------------------------------------------------------------------------
    # Data loading (from download_data.py outputs)
    # -------------------------------------------------------------------------

    def _load_credit_cards(self) -> Optional[pd.DataFrame]:
        ds = self.cfg.get("datasets", {}).get("credit_cards", {}) or {}
        if not ds.get("enabled", True):
            return None

        cards_df: Optional[pd.DataFrame] = None

        if ds.get("load_api_offers", True):
            offers_file = self.input_root / ds.get(
                "api_offers_file", "offers/creditcardbonuses_offers.json"
            )
            if offers_file.exists():
                payload = safe_read_json(offers_file)
                offers = payload.get("offers", [])
                if ds.get("flatten_api_offers", True):
                    cards_df = pd.json_normalize(offers)
                else:
                    cards_df = pd.DataFrame({"offer": offers})
            else:
                logger.warning("Credit card API offers file missing: %s", offers_file)

        return cards_df

    def _load_transactions(self) -> Optional[pd.DataFrame]:
        ds = self.cfg.get("datasets", {}).get("transactions", {}) or {}
        if not ds.get("enabled", True):
            return None
        path = self.input_root / ds.get("file", "synthetic/transactions.csv")
        if not path.exists():
            logger.warning("Transactions file missing: %s", path)
            return None
        return pd.read_csv(path)

    def _load_users(self) -> Optional[pd.DataFrame]:
        ds = self.cfg.get("datasets", {}).get("users", {}) or {}
        if not ds.get("enabled", True):
            return None
        path = self.input_root / ds.get("file", "synthetic/user_profiles.csv")
        if not path.exists():
            logger.warning("Users file missing: %s", path)
            return None
        return pd.read_csv(path)

    # -------------------------------------------------------------------------
    # Steps
    # -------------------------------------------------------------------------

    def _step_load(
        self,
    ) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Dict[str, Any],
    ]:
        step_name = "01_loaded"
        ckpt_dir = self._step_ckpt_dir(step_name)

        if (
            self.checkpoints_enabled
            and self.resume
            and not self.force_recompute
            and self._checkpoint_exists(step_name)
        ):
            logger.info("[LOAD] Using checkpoint: %s", ckpt_dir)
            cards = (
                self._load_df(ckpt_dir / "credit_cards_raw.csv")
                if (ckpt_dir / "credit_cards_raw.csv").exists()
                else None
            )
            txns = (
                self._load_df(ckpt_dir / "transactions_raw.csv")
                if (ckpt_dir / "transactions_raw.csv").exists()
                else None
            )
            users = (
                self._load_df(ckpt_dir / "users_raw.csv")
                if (ckpt_dir / "users_raw.csv").exists()
                else None
            )
            report = safe_read_json(ckpt_dir / "load_report.json")
            return cards, txns, users, report

        t0 = time.time()
        cards_df = self._load_credit_cards()
        txns_df = self._load_transactions()
        users_df = self._load_users()

        report = {
            "cards_rows": int(len(cards_df)) if cards_df is not None else 0,
            "txns_rows": int(len(txns_df)) if txns_df is not None else 0,
            "users_rows": int(len(users_df)) if users_df is not None else 0,
            "cards_cols": list(cards_df.columns) if cards_df is not None else [],
            "txns_cols": list(txns_df.columns) if txns_df is not None else [],
            "users_cols": list(users_df.columns) if users_df is not None else [],
        }

        if self.checkpoints_enabled:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if cards_df is not None:
                self._save_df(cards_df, ckpt_dir / "credit_cards_raw.csv")
            if txns_df is not None:
                self._save_df(txns_df, ckpt_dir / "transactions_raw.csv")
            if users_df is not None:
                self._save_df(users_df, ckpt_dir / "users_raw.csv")
            atomic_write_json(ckpt_dir / "load_report.json", report)
            self._mark_checkpoint_done(step_name)

        logger.info("[LOAD] Done in %.2fs", time.time() - t0)
        return cards_df, txns_df, users_df, report

    def _step_clean(
        self,
        cards_df: Optional[pd.DataFrame],
        txns_df: Optional[pd.DataFrame],
        users_df: Optional[pd.DataFrame],
    ) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Dict[str, Any],
    ]:
        step_name = "02_cleaned"
        ckpt_dir = self._step_ckpt_dir(step_name)

        if (
            self.checkpoints_enabled
            and self.resume
            and not self.force_recompute
            and self._checkpoint_exists(step_name)
        ):
            logger.info("[CLEAN] Using checkpoint: %s", ckpt_dir)
            cards = (
                self._load_df(ckpt_dir / "credit_cards_clean.csv")
                if (ckpt_dir / "credit_cards_clean.csv").exists()
                else None
            )
            txns = (
                self._load_df(ckpt_dir / "transactions_clean.csv")
                if (ckpt_dir / "transactions_clean.csv").exists()
                else None
            )
            users = (
                self._load_df(ckpt_dir / "users_clean.csv")
                if (ckpt_dir / "users_clean.csv").exists()
                else None
            )
            report = safe_read_json(ckpt_dir / "clean_report.json")
            return cards, txns, users, report

        t0 = time.time()

        clean_cards, clean_txns, clean_users, clean_report = clean_all_data(
            credit_cards_df=cards_df,
            transactions_df=txns_df,
            users_df=users_df,
            config=self.cleaning_config,
        )

        report = {
            "cleaning_config": json_sanitize(asdict(self.cleaning_config)),
            "per_dataset": clean_report,
            "hashes": {
                "credit_cards_clean": (
                    df_hash(clean_cards) if clean_cards is not None else None
                ),
                "transactions_clean": (
                    df_hash(clean_txns) if clean_txns is not None else None
                ),
                "users_clean": (
                    df_hash(clean_users) if clean_users is not None else None
                ),
            },
        }

        if self.checkpoints_enabled:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if clean_cards is not None:
                self._save_df(clean_cards, ckpt_dir / "credit_cards_clean.csv")
            if clean_txns is not None:
                self._save_df(clean_txns, ckpt_dir / "transactions_clean.csv")
            if clean_users is not None:
                self._save_df(clean_users, ckpt_dir / "users_clean.csv")
            atomic_write_json(ckpt_dir / "clean_report.json", report)
            self._mark_checkpoint_done(step_name)

        logger.info("[CLEAN] Done in %.2fs", time.time() - t0)
        return clean_cards, clean_txns, clean_users, report

    def _step_features(
        self,
        clean_cards: Optional[pd.DataFrame],
        clean_txns: Optional[pd.DataFrame],
        clean_users: Optional[pd.DataFrame],
    ) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Dict[str, Any],
    ]:
        step_name = "03_features"
        ckpt_dir = self._step_ckpt_dir(step_name)

        if (
            self.checkpoints_enabled
            and self.resume
            and not self.force_recompute
            and self._checkpoint_exists(step_name)
        ):
            logger.info("[FE] Using checkpoint: %s", ckpt_dir)
            cards_f = (
                self._load_df(ckpt_dir / "credit_cards_features.csv")
                if (ckpt_dir / "credit_cards_features.csv").exists()
                else None
            )
            txns_f = (
                self._load_df(ckpt_dir / "transactions_features.csv")
                if (ckpt_dir / "transactions_features.csv").exists()
                else None
            )
            users_f = (
                self._load_df(ckpt_dir / "users_features.csv")
                if (ckpt_dir / "users_features.csv").exists()
                else None
            )
            report = safe_read_json(ckpt_dir / "features_report.json")
            return cards_f, txns_f, users_f, report

        t0 = time.time()

        annual_spending = float(
            (self.cfg.get("datasets", {}).get("credit_cards", {}) or {}).get(
                "annual_spending", 25000
            )
        )

        cards_f, txns_f, users_f = engineer_all_features(
            credit_cards_df=clean_cards,
            transactions_df=clean_txns,
            users_df=clean_users,
            annual_spending=annual_spending,
            output_dir=None,
        )

        report = {
            "annual_spending": annual_spending,
            "shapes": {
                "credit_cards_features": (
                    list(cards_f.shape) if cards_f is not None else None
                ),
                "transactions_features": (
                    list(txns_f.shape) if txns_f is not None else None
                ),
                "users_features": list(users_f.shape) if users_f is not None else None,
            },
            "hashes": {
                "credit_cards_features": (
                    df_hash(cards_f) if cards_f is not None else None
                ),
                "transactions_features": (
                    df_hash(txns_f) if txns_f is not None else None
                ),
                "users_features": df_hash(users_f) if users_f is not None else None,
            },
        }

        if self.checkpoints_enabled:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if cards_f is not None:
                self._save_df(cards_f, ckpt_dir / "credit_cards_features.csv")
            if txns_f is not None:
                self._save_df(txns_f, ckpt_dir / "transactions_features.csv")
            if users_f is not None:
                self._save_df(users_f, ckpt_dir / "users_features.csv")
            atomic_write_json(ckpt_dir / "features_report.json", report)
            self._mark_checkpoint_done(step_name)

        logger.info("[FE] Done in %.2fs", time.time() - t0)
        return cards_f, txns_f, users_f, report

    def _write_final_outputs(
        self,
        cards_f: Optional[pd.DataFrame],
        txns_f: Optional[pd.DataFrame],
        users_f: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        if cards_f is not None:
            out = self.final_dir / "credit_cards_features.csv"
            self._save_df(cards_f, out)
            outputs["credit_cards_features"] = str(out.relative_to(self.output_root))
        if txns_f is not None:
            out = self.final_dir / "transactions_features.csv"
            self._save_df(txns_f, out)
            outputs["transactions_features"] = str(out.relative_to(self.output_root))
        if users_f is not None:
            out = self.final_dir / "users_features.csv"
            self._save_df(users_f, out)
            outputs["users_features"] = str(out.relative_to(self.output_root))
        return outputs

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self) -> Dict[str, Optional[pd.DataFrame]]:
        run_started = utc_now_iso()
        t_run0 = time.time()

        steps_audit: Dict[str, StepAudit] = {}

        def wrap_step(name: str, fn, *args, **kwargs):
            t0 = time.time()
            started_at = utc_now_iso()
            ckpt_dir = self._step_ckpt_dir(name)
            used_ckpt = (
                self.checkpoints_enabled
                and self.resume
                and not self.force_recompute
                and self._checkpoint_exists(name)
            )

            out = fn(*args, **kwargs)

            finished_at = utc_now_iso()
            dur = round(time.time() - t0, 3)

            inputs = {}
            outputs = {}
            report_path = None
            checkpoint_dir = (
                str(ckpt_dir.relative_to(self.output_root))
                if self.checkpoints_enabled
                else None
            )

            if self.checkpoints_enabled:
                if name == "01_loaded":
                    rp = ckpt_dir / "load_report.json"
                elif name == "02_cleaned":
                    rp = ckpt_dir / "clean_report.json"
                elif name == "03_features":
                    rp = ckpt_dir / "features_report.json"
                else:
                    rp = None
                if rp and rp.exists():
                    report_path = str(rp.relative_to(self.output_root))

            steps_audit[name] = StepAudit(
                name=name,
                started_at=started_at,
                finished_at=finished_at,
                duration_s=dur,
                used_checkpoint=used_ckpt,
                inputs=inputs,
                outputs=outputs,
                report_path=report_path,
                checkpoint_dir=checkpoint_dir,
            )
            return out

        cards_df, txns_df, users_df, load_report = wrap_step(
            "01_loaded", self._step_load
        )

        clean_cards, clean_txns, clean_users, clean_report = wrap_step(
            "02_cleaned", self._step_clean, cards_df, txns_df, users_df
        )

        cards_f, txns_f, users_f, fe_report = wrap_step(
            "03_features", self._step_features, clean_cards, clean_txns, clean_users
        )

        final_outputs = self._write_final_outputs(cards_f, txns_f, users_f)

        step_reports = {
            "load": load_report,
            "clean": clean_report,
            "features": fe_report,
            "final_outputs": final_outputs,
        }
        atomic_write_json(self.audit_dir / "step_reports.json", step_reports)

        run_finished = utc_now_iso()
        run_dur = round(time.time() - t_run0, 3)

        run_audit = RunAudit(
            run_id=self.run_id,
            started_at=run_started,
            finished_at=run_finished,
            duration_s=run_dur,
            config_path=str(self.config_path),
            config_sha256=self.config_sha256,
            input_root=str(self.input_root),
            output_root=str(self.output_root),
            steps=steps_audit,
        )
        atomic_write_json(self.audit_dir / "audit.json", asdict(run_audit))

        logger.info("✅ Transformation pipeline complete. Output: %s", self.output_root)

        return {
            "credit_cards_features": cards_f,
            "transactions_features": txns_f,
            "users_features": users_f,
        }


# =============================================================================
# CLI helper (optional but convenient)
# =============================================================================


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="RewardSense transformation pipeline")
    p.add_argument(
        "--config",
        default="configs/transform.yaml",
        help="Path to YAML config (version controlled)",
    )
    p.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write a default config to --config path and exit.",
    )
    args = p.parse_args(argv)

    cfg_path = Path(args.config)

    if args.write_default_config:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(cfg_path, DEFAULT_CONFIG_YAML)
        print(f"Wrote default config to {cfg_path}")
        return 0

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}. Create it with --write-default-config."
        )

    pipeline = TransformationPipeline(cfg_path)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
