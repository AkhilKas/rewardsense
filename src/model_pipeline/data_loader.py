"""
Data Pipeline Loader — Phase 1 to Phase 2 Handoff.

Reads DVC-tracked outputs from the Phase 1 data pipeline and provides
them as validated DataFrames for model training.

Story 1.4 — Integration Test: Data Pipeline to Model Pipeline Handoff.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class DataLoadError(Exception):
    """Raised when data loading or validation fails."""

    pass


# Required columns that must exist in each dataset for the model pipeline.
REQUIRED_CREDIT_CARD_COLUMNS = [
    "card_name",
    "issuer",
    "annual_fee",
    "base_reward_rate",
    "effective_annual_fee",
    "net_value_annual",
]

REQUIRED_USER_PROFILE_COLUMNS = [
    "user_id",
]

REQUIRED_TRANSACTION_COLUMNS = [
    "user_id",
    "amount",
    "category",
]

REQUIRED_MANIFEST_KEYS = [
    "run_id",
    "sources",
]


def _resolve_data_root(data_root: Optional[str] = None) -> Path:
    """Resolve the data root directory.

    Priority:
        1. Explicit ``data_root`` argument
        2. ``DATA_ROOT`` environment variable
        3. ``data/processed/current`` relative to project root
    """
    if data_root:
        return Path(data_root)

    env_root = os.getenv("DATA_ROOT")
    if env_root:
        return Path(env_root)

    # Default: project-relative path
    return Path("data/processed/current")


class DataPipelineLoader:
    """Load and validate Phase 1 data pipeline outputs for model training.

    Parameters
    ----------
    data_root : str or None
        Path to the processed data directory. If None, uses the
        ``DATA_ROOT`` env var or defaults to ``data/processed/current``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        self.data_root = _resolve_data_root(data_root)
        logger.info("DataPipelineLoader initialised with root: {}", self.data_root)

    # ------------------------------------------------------------------
    #  Manifest
    # ------------------------------------------------------------------

    def load_manifest(self) -> Dict[str, Any]:
        """Load and validate ``manifest_latest.json``.

        Returns
        -------
        dict
            Parsed manifest with run metadata and source information.

        Raises
        ------
        DataLoadError
            If the manifest file is missing, corrupt, or missing required keys.
        """
        manifest_path = self.data_root / "manifest_latest.json"

        if not manifest_path.exists():
            raise DataLoadError(f"Manifest file not found: {manifest_path}")

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataLoadError(
                f"Failed to parse manifest {manifest_path}: {exc}"
            ) from exc

        missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
        if missing:
            raise DataLoadError(f"Manifest missing required keys: {missing}")

        logger.info(
            "Loaded manifest: run_id={}, {} sources",
            manifest.get("run_id"),
            len(manifest.get("sources", [])),
        )
        return manifest

    # ------------------------------------------------------------------
    #  Credit Cards (feature-engineered)
    # ------------------------------------------------------------------

    def _find_latest_transformed(self) -> Path:
        """Locate the most recent ``final/credit_cards_features.csv``.

        Searches ``transformed/*/final/`` directories sorted by
        modification time (most recent first).

        Raises
        ------
        DataLoadError
            If no transformed data directory is found.
        """
        transformed_root = self.data_root / "transformed"
        if not transformed_root.exists():
            raise DataLoadError(
                f"Transformed data directory not found: {transformed_root}"
            )

        run_dirs = sorted(
            (
                p
                for p in transformed_root.iterdir()
                if p.is_dir() and p.name not in ("latest", "_latest")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for run_dir in run_dirs:
            final_csv = run_dir / "final" / "credit_cards_features.csv"
            if final_csv.exists():
                return final_csv
            # Fallback to checkpoint
            ckpt_csv = (
                run_dir / "checkpoints" / "03_features" / "credit_cards_features.csv"
            )
            if ckpt_csv.exists():
                return ckpt_csv

        raise DataLoadError(
            f"No credit_cards_features.csv found under {transformed_root}"
        )

    def load_credit_cards(self) -> pd.DataFrame:
        """Load the latest feature-engineered credit card dataset.

        Returns
        -------
        pd.DataFrame
            Credit cards with all feature-engineered columns.

        Raises
        ------
        DataLoadError
            If the file is missing, empty, or missing required columns.
        """
        csv_path = self._find_latest_transformed()
        return self._load_csv(
            csv_path,
            required_columns=REQUIRED_CREDIT_CARD_COLUMNS,
            dataset_name="credit_cards",
        )

    # ------------------------------------------------------------------
    #  Synthetic User Profiles
    # ------------------------------------------------------------------

    def load_user_profiles(self) -> pd.DataFrame:
        """Load synthetic user profiles.

        Returns
        -------
        pd.DataFrame
            User profiles with card ownership and preferences.

        Raises
        ------
        DataLoadError
            If the file is missing, empty, or missing required columns.
        """
        csv_path = self.data_root / "synthetic" / "user_profiles.csv"
        return self._load_csv(
            csv_path,
            required_columns=REQUIRED_USER_PROFILE_COLUMNS,
            dataset_name="user_profiles",
        )

    # ------------------------------------------------------------------
    #  Synthetic Transactions
    # ------------------------------------------------------------------

    def load_transactions(self) -> pd.DataFrame:
        """Load synthetic transaction history.

        Returns
        -------
        pd.DataFrame
            Transaction records with user_id, amount, category, etc.

        Raises
        ------
        DataLoadError
            If the file is missing, empty, or missing required columns.
        """
        csv_path = self.data_root / "synthetic" / "transactions.csv"
        return self._load_csv(
            csv_path,
            required_columns=REQUIRED_TRANSACTION_COLUMNS,
            dataset_name="transactions",
        )

    # ------------------------------------------------------------------
    #  Load All
    # ------------------------------------------------------------------

    def load_all(self) -> Dict[str, Any]:
        """Load all Phase 1 outputs and validate schemas.

        Returns
        -------
        dict
            Keys: ``manifest``, ``credit_cards``, ``user_profiles``,
            ``transactions``.

        Raises
        ------
        DataLoadError
            If any component fails to load or validate.
        """
        manifest = self.load_manifest()
        credit_cards = self.load_credit_cards()
        user_profiles = self.load_user_profiles()
        transactions = self.load_transactions()

        logger.info(
            "All Phase 1 data loaded: {} cards, {} users, {} transactions",
            len(credit_cards),
            len(user_profiles),
            len(transactions),
        )

        return {
            "manifest": manifest,
            "credit_cards": credit_cards,
            "user_profiles": user_profiles,
            "transactions": transactions,
        }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_schema(
        df: pd.DataFrame,
        required_columns: List[str],
        dataset_name: str = "dataset",
    ) -> None:
        """Verify that a DataFrame contains all required columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.
        required_columns : list of str
            Column names that must be present.
        dataset_name : str
            Human-readable name for error messages.

        Raises
        ------
        DataLoadError
            If any required columns are missing.
        """
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise DataLoadError(
                f"{dataset_name} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def _load_csv(
        self,
        path: Path,
        required_columns: List[str],
        dataset_name: str,
    ) -> pd.DataFrame:
        """Load a CSV file, validate schema, and return as DataFrame.

        Raises
        ------
        DataLoadError
            If the file is missing, corrupt, or fails schema validation.
        """
        if not path.exists():
            raise DataLoadError(f"{dataset_name} file not found: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise DataLoadError(
                f"Failed to read {dataset_name} from {path}: {exc}"
            ) from exc

        if df.empty:
            logger.warning("{} loaded but is empty: {}", dataset_name, path)
            return df

        self.validate_schema(df, required_columns, dataset_name)

        logger.info(
            "Loaded {}: {} rows × {} cols from {}",
            dataset_name,
            len(df),
            len(df.columns),
            path,
        )
        return df
