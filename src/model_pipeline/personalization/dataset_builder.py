"""
Dataset builder — joins Phase 1 feature-engineered outputs into a single
ML-ready training frame keyed by ``user_id``.

Responsibilities:
- Load users_features + transactions_features via DataPipelineLoader
- Inner-join on user_id
- Fill missing numeric values (median imputation)
- Fill missing one-hot values (0)
- Validate feature completeness
- Return (X, y) ready for splitting
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from model_pipeline.data_loader import DataPipelineLoader
from model_pipeline.personalization.features import (
    ALL_NUMERIC_FEATURES,
    TARGET_COLUMN,
    compute_derived_features,
    detect_onehot_columns,
    get_feature_columns,
    validate_feature_frame,
)


class DatasetBuildError(Exception):
    """Raised when building the training dataset fails."""


class DatasetBuilder:
    """Build the ML training frame from Phase 1 feature outputs.

    Parameters
    ----------
    loader : DataPipelineLoader or None
        If None, a default loader is created using env-based data root.
    """

    def __init__(self, loader: Optional[DataPipelineLoader] = None) -> None:
        self.loader = loader or DataPipelineLoader()

    def load_and_join(self) -> pd.DataFrame:
        """Load user and transaction features, inner-join on user_id.

        Returns
        -------
        pd.DataFrame
            Joined frame with one row per user.

        Raises
        ------
        DatasetBuildError
            If data loading or join produces zero rows.
        """
        users = self.loader.load_users_features()
        transactions = self.loader.load_transactions_features()

        logger.info(
            "Loaded {} user rows, {} transaction rows",
            len(users),
            len(transactions),
        )

        if "user_id" not in users.columns or "user_id" not in transactions.columns:
            raise DatasetBuildError("Both DataFrames must contain 'user_id'")

        merged = users.merge(transactions, on="user_id", how="inner")

        if merged.empty:
            raise DatasetBuildError(
                "Inner join on user_id produced zero rows — "
                "check that user_ids match between users_features and transactions_features"
            )

        logger.info("Joined frame: {} rows × {} cols", len(merged), len(merged.columns))
        return merged

    @staticmethod
    def impute_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Fill missing values and return imputation counts.

        Strategy:
        - Numeric features: median imputation
        - One-hot columns: fill with 0
        - All others: leave as-is

        Returns
        -------
        (df, counts)
            The imputed DataFrame and a dict mapping column -> #filled.
        """
        df = df.copy()
        counts: Dict[str, int] = {}

        numeric_cols = [c for c in ALL_NUMERIC_FEATURES if c in df.columns]
        for col in numeric_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                counts[col] = n_missing

        onehot_cols = detect_onehot_columns(df)
        for col in onehot_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                df[col] = df[col].fillna(0)
                counts[col] = n_missing

        if counts:
            logger.info("Imputed missing values in {} columns: {}", len(counts), counts)
        return df, counts

    @staticmethod
    def build_xy(
        df: pd.DataFrame,
        target: str = TARGET_COLUMN,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split the joined frame into feature matrix X and target y.

        Parameters
        ----------
        df : pd.DataFrame
            Joined + imputed frame.
        target : str
            Name of the target column.

        Returns
        -------
        (X, y)
            X contains only feature columns; y is the target series.

        Raises
        ------
        DatasetBuildError
            If target column is missing or no feature columns found.
        """
        if target not in df.columns:
            raise DatasetBuildError(f"Target column '{target}' not found in DataFrame")

        feature_cols = get_feature_columns(df)
        if not feature_cols:
            raise DatasetBuildError("No feature columns detected in DataFrame")

        X = df[feature_cols].copy().astype(np.float64)
        y = df[target].copy().astype(np.float64)

        logger.info(
            "Built X ({} rows × {} features) and y ({} rows)",
            X.shape[0],
            X.shape[1],
            len(y),
        )
        return X, y

    def build(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """End-to-end: load, join, impute, validate, split into X/y.

        Returns
        -------
        (X, y, full_df)
            Feature matrix, target vector, and the full joined frame
            (which retains user_id and metadata for downstream segmentation).
        """
        merged = self.load_and_join()
        imputed, _ = self.impute_missing(merged)
        imputed = compute_derived_features(imputed)

        warnings = validate_feature_frame(imputed)
        if any("Target column" in w for w in warnings):
            raise DatasetBuildError(
                f"Critical: target column '{TARGET_COLUMN}' missing after join"
            )

        X, y = self.build_xy(imputed)
        return X, y, imputed
