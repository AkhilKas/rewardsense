"""
Train / validation / test splitting with optional stratification.

Default split ratios (configurable via model_config.yaml):
  train 70% / val 15% / test 15%

Stratification is based on a discretised target column (``budget_quartile``)
to ensure each split has a representative distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    """Container for the three-way split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    meta_train: Optional[pd.DataFrame] = None
    meta_val: Optional[pd.DataFrame] = None
    meta_test: Optional[pd.DataFrame] = None


def create_stratify_bins(
    y: pd.Series,
    n_bins: int = 4,
) -> pd.Series:
    """Discretise a continuous target into quantile bins for stratification.

    Falls back to a single bin if the target has fewer unique values
    than requested bins.
    """
    if y.nunique() <= n_bins:
        return y.astype(str)
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop").astype(str)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    meta: Optional[pd.DataFrame] = None,
) -> SplitResult:
    """Three-way split: train / val / test.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    train_size, val_size, test_size : float
        Must sum to ~1.0.
    random_seed : int
        Reproducibility seed.
    stratify : bool
        If True, stratify on quantile bins of ``y``.
    meta : pd.DataFrame or None
        Optional metadata frame (same index as X) to split in parallel.
        Useful for keeping ``user_id`` and segment columns alongside splits.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"Split ratios must sum to ~1.0, got {total:.3f}")

    strat_col = create_stratify_bins(y) if stratify else None

    # First split: train vs (val + test)
    val_test_size = val_size + test_size
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=val_test_size,
        random_state=random_seed,
        stratify=strat_col,
    )

    # Second split: val vs test (relative sizes)
    relative_test = test_size / val_test_size
    strat_tmp = create_stratify_bins(y_tmp) if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=relative_test,
        random_state=random_seed,
        stratify=strat_tmp,
    )

    logger.info(
        "Split sizes — train: {}, val: {}, test: {}",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    result = SplitResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    if meta is not None:
        result.meta_train = meta.loc[X_train.index]
        result.meta_val = meta.loc[X_val.index]
        result.meta_test = meta.loc[X_test.index]

    return result
