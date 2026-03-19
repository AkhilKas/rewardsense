"""
Feature schema constants and selection utilities for the personalization model.

Defines which columns from Phase 1 feature-engineered outputs are used
as inputs to the point-valuation regression model (target: estimated_point_value).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# ── Target ────────────────────────────────────────────────────────────
TARGET_COLUMN = "estimated_point_value"

# ── Numeric features from users_features.csv ──────────────────────────
USER_NUMERIC_FEATURES: List[str] = [
    "monthly_budget",
    "annual_budget",
    "num_cards",
    "monthly_budget_log",
    "age_group_ordinal",
]

# ── Numeric features from transactions_features.csv ───────────────────
TRANSACTION_NUMERIC_FEATURES: List[str] = [
    "total_spending",
    "total_transactions",
    "avg_transaction_amount",
    "median_transaction_amount",
    "transaction_amount_std",
    "spending_diversity",
    "weekend_spending_ratio",
    "card_switch_rate",
    "num_cards_used",
    "num_unique_mccs",
    "num_unique_merchants",
    "repeat_merchant_ratio",
    "peak_spending_day",
    "peak_spending_month",
    "spending_velocity",
    "category_affinity_score",
]

# ── One-hot-encoded prefixes (auto-detected from columns) ────────────
ONEHOT_PREFIXES: List[str] = [
    "archetype_",
    "age_",
    "location_",
    "redemption_",
    "budget_",
]

ALL_NUMERIC_FEATURES = USER_NUMERIC_FEATURES + TRANSACTION_NUMERIC_FEATURES


def detect_onehot_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that match any of the one-hot prefixes.

    Only includes columns with numeric (int/float/bool) dtypes to avoid
    picking up raw string columns like ``age_group`` or ``budget_quartile``.
    """
    cols: List[str] = []
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ONEHOT_PREFIXES):
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
    return sorted(cols)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the full ordered list of feature columns present in *df*.

    Includes numeric features that exist in the DataFrame plus any
    detected one-hot columns (excluding those already in the numeric list
    to avoid duplicates).
    """
    present_numeric = [c for c in ALL_NUMERIC_FEATURES if c in df.columns]
    numeric_set = set(present_numeric)
    onehot = [c for c in detect_onehot_columns(df) if c not in numeric_set]
    return present_numeric + onehot


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features not present in Phase 1 outputs.

    Adds the following columns in-place (if missing):
    - ``spending_velocity``: coefficient of variation of transaction amounts
      (std / mean), capturing how erratic a user's spending is over time.
    - ``category_affinity_score``: normalised spending diversity (entropy
      divided by log2 of number of spending categories), measuring how
      concentrated a user's spend is across categories.  1.0 = uniform,
      closer to 0 = highly concentrated.
    - ``peak_spending_day`` / ``peak_spending_month``: carried from Phase 1
      temporal features if present, otherwise defaults to 0.
    """
    df = df.copy()

    if "spending_velocity" not in df.columns:
        avg = df.get("avg_transaction_amount")
        std = df.get("transaction_amount_std")
        if avg is not None and std is not None:
            safe_avg = avg.replace(0, np.nan)
            df["spending_velocity"] = (std / safe_avg).fillna(0.0).round(4)
        else:
            df["spending_velocity"] = 0.0
        logger.info("Computed spending_velocity for {} rows", len(df))

    if "category_affinity_score" not in df.columns:
        diversity = df.get("spending_diversity")
        if diversity is not None:
            max_entropy = np.log2(7)  # 7 standard spending categories
            df["category_affinity_score"] = (
                (diversity / max_entropy).clip(0, 1).fillna(0.0).round(4)
            )
        else:
            df["category_affinity_score"] = 0.0
        logger.info("Computed category_affinity_score for {} rows", len(df))

    for col in ("peak_spending_day", "peak_spending_month"):
        if col not in df.columns:
            df[col] = 0
            logger.debug("Defaulted {} to 0 (not in source data)", col)

    return df


def load_feature_config(config_path: str) -> Dict:
    """Load personalization feature config from the YAML config file."""
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("personalization", {})


def validate_feature_frame(
    df: pd.DataFrame,
    required_numeric: Optional[List[str]] = None,
    target: str = TARGET_COLUMN,
) -> List[str]:
    """Check that *df* contains the required features and target.

    Returns a list of warnings (empty if all good).
    """
    if required_numeric is None:
        required_numeric = ALL_NUMERIC_FEATURES

    warnings: List[str] = []

    if target not in df.columns:
        warnings.append(f"Target column '{target}' missing")

    missing = [c for c in required_numeric if c not in df.columns]
    if missing:
        warnings.append(f"Missing numeric features: {missing}")

    onehot = detect_onehot_columns(df)
    if not onehot:
        warnings.append("No one-hot encoded columns detected")

    if warnings:
        for w in warnings:
            logger.warning("Feature validation: {}", w)
    else:
        logger.info(
            "Feature validation passed — {} numeric + {} one-hot features",
            len([c for c in required_numeric if c in df.columns]),
            len(onehot),
        )
    return warnings
