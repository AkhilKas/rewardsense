"""
Data Normalization & Encoding Module

- Fits scalers/encoders on training data, persists them via joblib.
- At inference time, loads the fitted artifacts and transforms new data with consistent column alignment.
- Unseen categories map to a configurable default (zeros for OHE, -1 for label).
- Config-driven: a YAML file specifies which columns get which treatment per dataset, keeping the logic version-controlled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Safe Label Encoder (handles unseen categories)
# =============================================================================


class SafeLabelEncoder:
    """LabelEncoder wrapper that maps unseen categories to ``-1``.

    sklearn's :class:`LabelEncoder` raises on unseen values during
    ``transform``.  This wrapper catches those and returns ``-1``,
    satisfying the Story 3.4 acceptance criterion:
    *"Encoding handles unseen categories gracefully."*
    """

    def __init__(self) -> None:
        self._le = LabelEncoder()
        self.classes_: Optional[np.ndarray] = None

    def fit(self, y: Sequence) -> "SafeLabelEncoder":
        self._le.fit(y)
        self.classes_ = self._le.classes_
        return self

    def transform(self, y: Sequence) -> np.ndarray:
        arr = np.array(y, dtype=object)
        mask = np.isin(arr, self.classes_)
        result = np.full(len(arr), -1, dtype=int)
        if mask.any():
            result[mask] = self._le.transform(arr[mask])
        n_unseen = int((~mask).sum())
        if n_unseen > 0:
            logger.warning(
                "SafeLabelEncoder: %d unseen categories mapped to -1", n_unseen
            )
        return result

    def fit_transform(self, y: Sequence) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: Sequence) -> np.ndarray:
        arr = np.array(y, dtype=int)
        mask = arr >= 0
        result = np.full(len(arr), "<UNKNOWN>", dtype=object)
        if mask.any():
            result[mask] = self._le.inverse_transform(arr[mask])
        return result


# =============================================================================
# Dataset-level normalizer
# =============================================================================


@dataclass
class DatasetNormConfig:
    """Per-dataset normalization specification (parsed from YAML)."""

    standard_scale: List[str] = field(default_factory=list)
    minmax_scale: List[str] = field(default_factory=list)
    onehot_encode: List[str] = field(default_factory=list)
    label_encode: List[str] = field(default_factory=list)
    passthrough: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetNormConfig":
        return cls(
            standard_scale=list(d.get("standard_scale", [])),
            minmax_scale=list(d.get("minmax_scale", [])),
            onehot_encode=list(d.get("onehot_encode", [])),
            label_encode=list(d.get("label_encode", [])),
            passthrough=list(d.get("passthrough", [])),
        )

    @property
    def all_configured_columns(self) -> List[str]:
        return (
            self.standard_scale
            + self.minmax_scale
            + self.onehot_encode
            + self.label_encode
            + self.passthrough
        )


# Sentinel filename written alongside joblib artifacts
_META_FILE = "normalizer_meta.json"
_ARTIFACTS_FILE = "normalizer_artifacts.joblib"


class FeatureNormalizer:
    """Fits, transforms, persists and reloads normalization for one dataset.

    Parameters
    ----------
    config : DatasetNormConfig
        Specifies column → treatment mapping.
    name : str
        Human-readable dataset name (used in logging / metadata).
    """

    def __init__(self, config: DatasetNormConfig, name: str = "dataset") -> None:
        self.config = config
        self.name = name

        # fitted artifacts (populated after .fit())
        self._standard_scalers: Dict[str, StandardScaler] = {}
        self._minmax_scalers: Dict[str, MinMaxScaler] = {}
        self._onehot_encoders: Dict[str, OneHotEncoder] = {}
        self._label_encoders: Dict[str, SafeLabelEncoder] = {}
        self._onehot_column_names: Dict[str, List[str]] = {}

        self.is_fitted: bool = False
        self._fit_timestamp: Optional[str] = None
        self._fit_row_count: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml_section(
        cls, section: Dict[str, Any], name: str = "dataset"
    ) -> "FeatureNormalizer":
        """Build from a parsed YAML section for one dataset."""
        cfg = DatasetNormConfig.from_dict(section)
        return cls(config=cfg, name=name)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        """Fit all scalers and encoders on *df*.

        Only columns listed in the config **and** present in *df* are fitted.
        Missing columns are silently skipped (logged at DEBUG level) so the
        same config can be reused across slightly different feature sets.
        """
        logger.info("[%s] Fitting normalizer on %d rows", self.name, len(df))

        if df.empty:
            logger.warning(
                "[%s] Empty DataFrame — marking as fitted with no scalers", self.name
            )
            self.is_fitted = True
            self._fit_timestamp = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            self._fit_row_count = 0
            return self

        # --- Standard scaling ---
        for col in self._resolve_cols(self.config.standard_scale, df):
            scaler = StandardScaler()
            vals = df[col].to_numpy(dtype=float, na_value=np.nan).reshape(-1, 1)
            scaler.fit(vals)
            self._standard_scalers[col] = scaler

        # --- MinMax scaling ---
        for col in self._resolve_cols(self.config.minmax_scale, df):
            scaler = MinMaxScaler()
            vals = df[col].to_numpy(dtype=float, na_value=np.nan).reshape(-1, 1)
            scaler.fit(vals)
            self._minmax_scalers[col] = scaler

        # --- One-hot encoding ---
        for col in self._resolve_cols(self.config.onehot_encode, df):
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            vals = df[col].fillna("<MISSING>").to_numpy().reshape(-1, 1)
            enc.fit(vals)
            self._onehot_encoders[col] = enc
            self._onehot_column_names[col] = [
                f"{col}_{cat}" for cat in enc.categories_[0]
            ]

        # --- Label encoding ---
        for col in self._resolve_cols(self.config.label_encode, df):
            enc = SafeLabelEncoder()
            vals = df[col].fillna("<MISSING>").to_numpy()
            enc.fit(vals)
            self._label_encoders[col] = enc

        self.is_fitted = True
        self._fit_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._fit_row_count = len(df)

        logger.info(
            "[%s] Fitted: %d standard, %d minmax, %d onehot, %d label",
            self.name,
            len(self._standard_scalers),
            len(self._minmax_scalers),
            len(self._onehot_encoders),
            len(self._label_encoders),
        )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted normalization to *df* and return a new DataFrame.

        - Numerical columns are scaled in-place.
        - OHE columns replace the original column with N binary columns.
        - Label-encoded columns are replaced with integer columns.
        - Passthrough and unlisted columns are kept unchanged.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer is not fitted. Call .fit() first.")

        out = df.copy()

        # --- Standard scaling ---
        for col, scaler in self._standard_scalers.items():
            if col not in out.columns:
                logger.debug(
                    "[%s] Column %s missing during transform, skipping", self.name, col
                )
                continue
            vals = out[col].to_numpy(dtype=float, na_value=np.nan).reshape(-1, 1)
            out[col] = scaler.transform(vals).ravel()

        # --- MinMax scaling ---
        for col, scaler in self._minmax_scalers.items():
            if col not in out.columns:
                continue
            vals = out[col].to_numpy(dtype=float, na_value=np.nan).reshape(-1, 1)
            out[col] = scaler.transform(vals).ravel()

        # --- One-hot encoding ---
        for col, enc in self._onehot_encoders.items():
            if col not in out.columns:
                continue
            vals = out[col].fillna("<MISSING>").to_numpy().reshape(-1, 1)
            encoded = enc.transform(vals)
            col_names = self._onehot_column_names[col]
            ohe_df = pd.DataFrame(encoded, columns=col_names, index=out.index)
            # drop original, insert encoded
            insert_pos = out.columns.get_loc(col)
            out = out.drop(columns=[col])
            for i, c in enumerate(col_names):
                out.insert(insert_pos + i, c, ohe_df[c])

        # --- Label encoding ---
        for col, enc in self._label_encoders.items():
            if col not in out.columns:
                continue
            vals = out[col].fillna("<MISSING>").to_numpy()
            out[col] = enc.transform(vals)

        logger.info("[%s] Transformed %d rows", self.name, len(out))
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Persist fitted normalizer to *directory*.

        Writes:
          - ``normalizer_artifacts.joblib`` — sklearn objects + config
          - ``normalizer_meta.json`` — metadata for auditing / versioning

        Returns the directory path.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted normalizer.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # artifacts bundle
        bundle = {
            "config": self.config,
            "name": self.name,
            "standard_scalers": self._standard_scalers,
            "minmax_scalers": self._minmax_scalers,
            "onehot_encoders": self._onehot_encoders,
            "label_encoders": self._label_encoders,
            "onehot_column_names": self._onehot_column_names,
        }
        artifacts_path = directory / _ARTIFACTS_FILE
        joblib.dump(bundle, artifacts_path)

        # metadata (JSON-safe, version-controlled friendly)
        meta = {
            "name": self.name,
            "fit_timestamp": self._fit_timestamp,
            "fit_row_count": self._fit_row_count,
            "standard_scale_columns": list(self._standard_scalers.keys()),
            "minmax_scale_columns": list(self._minmax_scalers.keys()),
            "onehot_encode_columns": list(self._onehot_encoders.keys()),
            "label_encode_columns": list(self._label_encoders.keys()),
            "onehot_output_columns": {
                k: v for k, v in self._onehot_column_names.items()
            },
            "label_classes": {
                col: enc.classes_.tolist() for col, enc in self._label_encoders.items()
            },
            "standard_scaler_params": {
                col: {"mean": s.mean_.tolist(), "scale": s.scale_.tolist()}
                for col, s in self._standard_scalers.items()
            },
            "minmax_scaler_params": {
                col: {
                    "min": s.data_min_.tolist(),
                    "max": s.data_max_.tolist(),
                    "scale": s.scale_.tolist(),
                }
                for col, s in self._minmax_scalers.items()
            },
        }
        meta_path = directory / _META_FILE
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        logger.info("[%s] Saved normalizer to %s", self.name, directory)
        return directory

    @classmethod
    def load(cls, directory: Path) -> "FeatureNormalizer":
        """Reload a previously saved normalizer.

        Raises FileNotFoundError if artifacts are missing.
        """
        directory = Path(directory)
        artifacts_path = directory / _ARTIFACTS_FILE
        if not artifacts_path.exists():
            raise FileNotFoundError(f"No normalizer artifacts at {artifacts_path}")

        bundle = joblib.load(artifacts_path)

        obj = cls(config=bundle["config"], name=bundle["name"])
        obj._standard_scalers = bundle["standard_scalers"]
        obj._minmax_scalers = bundle["minmax_scalers"]
        obj._onehot_encoders = bundle["onehot_encoders"]
        obj._label_encoders = bundle["label_encoders"]
        obj._onehot_column_names = bundle["onehot_column_names"]
        obj.is_fitted = True

        # load metadata for timestamps
        meta_path = directory / _META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            obj._fit_timestamp = meta.get("fit_timestamp")
            obj._fit_row_count = meta.get("fit_row_count", 0)

        logger.info("[%s] Loaded normalizer from %s", obj.name, directory)
        return obj

    # ------------------------------------------------------------------
    # Metadata / introspection
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict describing the fitted state."""
        if not self.is_fitted:
            return {"is_fitted": False, "name": self.name}
        return {
            "is_fitted": True,
            "name": self.name,
            "fit_timestamp": self._fit_timestamp,
            "fit_row_count": self._fit_row_count,
            "standard_scale_columns": list(self._standard_scalers.keys()),
            "minmax_scale_columns": list(self._minmax_scalers.keys()),
            "onehot_encode_columns": list(self._onehot_encoders.keys()),
            "label_encode_columns": list(self._label_encoders.keys()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_cols(requested: List[str], df: pd.DataFrame) -> List[str]:
        """Return only the columns from *requested* that exist in *df*."""
        present = [c for c in requested if c in df.columns]
        missing = set(requested) - set(present)
        if missing:
            logger.debug("Columns not in DataFrame (skipped): %s", missing)
        return present


# =============================================================================
# Convenience: normalize all three datasets
# =============================================================================


def normalize_all_features(
    credit_cards_df: Optional[pd.DataFrame] = None,
    transactions_df: Optional[pd.DataFrame] = None,
    users_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    save_dir: Optional[Path] = None,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Dict[str, FeatureNormalizer],
]:
    """Normalize all feature datasets using a shared config dict.

    Parameters
    ----------
    credit_cards_df, transactions_df, users_df : DataFrame, optional
        Feature-engineered outputs.
    config : dict, optional
        Parsed YAML config with keys ``credit_cards``, ``transactions``,
        ``users``. If None, uses sensible defaults.
    save_dir : Path, optional
        If provided, fitted normalizers are persisted here.

    Returns
    -------
    (cards_norm, txns_norm, users_norm, normalizers_dict)
    """
    if config is None:
        config = {}

    normalizers: Dict[str, FeatureNormalizer] = {}
    results: List[Optional[pd.DataFrame]] = []

    for key, df in [
        ("credit_cards", credit_cards_df),
        ("transactions", transactions_df),
        ("users", users_df),
    ]:
        if df is None:
            results.append(None)
            continue

        section = config.get(key, {})
        normalizer = FeatureNormalizer.from_yaml_section(section, name=key)
        df_norm = normalizer.fit_transform(df)
        normalizers[key] = normalizer

        if save_dir:
            normalizer.save(Path(save_dir) / key)

        results.append(df_norm)
        logger.info("[%s] Normalized: %s -> %s", key, df.shape, df_norm.shape)

    return results[0], results[1], results[2], normalizers


# =============================================================================
# CLI helper
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    print("=" * 70)
    print("RewardSense Normalization & Encoding Module")
    print("Story 3.4: Persistent, inference-safe normalization")
    print("=" * 70)
    print("\nFeatures:")
    print("  - StandardScaler / MinMaxScaler for numerical columns")
    print("  - OneHotEncoder (handle_unknown='ignore') for categoricals")
    print("  - SafeLabelEncoder (unseen -> -1) for label encoding")
    print("  - Persist via joblib + JSON metadata")
    print("  - Config-driven column selection (YAML)")
    print("=" * 70)
