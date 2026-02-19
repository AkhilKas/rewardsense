"""
Unit tests for normalization & encoding module.

Acceptance criteria:
  - Encoders persist and reload correctly
  - Normalization parameters versioned with data
  - Encoding handles unseen categories gracefully
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.preprocessing.normalization import (
    DatasetNormConfig,
    FeatureNormalizer,
    SafeLabelEncoder,
    normalize_all_features,
)

# Fixtures


@pytest.fixture
def numeric_df():
    """DataFrame with continuous numeric features."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(50)],
            "total_spending": np.random.normal(5000, 2000, 50),
            "avg_amount": np.random.normal(50, 20, 50),
            "diversity": np.random.uniform(0, 1, 50),
            "num_cards": np.random.randint(1, 6, 50).astype(float),
        }
    )


@pytest.fixture
def categorical_df():
    """DataFrame with categorical string features."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "primary_card": [
                "Chase Sapphire",
                "Amex Gold",
                "Citi DC",
                "Chase Sapphire",
                "Amex Gold",
            ],
            "difficulty": ["easy", "medium", "hard", "easy", "none"],
            "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture
def mixed_df():
    """DataFrame resembling real transaction features."""
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(10)],
            "total_spending": np.random.normal(5000, 1500, 10),
            "avg_transaction_amount": np.random.normal(50, 15, 10),
            "spending_diversity": np.random.uniform(0, 1, 10),
            "card_switch_rate": np.random.uniform(0, 1, 10),
            "primary_card": [
                "Card A",
                "Card B",
                "Card A",
                "Card C",
                "Card B",
                "Card A",
                "Card D",
                "Card B",
                "Card C",
                "Card A",
            ],
            "favorite_merchant": [
                "Starbucks",
                "Amazon",
                "Starbucks",
                "Costco",
                "Amazon",
                "Target",
                "Starbucks",
                "Amazon",
                "Costco",
                "Target",
            ],
            "num_cards_used": [2, 3, 1, 4, 2, 3, 1, 2, 3, 2],
        }
    )


@pytest.fixture
def standard_config():
    """Config with standard scaling and label encoding."""
    return DatasetNormConfig(
        standard_scale=["total_spending", "avg_amount"],
        minmax_scale=["diversity", "num_cards"],
        onehot_encode=[],
        label_encode=[],
        passthrough=["user_id"],
    )


@pytest.fixture
def mixed_config():
    """Config with all transform types."""
    return DatasetNormConfig(
        standard_scale=["total_spending", "avg_transaction_amount"],
        minmax_scale=["spending_diversity", "card_switch_rate", "num_cards_used"],
        onehot_encode=["favorite_merchant"],
        label_encode=["primary_card"],
        passthrough=["user_id"],
    )


# SafeLabelEncoder


class TestSafeLabelEncoder:
    """Tests for the unseen-category-safe label encoder."""

    def test_fit_transform_known_values(self):
        enc = SafeLabelEncoder()
        vals = ["a", "b", "c", "a", "b"]
        result = enc.fit_transform(vals)
        assert set(result) == {0, 1, 2}
        assert len(result) == 5

    def test_unseen_categories_map_to_minus_one(self):
        enc = SafeLabelEncoder()
        enc.fit(["a", "b", "c"])
        result = enc.transform(["a", "d", "e", "b"])
        assert result[0] >= 0  # 'a' is known
        assert result[1] == -1  # 'd' unseen
        assert result[2] == -1  # 'e' unseen
        assert result[3] >= 0  # 'b' is known

    def test_inverse_transform_handles_unknown(self):
        enc = SafeLabelEncoder()
        enc.fit(["cat", "dog", "fish"])
        encoded = enc.transform(["cat", "alien"])
        decoded = enc.inverse_transform(encoded)
        assert decoded[0] == "cat"
        assert decoded[1] == "<UNKNOWN>"

    def test_all_unseen(self):
        enc = SafeLabelEncoder()
        enc.fit(["x", "y"])
        result = enc.transform(["a", "b", "c"])
        assert all(r == -1 for r in result)

    def test_empty_input(self):
        enc = SafeLabelEncoder()
        enc.fit(["a", "b"])
        result = enc.transform([])
        assert len(result) == 0

    def test_nan_in_input(self):
        enc = SafeLabelEncoder()
        enc.fit(["a", "b", None])
        result = enc.transform(["a", None, "c"])
        assert result[0] >= 0
        assert result[2] == -1  # 'c' unseen


# FeatureNormalizer — Fitting & Transforming


class TestFeatureNormalizerFitTransform:
    """Test fit/transform on various data configurations."""

    def test_standard_scaling_produces_zero_mean_unit_var(
        self, numeric_df, standard_config
    ):
        norm = FeatureNormalizer(config=standard_config, name="test")
        result = norm.fit_transform(numeric_df)

        for col in ["total_spending", "avg_amount"]:
            assert abs(result[col].mean()) < 0.01
            assert abs(result[col].std() - 1.0) < 0.1

    def test_minmax_scaling_produces_0_to_1_range(self, numeric_df, standard_config):
        norm = FeatureNormalizer(config=standard_config, name="test")
        result = norm.fit_transform(numeric_df)

        for col in ["diversity", "num_cards"]:
            assert result[col].min() >= -0.01  # float tolerance
            assert result[col].max() <= 1.01

    def test_passthrough_columns_unchanged(self, numeric_df, standard_config):
        norm = FeatureNormalizer(config=standard_config, name="test")
        result = norm.fit_transform(numeric_df)

        pd.testing.assert_series_equal(
            result["user_id"], numeric_df["user_id"], check_names=True
        )

    def test_label_encoding_produces_integers(self, categorical_df):
        cfg = DatasetNormConfig(label_encode=["primary_card", "difficulty"])
        norm = FeatureNormalizer(config=cfg, name="test")
        result = norm.fit_transform(categorical_df)

        assert result["primary_card"].dtype in (np.int64, np.int32, int)
        assert result["difficulty"].dtype in (np.int64, np.int32, int)
        assert (result["primary_card"] >= 0).all()

    def test_onehot_encoding_expands_columns(self, categorical_df):
        cfg = DatasetNormConfig(onehot_encode=["primary_card"])
        norm = FeatureNormalizer(config=cfg, name="test")
        result = norm.fit_transform(categorical_df)

        # original column gone
        assert "primary_card" not in result.columns
        # 3 unique cards -> 3 OHE columns
        ohe_cols = [c for c in result.columns if c.startswith("primary_card_")]
        assert len(ohe_cols) == 3
        # each row sums to 1
        assert (result[ohe_cols].sum(axis=1) == 1.0).all()

    def test_onehot_unseen_category_produces_all_zeros(self, categorical_df):
        """Acceptance criteria: encoding handles unseen categories gracefully."""
        cfg = DatasetNormConfig(onehot_encode=["primary_card"])
        norm = FeatureNormalizer(config=cfg, name="test")
        norm.fit(categorical_df)

        new_data = pd.DataFrame(
            {
                "user_id": ["u99"],
                "primary_card": ["BRAND_NEW_CARD"],
                "difficulty": ["easy"],
                "score": [99.0],
            }
        )
        result = norm.transform(new_data)
        ohe_cols = [c for c in result.columns if c.startswith("primary_card_")]
        # unseen card -> all zeros (handle_unknown='ignore')
        assert (result[ohe_cols].sum(axis=1) == 0.0).all()

    def test_mixed_config_all_transforms(self, mixed_df, mixed_config):
        norm = FeatureNormalizer(config=mixed_config, name="test")
        result = norm.fit_transform(mixed_df)

        # standard scaled cols near zero mean
        assert abs(result["total_spending"].mean()) < 0.1

        # minmax cols in [0, 1]
        assert result["spending_diversity"].min() >= -0.01
        assert result["spending_diversity"].max() <= 1.01

        # label encoded col is integer
        assert result["primary_card"].dtype in (np.int64, np.int32, int)

        # OHE expanded favorite_merchant
        ohe_cols = [c for c in result.columns if c.startswith("favorite_merchant_")]
        assert len(ohe_cols) >= 3

        # passthrough preserved
        assert "user_id" in result.columns

    def test_missing_columns_skipped_gracefully(self, numeric_df):
        """Config references columns not in DataFrame — should not crash."""
        cfg = DatasetNormConfig(
            standard_scale=["total_spending", "nonexistent_col"],
            minmax_scale=["also_missing"],
            passthrough=["user_id"],
        )
        norm = FeatureNormalizer(config=cfg, name="test")
        result = norm.fit_transform(numeric_df)

        # Only total_spending was actually scaled
        assert abs(result["total_spending"].mean()) < 0.1
        assert len(result) == len(numeric_df)

    def test_nan_values_handled(self):
        """NaN in numeric columns should not crash scalers."""
        df = pd.DataFrame(
            {
                "val": [1.0, 2.0, np.nan, 4.0, 5.0],
                "cat": ["a", "b", "a", None, "b"],
            }
        )
        cfg = DatasetNormConfig(standard_scale=["val"], label_encode=["cat"])
        norm = FeatureNormalizer(config=cfg, name="test")
        result = norm.fit_transform(df)
        assert len(result) == 5


# FeatureNormalizer — Persistence


class TestFeatureNormalizerPersistence:
    """Acceptance criteria: Encoders persist and reload correctly."""

    def test_save_and_load_roundtrip(self, tmp_path, mixed_df, mixed_config):
        # Fit and save
        norm = FeatureNormalizer(config=mixed_config, name="txns")
        df_orig = norm.fit_transform(mixed_df)
        save_dir = tmp_path / "normalizers" / "txns"
        norm.save(save_dir)

        # Load and transform same data
        loaded = FeatureNormalizer.load(save_dir)
        df_loaded = loaded.transform(mixed_df)

        # Outputs should be identical
        pd.testing.assert_frame_equal(df_orig, df_loaded)

    def test_save_creates_expected_files(self, tmp_path, numeric_df, standard_config):
        norm = FeatureNormalizer(config=standard_config, name="test")
        norm.fit_transform(numeric_df)
        save_dir = tmp_path / "norm"
        norm.save(save_dir)

        assert (save_dir / "normalizer_artifacts.joblib").exists()
        assert (save_dir / "normalizer_meta.json").exists()

    def test_metadata_json_is_valid_and_contains_params(
        self, tmp_path, numeric_df, standard_config
    ):
        """Acceptance criteria: Normalization parameters versioned with data."""
        norm = FeatureNormalizer(config=standard_config, name="test")
        norm.fit_transform(numeric_df)
        norm.save(tmp_path / "norm")

        meta = json.loads((tmp_path / "norm" / "normalizer_meta.json").read_text())

        assert meta["name"] == "test"
        assert meta["fit_timestamp"] is not None
        assert meta["fit_row_count"] == 50
        assert "total_spending" in meta["standard_scale_columns"]
        assert "diversity" in meta["minmax_scale_columns"]

        # Check scaler params are recorded
        assert "total_spending" in meta["standard_scaler_params"]
        assert "mean" in meta["standard_scaler_params"]["total_spending"]
        assert "scale" in meta["standard_scaler_params"]["total_spending"]

    def test_load_nonexistent_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FeatureNormalizer.load(tmp_path / "does_not_exist")

    def test_save_unfitted_raises(self, tmp_path):
        cfg = DatasetNormConfig(standard_scale=["x"])
        norm = FeatureNormalizer(config=cfg, name="test")
        with pytest.raises(RuntimeError, match="unfitted"):
            norm.save(tmp_path / "bad")

    def test_transform_unfitted_raises(self, numeric_df):
        cfg = DatasetNormConfig(standard_scale=["total_spending"])
        norm = FeatureNormalizer(config=cfg, name="test")
        with pytest.raises(RuntimeError, match="not fitted"):
            norm.transform(numeric_df)

    def test_loaded_normalizer_handles_unseen_categories(
        self, tmp_path, categorical_df
    ):
        """Full roundtrip: fit -> save -> load -> transform unseen data."""
        cfg = DatasetNormConfig(
            label_encode=["primary_card"],
            onehot_encode=["difficulty"],
        )
        norm = FeatureNormalizer(config=cfg, name="test")
        norm.fit(categorical_df)
        norm.save(tmp_path / "enc")

        loaded = FeatureNormalizer.load(tmp_path / "enc")

        new_data = pd.DataFrame(
            {
                "user_id": ["u99"],
                "primary_card": ["NEVER_SEEN_CARD"],
                "difficulty": ["NEVER_SEEN_DIFF"],
                "score": [100.0],
            }
        )
        result = loaded.transform(new_data)

        # label encoded unseen -> -1
        assert result["primary_card"].iloc[0] == -1
        # OHE unseen -> all zeros
        ohe_cols = [c for c in result.columns if c.startswith("difficulty_")]
        assert result[ohe_cols].iloc[0].sum() == 0.0


# FeatureNormalizer — Reproducibility


class TestNormalizerReproducibility:
    """Verify deterministic output."""

    def test_same_input_same_output(self, mixed_df, mixed_config):
        n1 = FeatureNormalizer(config=mixed_config, name="test")
        n2 = FeatureNormalizer(config=mixed_config, name="test")

        r1 = n1.fit_transform(mixed_df)
        r2 = n2.fit_transform(mixed_df)

        pd.testing.assert_frame_equal(r1, r2)

    def test_save_load_produces_same_output(self, tmp_path, mixed_df, mixed_config):
        n1 = FeatureNormalizer(config=mixed_config, name="test")
        r1 = n1.fit_transform(mixed_df)
        n1.save(tmp_path / "norm")

        n2 = FeatureNormalizer.load(tmp_path / "norm")
        r2 = n2.transform(mixed_df)

        pd.testing.assert_frame_equal(r1, r2)


# DatasetNormConfig


class TestDatasetNormConfig:

    def test_from_dict(self):
        d = {
            "standard_scale": ["a", "b"],
            "minmax_scale": ["c"],
            "label_encode": ["d"],
            "onehot_encode": ["e"],
            "passthrough": ["f"],
        }
        cfg = DatasetNormConfig.from_dict(d)
        assert cfg.standard_scale == ["a", "b"]
        assert cfg.minmax_scale == ["c"]
        assert cfg.label_encode == ["d"]
        assert cfg.onehot_encode == ["e"]
        assert cfg.passthrough == ["f"]

    def test_from_empty_dict(self):
        cfg = DatasetNormConfig.from_dict({})
        assert cfg.standard_scale == []
        assert cfg.all_configured_columns == []

    def test_all_configured_columns(self):
        cfg = DatasetNormConfig(
            standard_scale=["a"],
            minmax_scale=["b"],
            onehot_encode=["c"],
            label_encode=["d"],
            passthrough=["e"],
        )
        assert cfg.all_configured_columns == ["a", "b", "c", "d", "e"]


# normalize_all_features convenience function


class TestNormalizeAllFeatures:

    def test_normalizes_all_three_datasets(self, mixed_df, numeric_df):
        users_df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "monthly_budget": [3000.0, 5000.0, 8000.0],
                "budget_quartile": ["Q1", "Q2", "Q3"],
            }
        )
        config = {
            "credit_cards": {"standard_scale": ["total_spending"]},
            "transactions": {"standard_scale": ["avg_transaction_amount"]},
            "users": {
                "standard_scale": ["monthly_budget"],
                "label_encode": ["budget_quartile"],
            },
        }
        cards_n, txns_n, users_n, normalizers = normalize_all_features(
            credit_cards_df=mixed_df,
            transactions_df=numeric_df,
            users_df=users_df,
            config=config,
        )
        assert cards_n is not None
        assert txns_n is not None
        assert users_n is not None
        assert "credit_cards" in normalizers
        assert "transactions" in normalizers
        assert "users" in normalizers

    def test_handles_none_inputs(self):
        cards_n, txns_n, users_n, normalizers = normalize_all_features()
        assert cards_n is None
        assert txns_n is None
        assert users_n is None
        assert normalizers == {}

    def test_partial_inputs(self, numeric_df):
        config = {"transactions": {"standard_scale": ["total_spending"]}}
        cards_n, txns_n, users_n, normalizers = normalize_all_features(
            transactions_df=numeric_df, config=config
        )
        assert cards_n is None
        assert txns_n is not None
        assert users_n is None
        assert "transactions" in normalizers

    def test_saves_normalizers_when_save_dir_provided(self, tmp_path, mixed_df):
        config = {"credit_cards": {"standard_scale": ["total_spending"]}}
        _, _, _, normalizers = normalize_all_features(
            credit_cards_df=mixed_df,
            config=config,
            save_dir=tmp_path / "norms",
        )
        assert (
            tmp_path / "norms" / "credit_cards" / "normalizer_artifacts.joblib"
        ).exists()
        assert (tmp_path / "norms" / "credit_cards" / "normalizer_meta.json").exists()


# Edge Cases


class TestEdgeCases:

    def test_empty_dataframe(self):
        df = pd.DataFrame({"val": pd.Series(dtype=float), "cat": pd.Series(dtype=str)})
        cfg = DatasetNormConfig(standard_scale=["val"], label_encode=["cat"])
        norm = FeatureNormalizer(config=cfg, name="test")
        # fit on empty should not crash
        result = norm.fit_transform(df)
        assert len(result) == 0

    def test_single_row(self):
        df = pd.DataFrame({"val": [5.0], "cat": ["a"]})
        cfg = DatasetNormConfig(standard_scale=["val"], label_encode=["cat"])
        norm = FeatureNormalizer(config=cfg, name="test")
        result = norm.fit_transform(df)
        assert len(result) == 1

    def test_single_unique_value_standard_scaler(self):
        """Constant column: std=0, scaler should handle gracefully."""
        df = pd.DataFrame({"val": [5.0, 5.0, 5.0, 5.0]})
        cfg = DatasetNormConfig(standard_scale=["val"])
        norm = FeatureNormalizer(config=cfg, name="test")
        result = norm.fit_transform(df)
        # sklearn StandardScaler with constant col -> all zeros
        assert (result["val"] == 0.0).all()

    def test_all_nan_column(self):
        df = pd.DataFrame({"val": [np.nan, np.nan, np.nan]})
        cfg = DatasetNormConfig(standard_scale=["val"])
        norm = FeatureNormalizer(config=cfg, name="test")
        result = norm.fit_transform(df)
        assert len(result) == 3

    def test_config_with_no_transforms(self):
        """Empty config should pass everything through unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        cfg = DatasetNormConfig()
        norm = FeatureNormalizer(config=cfg, name="test")
        result = norm.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_get_metadata_unfitted(self):
        cfg = DatasetNormConfig()
        norm = FeatureNormalizer(config=cfg, name="test")
        meta = norm.get_metadata()
        assert meta["is_fitted"] is False

    def test_get_metadata_fitted(self, numeric_df, standard_config):
        norm = FeatureNormalizer(config=standard_config, name="test")
        norm.fit(numeric_df)
        meta = norm.get_metadata()
        assert meta["is_fitted"] is True
        assert meta["fit_row_count"] == 50
        assert "total_spending" in meta["standard_scale_columns"]
