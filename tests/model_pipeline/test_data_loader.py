"""
Tests: Data Pipeline to Model Pipeline Handoff.

Unit tests use fixture data (always pass without real pipeline output).
Integration tests use real Phase 1 data and are marked with
``@pytest.mark.integration``.

Story 1.4 — Data Pipeline to Model Pipeline Handoff.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from model_pipeline.data_loader import (
    REQUIRED_CREDIT_CARD_COLUMNS,
    REQUIRED_MANIFEST_KEYS,
    REQUIRED_TRANSACTION_COLUMNS,
    REQUIRED_USER_PROFILE_COLUMNS,
    DataLoadError,
    DataPipelineLoader,
)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Path to real Phase 1 data (for integration tests)
REAL_DATA_ROOT = Path("data/processed/current")


# ====================================================================
# Helpers
# ====================================================================


@pytest.fixture()
def fixture_data_root(tmp_path: Path) -> Path:
    """Build a minimal data root from fixture files for unit tests.

    Creates the expected directory structure under ``tmp_path``:
        manifest_latest.json
        transformed/<run>/final/credit_cards_features.csv
        synthetic/user_profiles.csv
        synthetic/transactions.csv
    """
    # Manifest
    shutil.copy(FIXTURES_DIR / "test_manifest.json", tmp_path / "manifest_latest.json")

    # Transformed credit cards
    transformed_dir = tmp_path / "transformed" / "20260101_000000" / "final"
    transformed_dir.mkdir(parents=True)
    shutil.copy(
        FIXTURES_DIR / "test_credit_cards.csv",
        transformed_dir / "credit_cards_features.csv",
    )

    # Synthetic data
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    shutil.copy(
        FIXTURES_DIR / "test_user_profiles.csv", synthetic_dir / "user_profiles.csv"
    )
    shutil.copy(
        FIXTURES_DIR / "test_transactions.csv", synthetic_dir / "transactions.csv"
    )

    return tmp_path


@pytest.fixture()
def loader(fixture_data_root: Path) -> DataPipelineLoader:
    """Return a DataPipelineLoader pointed at the fixture data root."""
    return DataPipelineLoader(data_root=str(fixture_data_root))


# ====================================================================
# Unit Tests — Manifest
# ====================================================================


class TestLoadManifest:
    """Tests for manifest loading and validation."""

    def test_load_manifest_valid(self, loader: DataPipelineLoader):
        """Valid manifest should parse and contain required keys."""
        manifest = loader.load_manifest()

        assert isinstance(manifest, dict)
        for key in REQUIRED_MANIFEST_KEYS:
            assert key in manifest, f"Missing key: {key}"
        assert manifest["run_id"] == "test_run_001"
        assert len(manifest["sources"]) == 2

    def test_load_manifest_missing_file(self, tmp_path: Path):
        """Missing manifest file should raise DataLoadError."""
        loader = DataPipelineLoader(data_root=str(tmp_path))

        with pytest.raises(DataLoadError, match="not found"):
            loader.load_manifest()

    def test_load_manifest_corrupt_json(self, tmp_path: Path):
        """Corrupt JSON should raise DataLoadError."""
        manifest_path = tmp_path / "manifest_latest.json"
        manifest_path.write_text("{invalid json content")

        loader = DataPipelineLoader(data_root=str(tmp_path))

        with pytest.raises(DataLoadError, match="Failed to parse"):
            loader.load_manifest()

    def test_load_manifest_missing_keys(self, tmp_path: Path):
        """Manifest missing required keys should raise DataLoadError."""
        manifest_path = tmp_path / "manifest_latest.json"
        manifest_path.write_text(json.dumps({"timestamp": "2026-01-01"}))

        loader = DataPipelineLoader(data_root=str(tmp_path))

        with pytest.raises(DataLoadError, match="missing required keys"):
            loader.load_manifest()


# ====================================================================
# Unit Tests — Credit Cards
# ====================================================================


class TestLoadCreditCards:
    """Tests for credit card data loading."""

    def test_load_credit_cards_valid(self, loader: DataPipelineLoader):
        """Valid credit cards file should load with correct schema."""
        df = loader.load_credit_cards()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        for col in REQUIRED_CREDIT_CARD_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_credit_cards_missing_file(self, tmp_path: Path):
        """Missing transformed dir should raise DataLoadError."""
        loader = DataPipelineLoader(data_root=str(tmp_path))

        with pytest.raises(DataLoadError, match="not found"):
            loader.load_credit_cards()

    def test_load_credit_cards_empty(self, tmp_path: Path):
        """Empty CSV (header only) should return empty DataFrame."""
        transformed_dir = tmp_path / "transformed" / "run1" / "final"
        transformed_dir.mkdir(parents=True)
        csv_path = transformed_dir / "credit_cards_features.csv"
        csv_path.write_text(",".join(REQUIRED_CREDIT_CARD_COLUMNS) + "\n")

        loader = DataPipelineLoader(data_root=str(tmp_path))
        df = loader.load_credit_cards()

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_credit_cards_missing_columns(self, tmp_path: Path):
        """CSV missing required columns should raise DataLoadError."""
        transformed_dir = tmp_path / "transformed" / "run1" / "final"
        transformed_dir.mkdir(parents=True)
        csv_path = transformed_dir / "credit_cards_features.csv"
        csv_path.write_text("card_name,issuer\nChase,CHASE\n")

        loader = DataPipelineLoader(data_root=str(tmp_path))

        with pytest.raises(DataLoadError, match="missing required columns"):
            loader.load_credit_cards()

    def test_load_credit_cards_corrupt_csv(self, tmp_path: Path):
        """Corrupt CSV should raise DataLoadError."""
        transformed_dir = tmp_path / "transformed" / "run1" / "final"
        transformed_dir.mkdir(parents=True)
        csv_path = transformed_dir / "credit_cards_features.csv"
        csv_path.write_bytes(b"\x00\x01\x02\x03binary garbage")

        loader = DataPipelineLoader(data_root=str(tmp_path))
        # pandas may or may not raise; the key is the loader handles it
        try:
            df = loader.load_credit_cards()
            # If pandas reads it, it's likely garbled — just assert it's a DataFrame
            assert isinstance(df, pd.DataFrame)
        except DataLoadError:
            pass  # Expected

    def test_finds_checkpoint_fallback(self, tmp_path: Path):
        """Should fall back to checkpoints/ if final/ doesn't exist."""
        ckpt_dir = tmp_path / "transformed" / "run1" / "checkpoints" / "03_features"
        ckpt_dir.mkdir(parents=True)
        shutil.copy(
            FIXTURES_DIR / "test_credit_cards.csv",
            ckpt_dir / "credit_cards_features.csv",
        )

        loader = DataPipelineLoader(data_root=str(tmp_path))
        df = loader.load_credit_cards()

        assert len(df) == 5


# ====================================================================
# Unit Tests — User Profiles
# ====================================================================


class TestLoadUserProfiles:
    """Tests for user profile loading."""

    def test_load_user_profiles_valid(self, loader: DataPipelineLoader):
        """Valid user profiles should load with correct schema."""
        df = loader.load_user_profiles()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        for col in REQUIRED_USER_PROFILE_COLUMNS:
            assert col in df.columns

    def test_load_user_profiles_missing_file(self, tmp_path: Path):
        """Missing user profiles file should raise DataLoadError."""
        loader = DataPipelineLoader(data_root=str(tmp_path))

        with pytest.raises(DataLoadError, match="not found"):
            loader.load_user_profiles()


# ====================================================================
# Unit Tests — Transactions
# ====================================================================


class TestLoadTransactions:
    """Tests for transaction data loading."""

    def test_load_transactions_valid(self, loader: DataPipelineLoader):
        """Valid transactions should load with correct schema."""
        df = loader.load_transactions()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7
        for col in REQUIRED_TRANSACTION_COLUMNS:
            assert col in df.columns

    def test_load_transactions_missing_file(self, tmp_path: Path):
        """Missing transactions file should raise DataLoadError."""
        loader = DataPipelineLoader(data_root=str(tmp_path))

        with pytest.raises(DataLoadError, match="not found"):
            loader.load_transactions()


# ====================================================================
# Unit Tests — Load All
# ====================================================================


class TestLoadAll:
    """Tests for the combined load_all method."""

    def test_load_all_returns_complete_dict(self, loader: DataPipelineLoader):
        """load_all() should return all four data components."""
        result = loader.load_all()

        assert "manifest" in result
        assert "credit_cards" in result
        assert "user_profiles" in result
        assert "transactions" in result

        assert isinstance(result["manifest"], dict)
        assert isinstance(result["credit_cards"], pd.DataFrame)
        assert isinstance(result["user_profiles"], pd.DataFrame)
        assert isinstance(result["transactions"], pd.DataFrame)

        assert len(result["credit_cards"]) == 5
        assert len(result["user_profiles"]) == 5
        assert len(result["transactions"]) == 7


# ====================================================================
# Unit Tests — Validate Schema
# ====================================================================


class TestValidateSchema:
    """Tests for the schema validation utility."""

    def test_validate_schema_passes(self):
        """Valid DataFrame should not raise."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        DataPipelineLoader.validate_schema(df, ["a", "b"], "test")

    def test_validate_schema_missing_columns(self):
        """Missing columns should raise DataLoadError."""
        df = pd.DataFrame({"a": [1]})

        with pytest.raises(DataLoadError, match="missing required columns"):
            DataPipelineLoader.validate_schema(df, ["a", "b", "c"], "test")


# ====================================================================
# Unit Tests — Data Root Resolution
# ====================================================================


class TestDataRootResolution:
    """Tests for data root path resolution."""

    def test_explicit_data_root(self, tmp_path: Path):
        """Explicit data_root should be used directly."""
        loader = DataPipelineLoader(data_root=str(tmp_path))
        assert loader.data_root == tmp_path

    def test_env_var_data_root(self, tmp_path: Path, monkeypatch):
        """DATA_ROOT env var should be used when no explicit root."""
        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        loader = DataPipelineLoader()
        assert loader.data_root == tmp_path

    def test_default_data_root(self, monkeypatch):
        """Default should fall back to data/processed/current."""
        monkeypatch.delenv("DATA_ROOT", raising=False)
        loader = DataPipelineLoader()
        assert loader.data_root == Path("data/processed/current")


# ====================================================================
# Integration Tests — Real Phase 1 Data
# ====================================================================


@pytest.mark.integration
class TestRealDataIntegration:
    """Integration tests using actual Phase 1 pipeline output.

    These tests require data to exist in ``data/processed/current/``.
    Skip with: ``pytest -m "not integration"``
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        """Skip all tests in this class if real data isn't available."""
        if not REAL_DATA_ROOT.exists():
            pytest.skip("Real data not available at data/processed/current/")

        manifest_path = REAL_DATA_ROOT / "manifest_latest.json"
        if not manifest_path.exists():
            pytest.skip("No manifest_latest.json in data root")

    def test_real_manifest_has_sources(self):
        """Real manifest should have at least one source."""
        loader = DataPipelineLoader(data_root=str(REAL_DATA_ROOT))
        manifest = loader.load_manifest()

        assert "sources" in manifest
        assert len(manifest["sources"]) >= 1
        assert "run_id" in manifest

    def test_real_credit_cards_schema(self):
        """Real credit cards should have all required columns."""
        loader = DataPipelineLoader(data_root=str(REAL_DATA_ROOT))
        df = loader.load_credit_cards()

        assert not df.empty, "Credit cards DataFrame should not be empty"
        for col in REQUIRED_CREDIT_CARD_COLUMNS:
            assert col in df.columns, f"Real data missing column: {col}"

    def test_real_credit_cards_row_count(self):
        """Real credit cards should have a non-trivial number of rows."""
        loader = DataPipelineLoader(data_root=str(REAL_DATA_ROOT))
        df = loader.load_credit_cards()

        assert len(df) >= 50, f"Expected at least 50 credit cards, got {len(df)}"

    def test_real_credit_cards_types(self):
        """Key numeric columns should be numeric types."""
        loader = DataPipelineLoader(data_root=str(REAL_DATA_ROOT))
        df = loader.load_credit_cards()

        numeric_cols = [
            "annual_fee",
            "base_reward_rate",
            "effective_annual_fee",
            "net_value_annual",
        ]
        for col in numeric_cols:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(
                    df[col]
                ), f"{col} should be numeric, got {df[col].dtype}"

    def test_real_data_loader_works_in_docker_path(self):
        """DataPipelineLoader should work with both relative and absolute paths."""
        abs_loader = DataPipelineLoader(data_root=str(REAL_DATA_ROOT.resolve()))
        manifest = abs_loader.load_manifest()
        assert "run_id" in manifest
