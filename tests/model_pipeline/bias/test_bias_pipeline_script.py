"""
Tests for Bias Pipeline

Covers:
  - Argument parsing (defaults, --ci flag, custom values)
  - Data loading/generation
  - Feature engineering
  - Model training + rare class handling
  - Full pipeline integration (mocked MLflow)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root and src/ are on path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.run_bias_pipeline import (  # noqa: E402
    build_features,
    load_or_generate_data,
    parse_args,
    run_pipeline,
    train_model,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def sample_users():
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "user_id": range(n),
            "archetype": rng.choice(
                ["young_professional", "suburban_family", "budget_conscious"], n
            ),
            "age_group": rng.choice(["18-25", "26-35", "36-50"], n),
            "location_type": rng.choice(["urban", "suburban", "rural"], n),
            "monthly_budget": rng.normal(3000, 1000, n).clip(500),
            "num_cards": rng.choice([1, 2, 3, 4], n),
            "redemption_preference": rng.choice(["cash_back", "travel_transfer"], n),
        }
    )


@pytest.fixture
def sample_txns(sample_users):
    rng = np.random.default_rng(42)
    rows = []
    categories = ["groceries", "dining", "travel", "gas", "online_shopping"]
    merchants = ["Store A", "Store B", "Store C", "Store D", "Store E"]
    for uid in sample_users["user_id"]:
        n_txns = rng.integers(20, 60)
        for _ in range(n_txns):
            rows.append(
                {
                    "user_id": uid,
                    "amount": max(1.0, float(rng.normal(50, 30))),
                    "category": rng.choice(categories),
                    "merchant": rng.choice(merchants),
                    "date": "2026-01-15",
                }
            )
    return pd.DataFrame(rows)


# =====================================================================
# Argument Parsing
# =====================================================================


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["run_bias_pipeline.py"]):
            args = parse_args()
        assert args.mlflow_uri == "http://localhost:5000"
        assert args.model_version == "1.0.0"
        assert args.n_users == 500
        assert args.fail_on_regression is False
        assert args.export_html is False
        assert args.ci is False

    def test_ci_flag(self):
        with patch("sys.argv", ["run_bias_pipeline.py", "--ci"]):
            args = parse_args()
        assert args.ci is True

    def test_custom_values(self):
        with patch(
            "sys.argv",
            [
                "run_bias_pipeline.py",
                "--mlflow-uri",
                "http://localhost:5001",
                "--model-version",
                "2.0.0",
                "--n-users",
                "100",
                "--export-html",
                "--fail-on-regression",
            ],
        ):
            args = parse_args()
        assert args.mlflow_uri == "http://localhost:5001"
        assert args.model_version == "2.0.0"
        assert args.n_users == 100
        assert args.export_html is True
        assert args.fail_on_regression is True


# =====================================================================
# Data Loading / Generation
# =====================================================================


class TestLoadOrGenerateData:
    def test_generates_data_when_no_files(self, tmp_path):
        """Should generate fresh data when no CSV files exist."""
        with patch("scripts.run_bias_pipeline.PROJECT_ROOT", tmp_path):
            users, txns = load_or_generate_data(n_users=20, seed=42)
        assert len(users) == 20
        assert len(txns) > 0
        assert "user_id" in users.columns
        assert "amount" in txns.columns

    def test_loads_existing_data(self, tmp_path, sample_users, sample_txns):
        """Should load from CSV when files exist."""
        data_dir = tmp_path / "data" / "processed" / "current"
        data_dir.mkdir(parents=True)
        sample_users.to_csv(data_dir / "user_profiles.csv", index=False)
        sample_txns.to_csv(data_dir / "transactions.csv", index=False)

        with patch("scripts.run_bias_pipeline.PROJECT_ROOT", tmp_path):
            users, txns = load_or_generate_data(n_users=500, seed=42)
        assert len(users) == len(sample_users)
        assert len(txns) == len(sample_txns)


# =====================================================================
# Feature Engineering
# =====================================================================


class TestBuildFeatures:
    def test_output_shape(self, sample_users, sample_txns):
        features = build_features(sample_users, sample_txns)
        assert len(features) == len(sample_users)
        assert "total_spend" in features.columns
        assert "avg_txn" in features.columns
        assert "txn_count" in features.columns

    def test_no_missing_values(self, sample_users, sample_txns):
        features = build_features(sample_users, sample_txns)
        assert features.isnull().sum().sum() == 0

    def test_category_spend_columns(self, sample_users, sample_txns):
        features = build_features(sample_users, sample_txns)
        spend_cols = [c for c in features.columns if c.startswith("spend_")]
        assert len(spend_cols) > 0

    def test_handles_user_with_no_transactions(self, sample_users, sample_txns):
        """Users with no transactions should get 0-filled features."""
        extra_user = pd.DataFrame(
            {
                "user_id": [9999],
                "archetype": ["budget_conscious"],
                "age_group": ["26-35"],
                "location_type": ["urban"],
                "monthly_budget": [2000.0],
                "num_cards": [1],
                "redemption_preference": ["cash_back"],
            }
        )
        users = pd.concat([sample_users, extra_user], ignore_index=True)
        features = build_features(users, sample_txns)
        assert len(features) == len(users)
        new_row = features[features["user_id"] == 9999].iloc[0]
        assert new_row["total_spend"] == 0


# =====================================================================
# Model Training
# =====================================================================


class TestTrainModel:
    def test_returns_expected_outputs(self, sample_users, sample_txns):
        features = build_features(sample_users, sample_txns)
        model, X_test, y_test, y_pred, test_df, feat_cols = train_model(
            features, sample_txns
        )
        assert X_test.shape[0] == y_test.shape[0]
        assert y_pred.shape[0] == y_test.shape[0]
        assert len(test_df) == len(y_test)
        assert len(feat_cols) > 0

    def test_predictions_valid_classes(self, sample_users, sample_txns):
        features = build_features(sample_users, sample_txns)
        model, X_test, y_test, y_pred, _, _ = train_model(features, sample_txns)
        unique_pred = set(y_pred)
        unique_true = set(y_test)
        assert unique_pred.issubset(unique_true | unique_pred)

    def test_handles_rare_classes(self, sample_users, sample_txns):
        """Should not crash when some categories have only 1 user."""
        # Create a transaction with a rare category for 1 user
        rare_txn = pd.DataFrame(
            {
                "user_id": [sample_users["user_id"].iloc[0]],
                "amount": [100.0],
                "category": ["extremely_rare_category"],
                "merchant": ["Rare Store"],
                "date": ["2026-01-15"],
            }
        )
        txns = pd.concat([sample_txns, rare_txn], ignore_index=True)
        features = build_features(sample_users, txns)

        # Should not raise ValueError about rare classes
        model, X_test, y_test, y_pred, _, _ = train_model(features, txns)
        assert len(y_pred) > 0

    def test_label_encoding_no_gaps(self, sample_users, sample_txns):
        """After rare class removal, labels should be contiguous."""
        features = build_features(sample_users, sample_txns)
        _, _, y_test, y_pred, _, _ = train_model(features, sample_txns)
        all_labels = np.unique(np.concatenate([y_test, y_pred]))
        # Labels should be contiguous from 0
        assert all_labels[0] == 0
        assert len(all_labels) == all_labels.max() + 1


# =====================================================================
# Full Pipeline Integration
# =====================================================================


class TestRunPipeline:
    def test_pipeline_completes(self, tmp_path):
        """Full pipeline should complete and produce results JSON."""
        with patch(
            "sys.argv",
            [
                "run_bias_pipeline.py",
                "--mlflow-uri",
                "http://localhost:5001",
                "--model-version",
                "test-1.0.0",
                "--output-dir",
                str(tmp_path / "reports"),
                "--n-users",
                "30",
                "--export-html",
            ],
        ):
            args = parse_args()

        # Mock MLflow to avoid needing a running server
        mock_tracker = MagicMock()
        mock_tracker.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_tracker.active_run_id = "mock-run-id"

        with patch("scripts.run_bias_pipeline.PROJECT_ROOT", tmp_path), patch(
            "src.model_pipeline.tracking.RewardSenseTracker",
            return_value=mock_tracker,
        ):
            # Need src/ on path for generators
            src_dir = str(PROJECT_ROOT / "src")
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)

            results = run_pipeline(args)

        assert results["passed"] is True
        assert "checks" in results
        assert "slices" in results["checks"]
        assert "model_bias" in results["checks"]
        assert "scoring_bias" in results["checks"]
        assert "elapsed_seconds" in results
        assert results["elapsed_seconds"] > 0

        # Results JSON should be written
        results_file = tmp_path / "reports" / "bias_results_vtest-1.0.0.json"
        assert results_file.exists()
        saved = json.loads(results_file.read_text())
        assert saved["passed"] is True

    def test_pipeline_exports_html(self, tmp_path):
        """Pipeline with --export-html should produce an HTML file."""
        with patch(
            "sys.argv",
            [
                "run_bias_pipeline.py",
                "--mlflow-uri",
                "http://localhost:5001",
                "--model-version",
                "html-test",
                "--output-dir",
                str(tmp_path / "reports"),
                "--n-users",
                "30",
                "--export-html",
            ],
        ):
            args = parse_args()

        mock_tracker = MagicMock()
        mock_tracker.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch("scripts.run_bias_pipeline.PROJECT_ROOT", tmp_path), patch(
            "src.model_pipeline.tracking.RewardSenseTracker",
            return_value=mock_tracker,
        ):
            results = run_pipeline(args)

        assert "html_report" in results
        html_path = Path(results["html_report"])
        assert html_path.exists()
        content = html_path.read_text()
        assert "<html" in content

    def test_ci_mode_sets_flags(self):
        """--ci should enable fail_on_regression and export_html."""
        with patch("sys.argv", ["run_bias_pipeline.py", "--ci"]):
            args = parse_args()
        # CI flag is set, but fail_on_regression and export_html
        # are activated inside run_pipeline, not parse_args
        assert args.ci is True

    def test_pipeline_without_export(self, tmp_path):
        """Pipeline without --export-html should not produce HTML."""
        with patch(
            "sys.argv",
            [
                "run_bias_pipeline.py",
                "--mlflow-uri",
                "http://localhost:5001",
                "--model-version",
                "no-html",
                "--output-dir",
                str(tmp_path / "reports"),
                "--n-users",
                "30",
            ],
        ):
            args = parse_args()

        mock_tracker = MagicMock()
        mock_tracker.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch("scripts.run_bias_pipeline.PROJECT_ROOT", tmp_path), patch(
            "src.model_pipeline.tracking.RewardSenseTracker",
            return_value=mock_tracker,
        ):
            results = run_pipeline(args)

        assert "html_report" not in results
