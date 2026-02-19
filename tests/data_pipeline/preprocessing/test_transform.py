# tests/test_transform_pipeline.py
"""
Unit tests for src/data_pipeline/preprocessing/transform.py

Given / When / Then style.

Notes:
- These tests avoid hitting real filesystem outside tmp_path.
- They stub/mimic cleaning + feature engineering so tests are deterministic and fast.
- They validate checkpoint/resume behavior, audit artifacts, config hashing,
  JSON sanitization, and dataframe hashing robustness.

Run:
  pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import yaml

# Import the module under test
import src.data_pipeline.preprocessing.transform as tr


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def minimal_config(
    input_root: Path, resume: bool = True, force: bool = False
) -> Dict[str, Any]:
    return {
        "version": 1,
        "pipeline": {
            "input_root": str(input_root),
            "output_subdir": "transformed",
            "resume": resume,
            "force_recompute": force,
        },
        "datasets": {
            "credit_cards": {
                "enabled": True,
                "load_api_offers": True,
                "load_issuer_offers": False,
                "load_nerdwallet_offers": False,
                "api_offers_file": "offers/creditcardbonuses_offers.json",
                "flatten_api_offers": True,
                "annual_spending": 25000,
            },
            "transactions": {"enabled": True, "file": "synthetic/transactions.csv"},
            "users": {"enabled": True, "file": "synthetic/user_profiles.csv"},
        },
        "cleaning": {
            "max_annual_fee": 1000.0,
            "min_annual_fee": 0.0,
            "min_transaction_amount": 0.0,
            "suspicious_amount_threshold": 10000.0,
            "validate_mcc": True,
        },
        "checkpoints": {"enabled": True, "format": "csv"},
        "logging": {"level": "INFO"},
    }


def seed_input_artifacts(root: Path) -> None:
    """Create minimal input artifacts that _step_load expects."""
    offers_dir = root / "offers"
    synth_dir = root / "synthetic"
    offers_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    # CreditCardBonuses-like payload
    payload = {
        "offers": [
            {
                "source": "creditcardbonuses",
                "card_id": "c1",
                "card_name": "Card One",
                "issuer": "CHASE",
                "annual_fee": 95.0,
                "reward_rates": {"universal_base_rate": 1.5},
                "offers": [
                    {
                        "spend": 4000,
                        "amount": [{"amount": 60000}],
                        "days": 90,
                        "credits": [],
                    }
                ],
                "credits": [
                    {"description": "Test Credit", "value": 100, "weight": 0.5}
                ],
                "discontinued": False,
                "currency": "POINTS",
                "network": "VISA",
            }
        ]
    }
    (offers_dir / "creditcardbonuses_offers.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Transactions
    txns = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "user_id": ["u1", "u1"],
            "date": ["2026-02-01", "2026-02-02"],
            "category": ["dining", "groceries"],
            "merchant": ["M1", "M2"],
            "mcc_code": [5812, 5411],
            "amount": [10.0, 20.0],
            "card_used": ["Card One", "Card One"],
        }
    )
    txns.to_csv(synth_dir / "transactions.csv", index=False)

    # Users
    users = pd.DataFrame(
        {
            "user_id": ["u1"],
            "archetype": ["optimizer"],
            "monthly_budget": [3000],
            "cards": ["['Card One']"],
            "redemption_preference": ["travel_transfer"],
            "age_group": ["26-35"],
            "location_type": ["urban"],
        }
    )
    users.to_csv(synth_dir / "user_profiles.csv", index=False)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def input_root(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "processed" / "current"
    seed_input_artifacts(root)
    return root


@pytest.fixture
def cfg_path(tmp_path: Path, input_root: Path) -> Path:
    cfg = minimal_config(input_root=input_root, resume=True, force=False)
    p = tmp_path / "configs" / "transform.yaml"
    write_yaml(p, cfg)
    return p


@pytest.fixture
def stub_clean_all_data(monkeypatch):
    """
    Given cleaning is a dependency,
    When pipeline calls clean_all_data(),
    Then return deterministic cleaned dfs + report.
    """

    def _fake_clean_all_data(
        credit_cards_df: Optional[pd.DataFrame],
        transactions_df: Optional[pd.DataFrame],
        users_df: Optional[pd.DataFrame],
        config: Any,
    ) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Dict[str, Any],
    ]:
        # passthrough but add a tiny change to prove it ran
        cards = credit_cards_df.copy() if credit_cards_df is not None else None
        txns = transactions_df.copy() if transactions_df is not None else None
        users = users_df.copy() if users_df is not None else None
        if cards is not None:
            cards["cleaned_flag"] = 1
        if txns is not None:
            txns["cleaned_flag"] = 1
        if users is not None:
            users["cleaned_flag"] = 1

        report = {
            "credit_cards": {"rows": int(len(cards)) if cards is not None else 0},
            "transactions": {"rows": int(len(txns)) if txns is not None else 0},
            "users": {"rows": int(len(users)) if users is not None else 0},
        }
        return cards, txns, users, report

    monkeypatch.setattr(tr, "clean_all_data", _fake_clean_all_data)


@pytest.fixture
def stub_engineer_all_features(monkeypatch):
    """
    Given feature engineering is a dependency,
    When pipeline calls engineer_all_features(),
    Then return deterministic engineered dfs.
    """

    def _fake_engineer_all_features(
        credit_cards_df: Optional[pd.DataFrame] = None,
        transactions_df: Optional[pd.DataFrame] = None,
        users_df: Optional[pd.DataFrame] = None,
        annual_spending: float = 25000,
        output_dir: Optional[Path] = None,
    ):
        cards_f = None
        txns_f = None
        users_f = None

        if credit_cards_df is not None:
            cards_f = credit_cards_df.copy()
            cards_f["base_reward_rate"] = 1.5
            cards_f["net_value_annual"] = 123.0

        if transactions_df is not None:
            # user-level aggregated row
            txns_f = pd.DataFrame(
                {
                    "user_id": sorted(transactions_df["user_id"].unique()),
                    "total_spending": [float(transactions_df["amount"].sum())],
                }
            )

        if users_df is not None:
            users_f = users_df.copy()
            users_f["estimated_point_value"] = 0.02

        return cards_f, txns_f, users_f

    monkeypatch.setattr(tr, "engineer_all_features", _fake_engineer_all_features)


# -----------------------------------------------------------------------------
# Tests: Utilities
# -----------------------------------------------------------------------------


def test_json_sanitize_handles_sets_paths_numpy_and_pandas():
    """
    Given an object containing non-JSON types (set, Path, numpy scalars, pandas Timestamp)
    When json_sanitize is called
    Then it returns a JSON-serializable structure.
    """
    obj = {
        "a_set": {"b", "a"},
        "a_path": Path("/tmp/x"),
        "npi": np.int64(7),
        "npf": np.float64(3.14),
        "npb": np.bool_(True),
        "ts": pd.Timestamp("2026-02-18"),
        "arr": np.array([1, 2, 3]),
        "nested": {"tup": (1, {"z"})},
    }

    out = tr.json_sanitize(obj)
    # Ensure serializable
    json.dumps(out)
    assert out["a_set"] == ["a", "b"]
    assert out["a_path"] == "/tmp/x"
    assert out["npi"] == 7
    assert abs(out["npf"] - 3.14) < 1e-9
    assert out["npb"] is True
    assert isinstance(out["ts"], str)
    assert out["arr"] == [1, 2, 3]
    assert out["nested"]["tup"][1] == ["z"]


def test_df_hash_is_stable_and_handles_object_cells():
    """
    Given a dataframe with object cells including dict/list/set/ndarray
    When df_hash is computed multiple times
    Then it is stable and does not raise.
    """
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [{"k": 1}, {"k": 2}],
            "c": [[1, 2], np.array([3, 4])],
            "d": [{"x", "y"}, set()],
        }
    )
    h1 = tr.df_hash(df)
    h2 = tr.df_hash(df)
    assert isinstance(h1, str) and len(h1) == 64
    assert h1 == h2


# -----------------------------------------------------------------------------
# Tests: Pipeline end-to-end
# -----------------------------------------------------------------------------


def test_pipeline_runs_end_to_end_and_writes_artifacts(
    cfg_path: Path,
    input_root: Path,
    stub_clean_all_data,
    stub_engineer_all_features,
    monkeypatch,
):
    """
    Given valid input artifacts and stubs for cleaning and feature engineering
    When TransformationPipeline.run() executes
    Then it completes end-to-end and writes checkpoints, final outputs, and audit logs.
    """
    # Given
    pipeline = tr.TransformationPipeline(cfg_path)

    # When
    outputs = pipeline.run()

    # Then: returns dfs
    assert outputs["credit_cards_features"] is not None
    assert outputs["transactions_features"] is not None
    assert outputs["users_features"] is not None

    # Then: final artifacts exist
    assert (pipeline.final_dir / "credit_cards_features.csv").exists()
    assert (pipeline.final_dir / "transactions_features.csv").exists()
    assert (pipeline.final_dir / "users_features.csv").exists()

    # Then: checkpoints exist with sentinels
    assert (pipeline.ckpt_dir / "01_loaded" / "_DONE").exists()
    assert (pipeline.ckpt_dir / "02_cleaned" / "_DONE").exists()
    assert (pipeline.ckpt_dir / "03_features" / "_DONE").exists()

    # Then: audit artifacts exist
    assert (pipeline.audit_dir / "audit.json").exists()
    assert (pipeline.audit_dir / "step_reports.json").exists()

    audit = read_json(pipeline.audit_dir / "audit.json")
    assert audit["run_id"] == pipeline.run_id
    assert audit["config_sha256"] == pipeline.config_sha256
    assert "01_loaded" in audit["steps"]
    assert "02_cleaned" in audit["steps"]
    assert "03_features" in audit["steps"]

    steps = audit["steps"]
    assert steps["01_loaded"]["report_path"].endswith(
        "checkpoints/01_loaded/load_report.json"
    )
    assert steps["02_cleaned"]["report_path"].endswith(
        "checkpoints/02_cleaned/clean_report.json"
    )
    assert steps["03_features"]["report_path"].endswith(
        "checkpoints/03_features/features_report.json"
    )


def test_pipeline_resume_uses_checkpoints_without_recomputing(
    cfg_path: Path,
    stub_clean_all_data,
    stub_engineer_all_features,
    monkeypatch,
):
    """
    Given a completed prior run with checkpoints
    When a second pipeline run reuses the same run output directory with resume=true
    Then it loads from checkpoints and does NOT call underlying load/clean/FE functions.
    """
    # -------------------------
    # Given: first run creates checkpoints
    # -------------------------
    p1 = tr.TransformationPipeline(cfg_path)
    _ = p1.run()

    assert (p1.ckpt_dir / "01_loaded" / "_DONE").exists()
    assert (p1.ckpt_dir / "02_cleaned" / "_DONE").exists()
    assert (p1.ckpt_dir / "03_features" / "_DONE").exists()

    # -------------------------
    # Given: second pipeline instance pointed at SAME prior run dirs
    # -------------------------
    p2 = tr.TransformationPipeline(cfg_path)

    # Force p2 to reuse p1's run folder/checkpoints
    p2.run_id = p1.run_id
    p2.output_root = p1.output_root
    p2.ckpt_dir = p1.ckpt_dir
    p2.final_dir = p1.final_dir
    p2.audit_dir = p1.audit_dir

    # -------------------------
    # When: make recompute paths bomb if called
    # -------------------------
    def bomb(*args, **kwargs):
        raise AssertionError("Should not recompute when checkpoints exist")

    monkeypatch.setattr(p2, "_load_credit_cards", bomb)
    monkeypatch.setattr(p2, "_load_transactions", bomb)
    monkeypatch.setattr(p2, "_load_users", bomb)

    # IMPORTANT: patch the names as imported in transform.py (tr.*), not original modules
    monkeypatch.setattr(tr, "clean_all_data", bomb)
    monkeypatch.setattr(tr, "engineer_all_features", bomb)

    # Run steps directly (avoid rewriting audit.json in same run folder)
    cards2, txns2, users2, _ = p2._step_load()
    clean_cards2, clean_txns2, clean_users2, _ = p2._step_clean(cards2, txns2, users2)
    cards_f2, txns_f2, users_f2, _ = p2._step_features(
        clean_cards2, clean_txns2, clean_users2
    )

    # -------------------------
    # Then: we got feature outputs from checkpoint loads
    # -------------------------
    assert cards_f2 is not None
    assert txns_f2 is not None
    assert users_f2 is not None


def test_pipeline_force_recompute_ignores_checkpoints_and_recomputes(
    cfg_path: Path,
    input_root: Path,
    stub_clean_all_data,
    stub_engineer_all_features,
):
    """
    Given existing checkpoints from a previous run
    When force_recompute=true
    Then the pipeline recomputes steps rather than using checkpoints.
    """
    # Given: first run to create checkpoints
    p1 = tr.TransformationPipeline(cfg_path)
    _ = p1.run()

    # Given: config with force_recompute true
    cfg2 = minimal_config(input_root=input_root, resume=True, force=True)
    cfg2_path = cfg_path.parent / "transform_force.yaml"
    write_yaml(cfg2_path, cfg2)

    # When: run again (it will produce a new run_id and new output_root)
    p2 = tr.TransformationPipeline(cfg2_path)
    out = p2.run()

    # Then
    assert out["credit_cards_features"] is not None
    assert (p2.ckpt_dir / "01_loaded" / "_DONE").exists()
    assert (p2.ckpt_dir / "02_cleaned" / "_DONE").exists()
    assert (p2.ckpt_dir / "03_features" / "_DONE").exists()

    audit = read_json(p2.audit_dir / "audit.json")
    assert audit["steps"]["01_loaded"]["used_checkpoint"] is False
    assert audit["steps"]["02_cleaned"]["used_checkpoint"] is False
    assert audit["steps"]["03_features"]["used_checkpoint"] is False


def test_pipeline_handles_missing_datasets_gracefully(
    tmp_path: Path,
    input_root: Path,
    stub_clean_all_data,
    stub_engineer_all_features,
):
    """
    Given config disabling transactions and users datasets
    When the pipeline runs
    Then it still completes and only writes outputs for enabled datasets.
    """
    # Given
    cfg = minimal_config(input_root=input_root, resume=True, force=False)
    cfg["datasets"]["transactions"]["enabled"] = False
    cfg["datasets"]["users"]["enabled"] = False

    cfg_path = tmp_path / "configs" / "transform.yaml"
    write_yaml(cfg_path, cfg)

    # When
    pipeline = tr.TransformationPipeline(cfg_path)
    out = pipeline.run()

    # Then
    assert out["credit_cards_features"] is not None
    assert out["transactions_features"] is None
    assert out["users_features"] is None

    assert (pipeline.final_dir / "credit_cards_features.csv").exists()
    assert not (pipeline.final_dir / "transactions_features.csv").exists()
    assert not (pipeline.final_dir / "users_features.csv").exists()

    step_reports = read_json(pipeline.audit_dir / "step_reports.json")
    assert "final_outputs" in step_reports
    assert "credit_cards_features" in step_reports["final_outputs"]
    assert "transactions_features" not in step_reports["final_outputs"]
    assert "users_features" not in step_reports["final_outputs"]


def test_step_reports_include_hashes_and_shapes(
    cfg_path: Path,
    stub_clean_all_data,
    stub_engineer_all_features,
):
    """
    Given the pipeline runs end-to-end
    When step_reports.json is written
    Then it includes cleaning hashes and feature shapes/hashes for auditing.
    """
    # Given
    pipeline = tr.TransformationPipeline(cfg_path)

    # When
    pipeline.run()

    # Then
    step_reports = read_json(pipeline.audit_dir / "step_reports.json")
    assert "clean" in step_reports
    assert "features" in step_reports

    clean = step_reports["clean"]
    assert "hashes" in clean
    assert "credit_cards_clean" in clean["hashes"]
    assert "transactions_clean" in clean["hashes"]
    assert "users_clean" in clean["hashes"]

    fe = step_reports["features"]
    assert "shapes" in fe
    assert "hashes" in fe
    assert "credit_cards_features" in fe["hashes"]


def test_atomic_write_json_can_write_non_serializable_structures(tmp_path: Path):
    """
    Given a dict containing a set and numpy scalars
    When atomic_write_json is called
    Then it writes a valid JSON file without raising.
    """
    # Given
    obj = {"a": {3, 1, 2}, "b": np.int64(9), "c": np.array([1, 2])}
    dest = tmp_path / "x.json"

    # When
    tr.atomic_write_json(dest, obj)

    # Then
    loaded = read_json(dest)
    assert loaded["a"] == [1, 2, 3]
    assert loaded["b"] == 9
    assert loaded["c"] == [1, 2]


def test_config_sha256_changes_when_config_changes(tmp_path: Path, input_root: Path):
    """
    Given two configs that differ in annual_spending
    When the pipeline is initialized
    Then config_sha256 differs, enabling versioned auditability.
    """
    # Given
    cfg1 = minimal_config(input_root=input_root)
    cfg2 = minimal_config(input_root=input_root)
    cfg2["datasets"]["credit_cards"]["annual_spending"] = 30000

    p1 = tmp_path / "c1.yaml"
    p2 = tmp_path / "c2.yaml"
    write_yaml(p1, cfg1)
    write_yaml(p2, cfg2)

    # When
    t1 = tr.TransformationPipeline(p1)
    t2 = tr.TransformationPipeline(p2)

    # Then
    assert t1.config_sha256 != t2.config_sha256
