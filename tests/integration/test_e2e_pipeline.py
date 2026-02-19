# tests/integration/test_pipeline_e2e.py
from __future__ import annotations


import subprocess
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_pipeline_end_to_end_produces_expected_outputs(
    generate_synthetic_data: Path,
    seed_minimal_offers: Path,
    transform_config_path: Path,
):
    """
    End-to-end integration test:

    Given:
      - synthetic datasets under data/processed/current/synthetic
      - minimal offers file under data/processed/current/offers
    When:
      - TransformationPipeline runs
    Then:
      - transformed/<run_id>/final/*.csv exist
      - transformed/<run_id>/audit/audit.json exists
      - outputs are non-empty and have expected high-level structure
    """
    # Import here (after sys.path is fixed by conftest)
    from src.data_pipeline.preprocessing.transform import TransformationPipeline

    pipeline = TransformationPipeline(config_path=transform_config_path)
    outputs = pipeline.run()

    # Output root is always: <input_root>/transformed/<run_id>
    out_root = pipeline.output_root
    final_dir = out_root / "final"
    audit_dir = out_root / "audit"

    assert final_dir.exists(), f"Missing final dir: {final_dir}"
    assert audit_dir.exists(), f"Missing audit dir: {audit_dir}"
    assert (audit_dir / "audit.json").exists(), "Missing audit.json"
    assert (audit_dir / "step_reports.json").exists(), "Missing step_reports.json"

    # Expected final artifacts
    cc_path = final_dir / "credit_cards_features.csv"
    tx_path = final_dir / "transactions_features.csv"
    user_path = final_dir / "users_features.csv"

    assert cc_path.exists(), "Missing credit_cards_features.csv"
    assert tx_path.exists(), "Missing transactions_features.csv"
    assert user_path.exists(), "Missing users_features.csv"

    # Sanity checks: non-empty, parseable CSV
    cc = pd.read_csv(cc_path)
    tx = pd.read_csv(tx_path)
    users = pd.read_csv(user_path)

    assert len(cc) >= 1, "credit cards features should not be empty"
    assert len(tx) >= 1, "transactions features should not be empty"
    assert len(users) >= 1, "users features should not be empty"

    # Data flows between stages: weak-but-useful checks
    # (features should contain engineered fields)
    assert "base_reward_rate" in cc.columns, "Expected engineered credit card feature"
    expected = {
        "user_id",
        "total_spending",
        "total_transactions",
        "avg_transaction_amount",
    }
    assert expected.issubset(set(tx.columns))
    assert any(c.lower().startswith("user") for c in users.columns) or (
        "user_id" in users.columns
    ), "Expected user identifiers in users_features"

    # And pipeline should return a dict of outputs (current contract: feature DataFrames)
    assert isinstance(outputs, dict)
    required_keys = {"credit_cards_features", "transactions_features", "users_features"}
    assert required_keys.issubset(set(outputs.keys())), (
        f"Pipeline outputs missing keys: {required_keys - set(outputs.keys())}. "
        f"Got keys: {sorted(outputs.keys())}"
    )


@pytest.mark.integration
def test_dvc_tracking_integration_optional(tmp_path: Path):
    """
    DVC integration test (optional):
    - Creates an isolated git repo
    - Runs `dvc init`
    - Writes a dummy artifact and runs `dvc add`
    - Verifies the .dvc file is created

    Skips if DVC is not installed.
    """
    pytest.importorskip("dvc", reason="dvc python package not installed")

    # But we also need the CLI for real integration.
    try:
        subprocess.run(["dvc", "--version"], capture_output=True, text=True, check=True)
    except Exception:
        pytest.skip("dvc CLI not available")

    repo = tmp_path / "dvc_repo"
    repo.mkdir(parents=True, exist_ok=True)

    def r(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, check=True
        )

    # init git + dvc
    r(["git", "init"])
    r(["dvc", "init", "-q"])

    # Create a dummy artifact
    artifact_dir = repo / "data" / "transformed" / "run_001" / "final"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "credit_cards_features.csv"
    artifact_path.write_text("card_id,base_reward_rate\nx,1.5\n", encoding="utf-8")

    # dvc add
    r(["dvc", "add", str(artifact_path)])

    # Verify .dvc file exists
    dvc_file = artifact_path.with_suffix(artifact_path.suffix + ".dvc")
    assert dvc_file.exists(), "Expected DVC sidecar file to be created"


# @pytest.mark.integration
@pytest.mark.integration
def test_airflow_dag_parsing_optional():
    """
    Airflow integration test (optional):
    - Verifies the DAG loads without import errors
    - Verifies a minimal set of task IDs exist (TaskGroup-safe)

    Skips if Airflow is not installed.
    """
    pytest.importorskip("airflow", reason="Airflow not installed in this environment")

    from airflow.models import DagBag  # type: ignore

    dagbag = DagBag(dag_folder="dags", include_examples=False)

    # If there are import errors, fail with details (super helpful)
    assert not dagbag.import_errors, f"DAG import errors: {dagbag.import_errors}"

    dag_id = "rewardsense_data_pipeline"
    assert (
        dag_id in dagbag.dags
    ), f"DAG '{dag_id}' not found. Known dags: {sorted(dagbag.dags.keys())}"

    dag = dagbag.dags[dag_id]
    task_ids = sorted(dag.task_ids)

    # Sanity: DAG should have tasks
    assert len(task_ids) > 0, "DAG has no tasks"

    # ✅ Minimal set to validate your DAG "so far"
    # Add/remove items here to reflect what is ACTUALLY defined today.
    required_suffixes = {
        # Keep this minimal until Story 5.x wires everything in:
        "run_transform_pipeline",
        # If these exist already, keep them; otherwise remove them for now:
        # "generate_synthetic_data",
        # "clean_data",
        # "engineer_features",
    }

    missing = {
        sfx for sfx in required_suffixes if not any(t.endswith(sfx) for t in task_ids)
    }

    assert not missing, (
        f"Missing required tasks (by suffix): {missing}\n"
        f"Actual task_ids:\n{task_ids}"
    )
