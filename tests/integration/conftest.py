# tests/integration/conftest.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Any

import pytest
import yaml


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """
    Resolve the repo root (the folder that contains 'src/', 'scripts/', 'dags/', etc.)
    tests/integration/conftest.py -> parents[2] == repo root
    """
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def _ensure_repo_on_syspath(repo_root: Path) -> None:
    """
    Many modules import as `from src....`. That requires the repo root
    (parent of /src) on sys.path.
    """
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@pytest.fixture
def run_cmd(repo_root: Path) -> Callable[..., subprocess.CompletedProcess]:
    """
    Helper to run commands with cwd pinned to repo root.
    """

    def _run(
        args: list[str],
        *,
        env_overrides: Dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        return subprocess.run(
            args,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=check,
        )

    return _run


@pytest.fixture
def processed_current_dir(tmp_path: Path) -> Path:
    """
    Provide an isolated data/processed/current directory inside tmp_path.
    download_data.py writes into: <out-dir>/current
    """
    out_dir = tmp_path / "data" / "processed"
    current = out_dir / "current"
    current.mkdir(parents=True, exist_ok=True)
    return current


@pytest.fixture
def seed_minimal_offers(processed_current_dir: Path) -> Path:
    """
    Create a minimal offers JSON file that is compatible with:
      - Cleaning (needs card_name, issuer, annual_fee)
      - Feature engineering (reward_rates / offers / credits are optional but helpful)
    """
    offers_dir = processed_current_dir / "offers"
    offers_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "source": "integration-test",
        "fetched_at": "2026-02-19T00:00:00Z",
        "offers": [
            {
                "card_id": "test_card_001",
                "card_name": "Test Rewards Card",
                "issuer": "TEST BANK",
                "network": "VISA",
                "currency": "points",
                "annual_fee": 95,
                "is_annual_fee_waived": False,
                "is_business": False,
                "discontinued": False,
                "reward_rates": {"universal_base_rate": 1.5},
                "offers": [{"spend": 1000, "amount": [{"amount": 20000}], "days": 90}],
                "credits": [{"description": "Test credit", "value": 50.0}],
                "universal_cashback_percent": None,
            }
        ],
    }

    out_path = offers_dir / "creditcardbonuses_offers.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


@pytest.fixture
def generate_synthetic_data(
    run_cmd: Callable[..., subprocess.CompletedProcess],
    repo_root: Path,
    tmp_path: Path,
    processed_current_dir: Path,
) -> Path:
    """
    Run scripts/download_data.py in synthetic-only mode into tmp_path.
    This avoids network calls and still produces:
      - synthetic/user_profiles.csv
      - synthetic/user_cards.csv
      - synthetic/transactions.csv
      - manifest_latest.json
    """
    out_dir = tmp_path / "data" / "processed"

    cmd = [
        sys.executable,
        "scripts/download_data.py",
        "--sources",
        "synthetic",
        "--out-dir",
        str(out_dir),
        "--num-users",
        "25",
        "--history-months",
        "2",
        "--seed",
        "123",
        "--log-level",
        "ERROR",
    ]
    res = run_cmd(cmd, check=True)

    # Useful debug if something fails in CI
    assert (
        res.returncode == 0
    ), f"download_data failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Confirm expected files exist
    assert (processed_current_dir / "synthetic" / "user_profiles.csv").exists()
    assert (processed_current_dir / "synthetic" / "transactions.csv").exists()
    assert (processed_current_dir / "manifest_latest.json").exists()

    return processed_current_dir


@pytest.fixture
def transform_config_path(tmp_path: Path, processed_current_dir: Path) -> Path:
    """
    Write a minimal YAML config for TransformationPipeline that points to the
    isolated processed/current directory in tmp_path.
    """
    cfg = {
        "pipeline": {
            "input_root": str(processed_current_dir),
            "output_subdir": "transformed",
            "resume": False,  # integration test: keep it deterministic
            "force_recompute": True,  # avoid checkpoint reuse
        },
        "checkpoints": {"enabled": True, "format": "csv"},
        "datasets": {
            "credit_cards": {
                "enabled": True,
                "load_api_offers": True,
                "api_offers_file": "offers/creditcardbonuses_offers.json",
                "flatten_api_offers": True,
            },
            "transactions": {"enabled": True, "file": "synthetic/transactions.csv"},
            "users": {"enabled": True, "file": "synthetic/user_profiles.csv"},
        },
        "cleaning": {
            # keep defaults; you can override thresholds here if needed
        },
        "logging": {"level": "ERROR"},
    }

    p = tmp_path / "transform_test.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p
