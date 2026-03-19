#!/usr/bin/env python3
"""
Smoke Test: Model Training Environment Validation.

Validates that the Docker container (or local environment) has all
required dependencies, configuration, and connectivity for model
training. Designed to run inside the model-training container:

    docker compose run --rm model-training python scripts/smoke_test_model_env.py

Exit codes:
    0 — All checks passed
    1 — One or more checks failed
"""

import importlib
import os
import sys

# ---- Configuration ----
REQUIRED_PYTHON_VERSION = (3, 9)

REQUIRED_PACKAGES = [
    "xgboost",
    "lightgbm",
    "sklearn",
    "mlflow",
    "shap",
    "lime",
    "fairlearn",
    "optuna",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "yaml",
    "dotenv",
    "pydantic",
    "loguru",
    "joblib",
    "tqdm",
]

REQUIRED_ENV_VARS = [
    "EXECUTION_ENV",
    "MLFLOW_TRACKING_URI",
]

CONFIG_FILES = [
    "config/model_config.yaml",
]


def check_python_version():
    """Check Python version meets minimum requirement."""
    current = sys.version_info[:2]
    if current >= REQUIRED_PYTHON_VERSION:
        return True, f"Python {current[0]}.{current[1]}"
    return (
        False,
        f"Python {current[0]}.{current[1]} (need >= {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]})",
    )


def check_packages():
    """Check all required packages are importable."""
    results = []
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            results.append((True, pkg, version))
        except ImportError as e:
            results.append((False, pkg, str(e)))
    return results


def check_project_importable():
    """Check that the model_pipeline package is importable."""
    try:
        import model_pipeline

        return True, f"v{model_pipeline.__version__}"
    except ImportError as e:
        return False, str(e)


def check_env_vars():
    """Check required environment variables are set."""
    results = []
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            results.append((True, var, value))
        else:
            results.append((False, var, "NOT SET"))
    return results


def check_config_files():
    """Check required config files are accessible."""
    results = []
    for path in CONFIG_FILES:
        if os.path.isfile(path):
            results.append((True, path))
        else:
            results.append((False, path))
    return results


def check_mlflow_connectivity():
    """Check MLflow server is reachable (if URI is set)."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return None, "MLFLOW_TRACKING_URI not set, skipping"

    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        # Try listing experiments as a connectivity check
        mlflow.search_experiments()
        return True, f"Connected to {uri}"
    except Exception as e:
        return False, f"Cannot connect to {uri}: {e}"


def main():
    """Run all smoke test checks."""
    print("=" * 60)
    print("  RewardSense Model Environment Smoke Test")
    print("=" * 60)
    print()

    all_passed = True

    # 1. Python version
    print("[1/6] Python Version")
    ok, detail = check_python_version()
    print(f"  {'✅' if ok else '❌'} {detail}")
    if not ok:
        all_passed = False
    print()

    # 2. Required packages
    print("[2/6] Required Packages")
    pkg_results = check_packages()
    for ok, pkg, detail in pkg_results:
        print(f"  {'✅' if ok else '❌'} {pkg}: {detail}")
        if not ok:
            all_passed = False
    print()

    # 3. Project importable
    print("[3/6] Project Package (model_pipeline)")
    ok, detail = check_project_importable()
    print(f"  {'✅' if ok else '❌'} model_pipeline: {detail}")
    if not ok:
        all_passed = False
    print()

    # 4. Config files
    print("[4/6] Config Files")
    cfg_results = check_config_files()
    for ok, path in cfg_results:
        print(f"  {'✅' if ok else '❌'} {path}")
        if not ok:
            all_passed = False
    print()

    # 5. Environment variables
    print("[5/6] Environment Variables")
    env_results = check_env_vars()
    for ok, var, value in env_results:
        print(f"  {'✅' if ok else '❌'} {var}={value}")
        if not ok:
            all_passed = False
    print()

    # 6. MLflow connectivity
    print("[6/6] MLflow Connectivity")
    ok, detail = check_mlflow_connectivity()
    if ok is None:
        print(f"  ⏭️  {detail}")
    elif ok:
        print(f"  ✅ {detail}")
    else:
        print(f"  ⚠️  {detail} (non-blocking)")
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("  ✅ ALL CHECKS PASSED — environment is ready!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("  ❌ SOME CHECKS FAILED — see details above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
