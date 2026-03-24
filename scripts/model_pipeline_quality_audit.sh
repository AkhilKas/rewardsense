#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PYTEST_BIN="${PYTEST_BIN:-.venv/bin/pytest}"
RUFF_BIN="${RUFF_BIN:-.venv/bin/ruff}"

COV_XML="/tmp/model_pipeline_coverage.xml"
COV_THRESHOLD="${COV_THRESHOLD:-80}"

echo "[1/6] Ruff lint (model pipeline + tests)"
"$RUFF_BIN" check src/model_pipeline tests/model_pipeline

echo "[2/6] Mypy type check"
"$PYTHON_BIN" -m mypy src/model_pipeline --ignore-missing-imports

echo "[3/6] Model pipeline test suite"
"$PYTEST_BIN" -q tests/model_pipeline

echo "[4/6] Full project test suite"
"$PYTEST_BIN" -q

echo "[5/6] Coverage (src/model_pipeline)"
"$PYTEST_BIN" tests/model_pipeline \
  --override-ini="addopts=" \
  --cov=src/model_pipeline \
  --cov-report=term-missing \
  --cov-report=xml:"$COV_XML"

echo "[5b/6] Per-module coverage threshold check >= ${COV_THRESHOLD}%"
"$PYTHON_BIN" scripts/check_model_coverage_threshold.py "$COV_XML" --threshold "$COV_THRESHOLD"

echo "[6/6] Docker build verification"
if command -v docker >/dev/null 2>&1; then
  docker build -f Dockerfile.model -t rewardsense-model:epic8-audit .
  docker build -f Dockerfile.mlflow -t rewardsense-mlflow:epic8-audit .
else
  echo "docker not installed; skipping Docker verification" >&2
fi

echo "Epic 8 quality audit completed successfully."
