# Model Pipeline Quality Audit Report (Story 8.3)

Audit date: 2026-03-24

## Commands Executed

```bash
.venv/bin/ruff check src/model_pipeline tests/model_pipeline scripts/check_model_coverage_threshold.py
.venv/bin/python -m mypy src/model_pipeline --ignore-missing-imports
.venv/bin/pytest -q
.venv/bin/pytest tests/model_pipeline --override-ini="addopts=" --cov=src/model_pipeline --cov-report=xml:/tmp/model_pipeline_coverage.xml --cov-report=term-missing
.venv/bin/python scripts/check_model_coverage_threshold.py /tmp/model_pipeline_coverage.xml --threshold 80
```

## Results

- Ruff lint: PASS
- Mypy type check: PASS (`Success: no issues found in 47 source files`)
- Full test suite: PASS (`1177 passed, 10 skipped`)
- Model pipeline test suite: PASS (`492 passed, 8 skipped`)
- Model pipeline total coverage: 83%
- Per-module >=80% gate: FAIL (15 modules below threshold)
- Docker build verification: BLOCKED (Docker daemon unavailable at `~/.docker/run/docker.sock`)

## Modules Below 80% Coverage

- `train.py` (21.7%)
- `cd/notifier.py` (59.1%)
- `bias/model_bias_mitigator.py` (66.9%)
- `personalization/sensitivity/shap_analysis.py` (67.3%)
- `registry/artifact_registry.py` (67.3%)
- `scoring/spending_cap_tracker.py` (69.7%)
- `personalization/tuning.py` (71.6%)
- `tracking.py` (75.3%)
- `personalization/sensitivity/hyperparameter_sensitivity.py` (76.7%)
- `personalization/sensitivity/lime_analysis.py` (77.4%)
- `scoring/merchant_mapper.py` (77.8%)
- `data_loader.py` (77.9%)
- `bias/drift_monitor.py` (78.9%)
- `bias/report_export.py` (79.0%)
- `llm/vertex_gemini_client.py` (79.2%)

## Actions Completed in This Audit

- Added type-check fixes across model modules.
- Added coverage threshold checker script (`scripts/check_model_coverage_threshold.py`).
- Added one-command audit script (`scripts/model_pipeline_quality_audit.sh`).
- Fixed `BiasGate` parsing to handle `all_metrics` reports.

## Remaining Work to Fully Satisfy Story 8.3 Acceptance

- Raise all listed modules to >=80% line coverage.
- Run Docker image build verification after Docker daemon is available.
