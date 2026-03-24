# RewardSense Model Version Changelog (Story 8.2)

This changelog tracks model-facing versions across scoring, personalization, and explainability components.

## Versioning Convention

- Format: `major.minor.patch`
- Registry tag format: `<model-name>-v<version>-<timestamp>`

## Versions

## v0.1.0 (2026-03-24)

- Added deterministic scoring core + ranking + validation harness
- Added personalization training/tuning/validation pipeline
- Added LLM explainability prompt/generation/validation modules
- Added model bias detection + mitigation modules
- Added CD gates, notifier, rollback scaffolding
- Added model pipeline DAG orchestration

## v0.1.1 (planned)

- Harden Artifact Registry remote pull path to download all artifacts
- Expand per-module coverage to >80% for all model modules
- Convert CD registry push step from placeholder to enforced promotion step

## Key Registry Artifacts

- Model package: `personalization`
- Repository: `rewardsense-models`
- Region: `us-central1`

## Release Checklist

- [ ] MLflow run IDs linked in experiment report
- [ ] Bias/validation gate outcomes attached
- [ ] Registry version tag recorded
- [ ] Rollback reference version recorded
