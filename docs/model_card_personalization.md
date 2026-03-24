# Model Card: RewardSense Personalization Model (Story 8.2)

## 1. Model Details

- Model Name: RewardSense Personalization Point-Valuation Model
- Version: `v0.1.0` (see changelog in `docs/model_changelog.md`)
- Owner: RewardSense Model Team
- Primary Task: Regress user-specific point valuation multipliers used by recommendation ranking
- Code:
  - `src/model_pipeline/personalization/models.py`
  - `src/model_pipeline/personalization/trainer.py`
  - `src/model_pipeline/personalization/tuning.py`

## 2. Intended Use

- Intended for reward recommendation optimization in the RewardSense decision pipeline.
- Input:
  - user profile features
  - transaction behavior aggregates
  - engineered interaction signals
- Output:
  - point valuation multiplier applied by `PersonalizedScorer`.

## 3. Out-of-Scope Use

- Not for credit underwriting or risk scoring.
- Not for adverse-action decisions or eligibility denials.
- Not for production use with raw PII.
- Not a stand-alone recommendation without deterministic scoring and policy checks.

## 4. Training Data

- Data source: Phase 1 transformed outputs (`data/processed/current/transformed/*/final`).
- Includes synthetic user/transaction data and merged card catalog artifacts.
- Key features are built in `DatasetBuilder` and `features` modules.
- Known caveat: synthetic distributions may not fully represent real consumer behavior.

## 5. Evaluation

Primary model evaluation logic:

- `src/model_pipeline/personalization/evaluation.py`
- `src/model_pipeline/personalization/validation.py`

Key metrics used in pipeline:

- Regression: RMSE, MAE, R2
- Ranking-oriented checks: NDCG@K (used in downstream gate artifacts)

Validation gate implementation:

- `src/model_pipeline/cd/gates.py`

## 6. Fairness / Bias Evaluation

Bias and fairness modules:

- `src/model_pipeline/bias/slice_evaluator.py`
- `src/model_pipeline/bias/model_bias_detector.py`
- `src/model_pipeline/bias/model_bias_mitigator.py`
- config: `config/bias_slices.yaml`

Current fairness checks include:

- demographic parity difference
- equalized odds difference
- slice-level performance disparity

Mitigation options implemented:

- Exponentiated Gradient
- Threshold Optimizer
- sample reweighting

## 7. Limitations

- Model behavior depends on quality/coverage of synthetic data.
- MLflow and Artifact Registry availability are operational dependencies.
- Personalization fallback defaults may underfit cold-start edge populations.
- Some advanced explainability and sensitivity workflows are optional/manual and not hard-wired in one command.

## 8. Ethical Considerations

- Recommendations can influence user spending behavior; transparency is mandatory.
- Fairness audits should be run before each promotion.
- Generated explanations must not fabricate rates, fees, or benefits.

## 9. Operational Requirements

- Tracking URI and cloud credentials must be configured.
- Registry role permissions must allow read/write for deployment service accounts.
- CI/CD gates must remain enabled for validation + bias checks before push.

## 10. Contacts and Escalation

- Team: RewardSense Model Pipeline Team
- Escalation path: Open issue + CI/CD run links + MLflow run IDs
