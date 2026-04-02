# %% [markdown]
# # Bias Pipeline Demo
#
# End-to-end demonstration of the model bias detection, drift monitoring,
# counterfactual fairness, and report export pipeline. Trains a real
# XGBoost model on Phase 1 synthetic data, then runs all bias checks.
#
# **Prerequisites:**
# ```bash
# pip install mlflow xgboost shap matplotlib seaborn fairlearn
# # Start MLflow server in a separate terminal:
# mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
# ```
#
# **Run as notebook:**
# ```bash
# pip install jupytext
# jupytext --to notebook notebooks/bias_pipeline_demo.py
# jupyter notebook notebooks/bias_pipeline_demo.ipynb
# ```

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*PyparsingDeprecationWarning.*")

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")

# %%
# --- Load Phase 1 synthetic data ---
# Try loading from DVC-tracked processed outputs first,
# fall back to generating fresh data via Phase 1 generators.

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "current"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"

users_df = None
txns_df = None
cards_df = None

# Attempt 1: Load from Phase 1 pipeline output (post-transform)
for search_dir in [DATA_DIR, GENERATED_DIR, PROJECT_ROOT / "data"]:
    if (search_dir / "user_profiles.csv").exists():
        users_df = pd.read_csv(search_dir / "user_profiles.csv")
        print(f"Loaded users from {search_dir}")
    elif (search_dir / "synthetic" / "user_profiles.csv").exists():
        users_df = pd.read_csv(search_dir / "synthetic" / "user_profiles.csv")
        print(f"Loaded users from {search_dir / 'synthetic'}")

    if (search_dir / "transactions.csv").exists():
        txns_df = pd.read_csv(search_dir / "transactions.csv")
        print(f"Loaded transactions from {search_dir}")
    elif (search_dir / "synthetic" / "transactions.csv").exists():
        txns_df = pd.read_csv(search_dir / "synthetic" / "transactions.csv")
        print(f"Loaded transactions from {search_dir / 'synthetic'}")

    if users_df is not None and txns_df is not None:
        break

# Attempt 2: Generate fresh data via Phase 1 generators
if users_df is None or txns_df is None:
    print("Processed data not found — generating fresh synthetic data...")
    # Generators use internal imports (from data_pipeline...) so add src/ to path
    src_dir = str(PROJECT_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from src.data_pipeline.generators import UserProfileGenerator, TransactionGenerator

    user_gen = UserProfileGenerator(num_users=500, seed=42)
    users_df = user_gen.generate()

    txn_gen = TransactionGenerator(seed=42, history_months=14)
    txns_df = txn_gen.generate(users_df)
    print(f"Generated {len(users_df)} users, {len(txns_df)} transactions")

print(f"\nUsers: {users_df.shape}")
print(f"Transactions: {txns_df.shape}")
print(f"\nUser columns: {list(users_df.columns)}")
print(f"\nArchetype distribution:\n{users_df['archetype'].value_counts()}")

# %% [markdown]
# ## 2. Feature Engineering for Model Training

# %%
# Build ML features from transaction data
# Aggregate per-user spending patterns for the personalization model


def build_user_features(users: pd.DataFrame, txns: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from Phase 1 data."""
    # Per-user transaction aggregates
    user_txn_agg = (
        txns.groupby("user_id")
        .agg(
            total_spend=("amount", "sum"),
            avg_txn=("amount", "mean"),
            txn_count=("amount", "count"),
            n_categories=("category", "nunique"),
            n_merchants=("merchant", "nunique"),
        )
        .reset_index()
    )

    # Category spending distribution (top categories as features)
    cat_spend = (
        txns.groupby(["user_id", "category"])["amount"].sum().unstack(fill_value=0)
    )
    cat_spend.columns = [f"spend_{c}" for c in cat_spend.columns]
    cat_spend = cat_spend.reset_index()

    # Spending diversity (Shannon entropy)
    cat_fracs = (
        txns.groupby(["user_id", "category"])["amount"].sum().unstack(fill_value=0)
    )
    cat_fracs = cat_fracs.div(cat_fracs.sum(axis=1), axis=0).fillna(0)
    entropy = -(cat_fracs * np.log2(cat_fracs.clip(1e-10))).sum(axis=1)
    entropy_df = entropy.reset_index()
    entropy_df.columns = ["user_id", "spending_entropy"]

    # Merge everything
    features = users.merge(user_txn_agg, on="user_id", how="left")
    features = features.merge(cat_spend, on="user_id", how="left")
    features = features.merge(entropy_df, on="user_id", how="left")
    features = features.fillna(0)

    return features


features_df = build_user_features(users_df, txns_df)
print(f"Feature matrix: {features_df.shape}")
print(f"Columns: {list(features_df.columns[:20])}...")

# %% [markdown]
# ## 3. Train XGBoost Personalization Model

# %%
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Target: predict user's top spending category (simulates personalization)
# This is a stand-in for the actual personalization model (Member D's work)
top_cat = txns_df.groupby("user_id")["amount"].apply(
    lambda x: txns_df.loc[x.index, "category"].value_counts().index[0]
)
features_df = features_df.merge(
    top_cat.reset_index().rename(columns={"amount": "top_category"}),
    on="user_id",
    how="left",
)

le_target = LabelEncoder()
features_df["target"] = le_target.fit_transform(
    features_df["top_category"].fillna("other")
)

# Encode categorical features
le_arch = LabelEncoder()
features_df["archetype_encoded"] = le_arch.fit_transform(features_df["archetype"])

le_age = LabelEncoder()
if "age_group" in features_df.columns:
    features_df["age_group_encoded"] = le_age.fit_transform(features_df["age_group"])

le_loc = LabelEncoder()
if "location_type" in features_df.columns:
    features_df["location_encoded"] = le_loc.fit_transform(features_df["location_type"])

# Select numeric features for training
numeric_cols = [
    c
    for c in features_df.columns
    if features_df[c].dtype in [np.float64, np.int64, np.int32]
]
exclude = ["user_id", "target"]
feature_cols = [c for c in numeric_cols if c not in exclude]

X = features_df[feature_cols].values
y = features_df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)
train_idx, test_idx = train_test_split(
    np.arange(len(features_df)),
    test_size=0.25,
    random_state=42,
    stratify=y,
)
test_features_df = features_df.iloc[test_idx].reset_index(drop=True)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {le_target.classes_}")

# %%
# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print("\nModel Performance:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 (weighted): {f1:.4f}")

# %% [markdown]
# ## 4. Initialize MLflow Tracking

# %%
from src.model_pipeline.tracking import RewardSenseTracker

# Connect to your local MLflow server
tracker = RewardSenseTracker(
    experiment="personalization-model",
    tracking_uri="http://localhost:5000",
)

# Create all experiment namespaces
namespaces = tracker.create_all_namespaces()
print(f"MLflow experiments: {namespaces}")

# %% [markdown]
# ## 5. Log Model Training to MLflow

# %%
with tracker.start_run(run_name="xgboost-personalization-v1") as run:
    # Log parameters
    tracker.log_params(
        {
            "model_type": "XGBClassifier",
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_features": len(feature_cols),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": len(le_target.classes_),
        }
    )

    # Log metrics
    tracker.log_metrics(
        {
            "accuracy": acc,
            "f1_weighted": f1,
        }
    )

    # Log model
    tracker.log_model(model, artifact_path="xgboost_model")

    # Log feature names
    tracker.log_dict(
        {"feature_columns": feature_cols},
        "feature_columns.json",
    )

    run_id = tracker.active_run_id
    print(f"\nMLflow Run ID: {run_id}")
    print("View at: http://localhost:5000/#/experiments/")

# %% [markdown]
# ## 6. Run Slice Evaluation (Story 6.1)

# %%
from src.model_pipeline.bias.slice_evaluator import SliceEvaluator

evaluator = SliceEvaluator(
    slicing_config={
        "archetype": {"column": "archetype", "type": "categorical"},
        "age_group": {"column": "age_group", "type": "categorical"},
        "location_type": {"column": "location_type", "type": "categorical"},
        "budget_tier": {
            "column": "total_spend",
            "type": "quantile",
            "n_quantiles": 4,
            "labels": ["low_spend", "mid_low", "mid_high", "high_spend"],
        },
    },
    disparity_threshold=0.10,
)

# Use prediction probabilities for ranking metrics
y_pred_scores = y_pred_proba.max(axis=1)
y_true_binary = (y_test == y_pred).astype(float)

slice_report = evaluator.evaluate(
    test_features_df,
    y_true_binary,
    y_pred_scores,
)

print(f"\n{'='*60}")
print("SLICE EVALUATION REPORT")
print(f"{'='*60}")
print(f"Overall metrics: {slice_report.overall_metrics}")
print(f"Total slices: {len(slice_report.slices)}")
print(f"Disparities found: {len(slice_report.disparities)}")

if slice_report.disparities:
    print("\nDisparities:")
    for d in slice_report.disparities[:5]:
        print(
            f"  {d['slice']}: {d['metric']}={d['slice_value']:.4f} "
            f"(overall={d['overall_value']:.4f}, deviation={d['deviation']:.2%})"
        )

# Log to MLflow
with tracker.start_run(run_name="bias-slice-evaluation"):
    slice_report.log_to_mlflow(tracker)
    print("\nSlice report logged to MLflow (check Artifacts tab)")

# %% [markdown]
# ## 7. Run Model Bias Detection with Fairlearn (Story 6.2)

# %%
from src.model_pipeline.bias.model_bias_detector import (
    ModelBiasDetector,
    ModelBiasConfig,
)

detector = ModelBiasDetector(
    config=ModelBiasConfig(
        demographic_parity_threshold=0.10,
        equalized_odds_threshold=0.10,
        performance_disparity_threshold=0.10,
    )
)

# Run detection across multiple sensitive features
sensitive_features = test_features_df[["archetype", "age_group", "location_type"]]

model_bias_report = detector.detect(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features,
    model_name="xgboost-personalization",
)

print(f"\n{'='*60}")
print("MODEL BIAS DETECTION REPORT")
print(f"{'='*60}")
print(f"Summary: {model_bias_report.summary}")
print("\nAll metrics:")
for m in model_bias_report.metrics:
    status = "FLAGGED" if m.is_biased else "PASS"
    print(
        f"  [{status}] {m.name} ({m.sensitive_feature}): {m.value:.4f} "
        f"(threshold={m.threshold})"
    )

if model_bias_report.per_group_metrics:
    print("\nPer-group breakdowns:")
    for feat, groups in model_bias_report.per_group_metrics.items():
        print(f"  {feat}: {groups}")

# Log to MLflow
with tracker.start_run(run_name="bias-model-detection"):
    model_bias_report.log_to_mlflow(tracker)
    print("\nBias report logged to MLflow (check Artifacts for charts)")

# %% [markdown]
# ## 8. Run Scoring Engine Bias Check (Story 6.3)

# %%
from src.model_pipeline.bias.component_bias import (
    ScoringBiasChecker,
    ExplanationBiasChecker,
)

# Simulate scoring engine recommendations
# (In production, this comes from Member C's RewardCalculator)
rng = np.random.default_rng(42)
recommendations_df = pd.DataFrame(
    {
        "user_id": test_features_df["user_id"].values,
        "archetype": test_features_df["archetype"].values,
        "recommended_card_issuer": rng.choice(
            ["Chase", "Amex", "Capital One", "Citi", "Discover"],
            len(test_features_df),
            p=[0.30, 0.25, 0.20, 0.15, 0.10],
        ),
        "recommended_card_type": rng.choice(
            ["premium", "standard"],
            len(test_features_df),
            p=[0.35, 0.65],
        ),
    }
)

scoring_checker = ScoringBiasChecker(issuer_disparity_threshold=0.15)
scoring_report = scoring_checker.check_issuer_bias(recommendations_df, "archetype")

print(f"\n{'='*60}")
print("SCORING ENGINE BIAS REPORT")
print(f"{'='*60}")
print(f"Summary: {scoring_report.summary}")
for m in scoring_report.metrics:
    status = "FLAGGED" if m.is_biased else "PASS"
    print(f"  [{status}] {m.check_name}: disparity={m.value:.4f}")

with tracker.start_run(run_name="bias-scoring-engine"):
    scoring_report.log_to_mlflow(tracker)

# %% [markdown]
# ## 9. Run LLM Explanation Bias Check (Story 6.3)

# %%
# Simulate LLM explanations
# (In production, this comes from Member E's ExplanationGenerator)
explanations = []
for _, row in test_features_df.iterrows():
    arch = row.get("archetype", "user")
    base = (
        f"Based on your {arch.replace('_', ' ')} spending pattern, "
        f"we recommend this card for its strong {rng.choice(['dining', 'travel', 'grocery'])} rewards. "
        f"You'll earn up to {rng.choice([3, 4, 5])}x points in your top categories."
    )
    explanations.append(base)

explanations_df = pd.DataFrame(
    {
        "user_segment": test_features_df["archetype"].values,
        "explanation_text": explanations,
    }
)

explanation_checker = ExplanationBiasChecker(
    length_disparity_threshold=0.20,
    readability_disparity_threshold=0.15,
)
explanation_report = explanation_checker.check_quality_consistency(
    explanations_df,
    "user_segment",
)

print(f"\n{'='*60}")
print("LLM EXPLANATION BIAS REPORT")
print(f"{'='*60}")
print(f"Summary: {explanation_report.summary}")
for m in explanation_report.metrics:
    status = "FLAGGED" if m.is_biased else "PASS"
    print(f"  [{status}] {m.check_name}: deviation={m.value:.4f}")

with tracker.start_run(run_name="bias-llm-explanations"):
    explanation_report.log_to_mlflow(tracker)

# %% [markdown]
# ## 10. Counterfactual Fairness Analysis (Story 6.9)

# %%
from src.model_pipeline.bias.counterfactual import CounterfactualAnalyzer

# Build a DataFrame-based predict function for the trained model
test_feature_df_numeric = test_features_df[feature_cols].copy()


def model_predict(X):
    if isinstance(X, pd.DataFrame):
        return (
            model.predict_proba(X.values)[:, 1]
            if y_pred_proba.shape[1] > 1
            else model.predict(X.values)
        )
    return model.predict_proba(X)[:, 1]


# We need a DataFrame with both sensitive columns and numeric features
cf_df = test_features_df[
    feature_cols + ["archetype", "age_group", "location_type"]
].copy()


# Custom predict that handles the mixed DataFrame
def cf_predict(X):
    numeric_X = X[feature_cols].values
    return model.predict_proba(numeric_X).max(axis=1)


analyzer = CounterfactualAnalyzer(
    predict_fn=cf_predict,
    flip_threshold=0.05,
)

cf_report = analyzer.analyze_batch(
    cf_df,
    sensitive_columns=["archetype_encoded", "age_group_encoded", "location_encoded"],
    sample_size=min(100, len(cf_df)),
    seed=42,
)

print(f"\n{'='*60}")
print("COUNTERFACTUAL FAIRNESS REPORT")
print(f"{'='*60}")
print(f"Summary: {cf_report.summary}")
print("\nPer-feature sensitivity:")
for feat, rate in cf_report.per_feature_sensitivity.items():
    flag = " ← SENSITIVE" if rate > 0.05 else ""
    print(f"  {feat}: {rate:.1%} of users affected{flag}")

with tracker.start_run(run_name="bias-counterfactual"):
    cf_report.log_to_mlflow(tracker)

# %% [markdown]
# ## 11. Bias Drift Monitoring (Story 6.7)

# %%
from src.model_pipeline.bias.drift_monitor import BiasDriftMonitor

monitor = BiasDriftMonitor(
    history_dir=PROJECT_ROOT / "data" / "bias_history",
    regression_threshold=0.05,
)

# Record current report as v1.0.0
monitor.record(model_bias_report, model_version="1.0.0", model_name="personalization")
print("Recorded bias report for v1.0.0")

# Simulate a "v2.0.0" with slightly different predictions
# (In production, this happens after a retraining cycle)
rng2 = np.random.default_rng(99)
y_pred_v2 = y_pred.copy()
# Introduce slight bias: flip 10% of predictions for one archetype
mask = test_features_df["archetype"] == "young_professional"
flip_idx = rng2.choice(np.where(mask)[0], size=int(mask.sum() * 0.1), replace=False)
y_pred_v2[flip_idx] = (y_pred_v2[flip_idx] + 1) % len(le_target.classes_)

v2_report = detector.detect(
    y_true=y_test,
    y_pred=y_pred_v2,
    sensitive_features=sensitive_features,
    model_name="personalization",
)
monitor.record(v2_report, model_version="2.0.0", model_name="personalization")
print("Recorded bias report for v2.0.0")

# Compare
drift = monitor.compare("1.0.0", "2.0.0", model_name="personalization")

print(f"\n{'='*60}")
print("BIAS DRIFT REPORT: v1.0.0 → v2.0.0")
print(f"{'='*60}")
print(f"Summary: {drift.summary}")

if drift.has_regression:
    print("\n⚠ REGRESSIONS DETECTED:")
    for m in drift.regressions:
        print(
            f"  {m.name} ({m.sensitive_feature}): "
            f"{m.before_value:.4f} → {m.after_value:.4f} "
            f"(+{m.relative_change:.1%})"
        )
else:
    print("\nNo fairness regressions detected.")

if drift.improvements:
    print("\nImprovements:")
    for m in drift.improvements:
        print(
            f"  {m.name} ({m.sensitive_feature}): "
            f"{m.before_value:.4f} → {m.after_value:.4f} "
            f"({m.relative_change:+.1%})"
        )

with tracker.start_run(run_name="bias-drift-v1-to-v2"):
    drift.log_to_mlflow(tracker)

# Plot trend
fig = monitor.plot_trend(
    "demographic_parity_difference",
    "archetype",
    model_name="personalization",
    threshold=0.10,
)
fig.savefig(PROJECT_ROOT / "reports" / "drift_trend.png", dpi=150, bbox_inches="tight")
print("\nDrift trend chart saved to reports/drift_trend.png")

# %% [markdown]
# ## 12. Export Full HTML Report (Story 6.8)

# %%
from src.model_pipeline.bias.report_export import BiasReportExporter

exporter = BiasReportExporter()

# Full report with all components
report_path = exporter.export_full_report(
    model_report=model_bias_report,
    scoring_report=scoring_report,
    explanation_report=explanation_report,
    slice_report=slice_report,
    output_path=PROJECT_ROOT / "reports" / "full_bias_report.html",
    title="RewardSense — Phase 2 Bias Analysis (XGBoost v1.0.0)",
)

print(f"\n{'='*60}")
print("FULL BIAS REPORT EXPORTED")
print(f"{'='*60}")
print(f"HTML: {report_path}")
print(f"Open in browser: file://{report_path.resolve()}")

# Individual reports
for name, report in [
    ("model_bias", model_bias_report),
    ("scoring_bias", scoring_report),
    ("explanation_bias", explanation_report),
]:
    p = exporter.export_html(
        report,
        output_path=PROJECT_ROOT / "reports" / f"{name}_report.html",
    )
    print(f"  {name}: {p}")

# %% [markdown]
# ## 13. Push Model to Registry (Story 1.2)

# %%
from src.model_pipeline.registry.artifact_registry import RegistryClient
import joblib

# Save model locally
model_dir = PROJECT_ROOT / "models" / "personalization"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "xgboost_v1.pkl"
joblib.dump(model, model_path)

# Push to registry (local cache if no GCP credentials)
registry = RegistryClient(
    project="rewardsense-prod",
    location="us-central1",
    repository="rewardsense-models",
    local_cache=PROJECT_ROOT / ".model_cache",
)

mv = registry.push_model(
    model_path,
    model_name="personalization",
    version="1.0.0",
    metadata={
        "accuracy": acc,
        "f1_weighted": f1,
        "n_features": len(feature_cols),
        "bias_flagged": len(model_bias_report.biased_metrics),
        "drift_regressions": len(drift.regressions),
    },
)
print(f"\nModel pushed to registry: {mv.tag}")
print(f"SHA-256: {mv.sha256}")

# %% [markdown]
# ## 14. Summary
#
# **What just happened:**
# 1. Loaded Phase 1 synthetic data (500 users, ~300K transactions)
# 2. Built ML features (spending patterns, category distributions, entropy)
# 3. Trained XGBoost classifier (personalization model stand-in)
# 4. **Slice Evaluation** — computed NDCG/Precision/Recall per data slice
# 5. **Fairlearn Bias Detection** — demographic parity, equalized odds, performance disparity
# 6. **Scoring Engine Bias** — checked issuer/card-type distribution fairness
# 7. **LLM Explanation Bias** — checked explanation quality consistency
# 8. **Counterfactual Fairness** — measured prediction sensitivity to demographic flips
# 9. **Bias Drift Monitoring** — compared v1.0.0 vs v2.0.0, flagged regressions
# 10. **HTML Report Export** — stakeholder-ready report with embedded charts
# 11. **Model Registry** — pushed model with bias metadata
#
# **Check MLflow UI:** http://localhost:5000
# - Experiment: `personalization-model`
# - Each run has: params, metrics, JSON artifacts, PNG chart artifacts
#
# **Open the HTML report:** `reports/full_bias_report.html`

# %%
print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print(f"{'='*60}")
print("\nMLflow UI:    http://localhost:5000")
print(
    f"HTML Report:  file://{(PROJECT_ROOT / 'reports' / 'full_bias_report.html').resolve()}"
)
print(
    f"Drift Charts: file://{(PROJECT_ROOT / 'reports' / 'drift_trend.png').resolve()}"
)
print("\nRun IDs logged — check MLflow Artifacts tab for:")
print("  - bias_summary_*.png")
print("  - fairness_groups_*.png")
print("  - slice_metrics.png")
print("  - slice_disparity_heatmap.png")
print("  - counterfactual_sensitivity.png")
print("  - bias_drift_*.png")
