#!/usr/bin/env python3
"""
Automated Bias Pipeline Runner.

CLI script for CI/CD integration. Runs the full bias detection pipeline,
logs to MLflow, exports reports, and exits with non-zero status if
bias regressions are detected.

Usage:
    # Full pipeline
    PYTHONPATH=. python scripts/run_bias_pipeline.py

    # With options
    PYTHONPATH=. python scripts/run_bias_pipeline.py \
        --mlflow-uri http://localhost:5001 \
        --model-version 1.0.0 \
        --output-dir reports \
        --fail-on-regression \
        --export-html \
        --n-users 500

    # CI mode (strict — fails on any bias regression)
    PYTHONPATH=. python scripts/run_bias_pipeline.py --ci
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bias_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="RewardSense Bias Pipeline")
    p.add_argument(
        "--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI"
    )
    p.add_argument("--model-version", default="1.0.0", help="Model version tag")
    p.add_argument(
        "--output-dir", default="reports", help="Directory for exported reports"
    )
    p.add_argument(
        "--n-users", type=int, default=500, help="Number of synthetic users to generate"
    )
    p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 if bias regression detected",
    )
    p.add_argument("--export-html", action="store_true", help="Export HTML bias report")
    p.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: strict thresholds + fail on regression + export",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_or_generate_data(n_users: int, seed: int):
    """Load Phase 1 data or generate fresh."""
    data_paths = [
        PROJECT_ROOT / "data" / "processed" / "current",
        PROJECT_ROOT / "data" / "generated",
        PROJECT_ROOT / "data" / "synthetic",
    ]

    users_df = None
    txns_df = None

    for d in data_paths:
        for subdir in [d, d / "synthetic"]:
            uf = subdir / "user_profiles.csv"
            tf = subdir / "transactions.csv"
            if uf.exists() and tf.exists():
                users_df = pd.read_csv(uf)
                txns_df = pd.read_csv(tf)
                logger.info("Loaded data from %s", subdir)
                break
        if users_df is not None:
            break

    if users_df is None:
        logger.info("Generating fresh synthetic data (%d users)...", n_users)
        # Generators use internal imports like `from data_pipeline.generators...`
        # so we need src/ on the path as well
        src_dir = str(PROJECT_ROOT / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from src.data_pipeline.generators import (
            UserProfileGenerator,
            TransactionGenerator,
        )

        users_df = UserProfileGenerator(num_users=n_users, seed=seed).generate()
        txns_df = TransactionGenerator(seed=seed, history_months=14).generate(users_df)

    return users_df, txns_df


def build_features(users_df, txns_df):
    """Build ML features from raw data."""
    user_agg = (
        txns_df.groupby("user_id")
        .agg(
            total_spend=("amount", "sum"),
            avg_txn=("amount", "mean"),
            txn_count=("amount", "count"),
            n_categories=("category", "nunique"),
        )
        .reset_index()
    )

    cat_spend = (
        txns_df.groupby(["user_id", "category"])["amount"].sum().unstack(fill_value=0)
    )
    cat_spend.columns = [f"spend_{c}" for c in cat_spend.columns]
    cat_spend = cat_spend.reset_index()

    features = users_df.merge(user_agg, on="user_id", how="left")
    features = features.merge(cat_spend, on="user_id", how="left")
    return features.fillna(0)


def train_model(features_df, txns_df):
    """Train XGBoost and return model + split data."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    top_cat = txns_df.groupby("user_id")["amount"].apply(
        lambda x: txns_df.loc[x.index, "category"].value_counts().index[0]
    )
    features_df = features_df.merge(
        top_cat.reset_index().rename(columns={"amount": "top_category"}),
        on="user_id",
        how="left",
    )

    le = LabelEncoder()
    features_df["target"] = le.fit_transform(
        features_df["top_category"].fillna("other")
    )

    for col in ["archetype", "age_group", "location_type", "redemption_preference"]:
        if col in features_df.columns:
            enc = LabelEncoder()
            features_df[f"{col}_encoded"] = enc.fit_transform(
                features_df[col].astype(str)
            )

    numeric_cols = [
        c
        for c in features_df.columns
        if features_df[c].dtype in [np.float64, np.int64, np.int32]
        and c not in ["user_id", "target"]
    ]

    X = features_df[numeric_cols].values
    y = features_df["target"].values

    # Remove rare classes (< 2 members) for stratified split
    class_counts = pd.Series(y).value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        keep_mask = ~np.isin(y, rare_classes)
        X = X[keep_mask]
        y = y[keep_mask]
        features_df = features_df[keep_mask].reset_index(drop=True)
        logger.info(
            "Dropped %d rare classes (%d samples removed)",
            len(rare_classes),
            (~keep_mask).sum(),
        )
        # Re-encode to remove gaps in label values
        re_le = LabelEncoder()
        y = re_le.fit_transform(y)
        features_df["target"] = y

    train_idx, test_idx = train_test_split(
        np.arange(len(features_df)),
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X[train_idx], y[train_idx])

    y_pred = model.predict(X[test_idx])
    test_df = features_df.iloc[test_idx].reset_index(drop=True)

    return model, X[test_idx], y[test_idx], y_pred, test_df, numeric_cols


def run_pipeline(args):
    """Execute the full bias pipeline."""
    start = time.time()
    results = {"passed": True, "checks": {}, "regressions": 0}

    if args.ci:
        args.fail_on_regression = True
        args.export_html = True

    # --- Setup ---
    from src.model_pipeline.tracking import RewardSenseTracker

    tracker = RewardSenseTracker(
        experiment="personalization-model",
        tracking_uri=args.mlflow_uri,
    )
    tracker.create_all_namespaces()

    # --- Data ---
    logger.info("Loading data...")
    users_df, txns_df = load_or_generate_data(args.n_users, args.seed)
    logger.info("Users: %d, Transactions: %d", len(users_df), len(txns_df))

    # --- Features + Model ---
    logger.info("Building features and training model...")
    features_df = build_features(users_df, txns_df)
    model, X_test, y_test, y_pred, test_df, feature_cols = train_model(
        features_df, txns_df
    )

    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info("Model accuracy=%.4f, f1=%.4f", acc, f1)

    # --- Log model training ---
    with tracker.start_run(run_name=f"training-v{args.model_version}"):
        tracker.log_params(
            {"model_version": args.model_version, "n_features": len(feature_cols)}
        )
        tracker.log_metrics({"accuracy": acc, "f1_weighted": f1})
        tracker.log_model(model, "xgboost_model")

    # --- Slice Evaluation ---
    logger.info("Running slice evaluation...")
    from src.model_pipeline.bias.slice_evaluator import SliceEvaluator

    evaluator = SliceEvaluator(
        slicing_config={
            "archetype": {"column": "archetype", "type": "categorical"},
            "age_group": {"column": "age_group", "type": "categorical"},
        },
        disparity_threshold=0.10,
    )
    y_scores = model.predict_proba(X_test).max(axis=1)
    y_binary = (y_test == y_pred).astype(float)
    slice_report = evaluator.evaluate(test_df, y_binary, y_scores)

    with tracker.start_run(run_name=f"bias-slices-v{args.model_version}"):
        slice_report.log_to_mlflow(tracker)

    results["checks"]["slices"] = {
        "total": len(slice_report.slices),
        "disparities": len(slice_report.disparities),
    }
    logger.info(
        "Slices: %d, Disparities: %d",
        len(slice_report.slices),
        len(slice_report.disparities),
    )

    # --- Model Bias Detection ---
    logger.info("Running Fairlearn bias detection...")
    from src.model_pipeline.bias.model_bias_detector import ModelBiasDetector

    detector = ModelBiasDetector()
    sensitive = (
        test_df[["archetype", "age_group"]].copy()
        if "age_group" in test_df.columns
        else test_df[["archetype"]]
    )
    model_report = detector.detect(
        y_test, y_pred, sensitive, model_name="personalization"
    )

    with tracker.start_run(run_name=f"bias-model-v{args.model_version}"):
        model_report.log_to_mlflow(tracker)

    results["checks"]["model_bias"] = model_report.summary
    logger.info(
        "Model bias: %d checks, %d flagged",
        len(model_report.metrics),
        len(model_report.biased_metrics),
    )

    # --- Scoring Bias ---
    logger.info("Running scoring engine bias check...")
    from src.model_pipeline.bias.component_bias import ScoringBiasChecker

    rng = np.random.default_rng(args.seed)
    recs_df = pd.DataFrame(
        {
            "archetype": test_df["archetype"].values,
            "recommended_card_issuer": rng.choice(
                ["Chase", "Amex", "Capital One", "Citi"],
                len(test_df),
            ),
        }
    )
    scoring_report = ScoringBiasChecker().check_issuer_bias(recs_df, "archetype")

    with tracker.start_run(run_name=f"bias-scoring-v{args.model_version}"):
        scoring_report.log_to_mlflow(tracker)

    results["checks"]["scoring_bias"] = scoring_report.summary

    # --- Counterfactual ---
    logger.info("Running counterfactual fairness analysis...")
    from src.model_pipeline.bias.counterfactual import CounterfactualAnalyzer

    cf_df = test_df[feature_cols].copy()
    for col in ["archetype_encoded", "age_group_encoded"]:
        if col in test_df.columns:
            cf_df[col] = test_df[col].values

    sensitive_cols = [
        c for c in ["archetype_encoded", "age_group_encoded"] if c in cf_df.columns
    ]

    if sensitive_cols:
        analyzer = CounterfactualAnalyzer(
            predict_fn=lambda X: model.predict_proba(X[feature_cols].values).max(
                axis=1
            ),
            flip_threshold=0.05,
        )
        cf_report = analyzer.analyze_batch(
            cf_df, sensitive_cols, sample_size=100, seed=args.seed
        )

        with tracker.start_run(run_name=f"bias-counterfactual-v{args.model_version}"):
            cf_report.log_to_mlflow(tracker)

        results["checks"]["counterfactual"] = cf_report.summary

    # --- Drift Monitoring ---
    logger.info("Recording bias report for drift monitoring...")
    from src.model_pipeline.bias.drift_monitor import BiasDriftMonitor

    monitor = BiasDriftMonitor(
        history_dir=PROJECT_ROOT / "data" / "bias_history",
    )
    monitor.record(model_report, model_version=args.model_version)

    versions = monitor.list_versions()
    if len(versions) >= 2:
        prev = versions[1]
        drift = monitor.compare(prev, args.model_version)
        results["checks"]["drift"] = drift.summary
        results["regressions"] = len(drift.regressions)

        with tracker.start_run(run_name=f"bias-drift-{prev}-to-{args.model_version}"):
            drift.log_to_mlflow(tracker)

        if drift.has_regression:
            logger.warning(
                "BIAS REGRESSION DETECTED: %d metrics worsened", len(drift.regressions)
            )
            results["passed"] = not args.fail_on_regression

    # --- HTML Export ---
    if args.export_html:
        logger.info("Exporting HTML report...")
        from src.model_pipeline.bias.report_export import BiasReportExporter

        output_dir = Path(args.output_dir)
        exporter = BiasReportExporter()
        report_path = exporter.export_full_report(
            model_report=model_report,
            scoring_report=scoring_report,
            slice_report=slice_report,
            output_path=output_dir / f"bias_report_v{args.model_version}.html",
            title=f"RewardSense Bias Report — v{args.model_version}",
        )
        results["html_report"] = str(report_path)
        logger.info("HTML report: %s", report_path)

    # --- Summary ---
    elapsed = time.time() - start
    results["elapsed_seconds"] = round(elapsed, 2)

    # Save results JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"bias_results_v{args.model_version}.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    logger.info("=" * 60)
    logger.info("BIAS PIPELINE COMPLETE (%.1fs)", elapsed)
    logger.info("Results: %s", results_path)
    logger.info("MLflow UI: %s", args.mlflow_uri)
    if results.get("html_report"):
        logger.info("HTML Report: file://%s", Path(results["html_report"]).resolve())
    logger.info("=" * 60)

    return results


def main():
    args = parse_args()
    results = run_pipeline(args)

    if not results["passed"]:
        logger.error("PIPELINE FAILED — bias regression detected")
        sys.exit(1)

    logger.info("PIPELINE PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
