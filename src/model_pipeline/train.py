"""
Placeholder for model training entrypoint.

This module will be implemented in later stories (Epic 3).
For now, it serves as the default CMD target for the Docker container
and validates that the model_pipeline package is importable.
"""

import sys


def main():
    """Model training entrypoint placeholder."""
    print("=" * 60)
    print("RewardSense Model Pipeline")
    print("=" * 60)
    print("Model training module loaded successfully.")
    print("Training logic will be implemented in Epic 3 (Stories 3.1-3.5).")
    print()
    print("To run the smoke test instead:")
    print("  python scripts/smoke_test_model_env.py")
    print("=" * 60)

    # ---------------------------------------------------------
    # Integration hooks: save metrics, bias, and mock artifacts
    # to /tmp/model_pipeline/ for DAG components to consume.
    # ---------------------------------------------------------
    import json
    from pathlib import Path

    out_dir = Path("/tmp/model_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"ndcg@10": 0.85, "rmse": 0.45, "run_id": "testsuite1"}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    bias_report = {"metrics": [{"is_biased": False, "value": 0.05}]}
    with open(out_dir / "bias_report.json", "w") as f:
        json.dump(bias_report, f)

    (out_dir / "model_artifact").mkdir(exist_ok=True)
    with open(out_dir / "model_artifact" / "dummy_model.pkl", "w") as f:
        f.write("serialized_model")

    print(f"Integration artifacts written to {out_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
