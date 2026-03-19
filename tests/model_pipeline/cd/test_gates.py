import json
import tempfile
from pathlib import Path

from src.model_pipeline.cd.gates import ValidationGate, BiasGate


def test_validation_gate_pass():
    gate = ValidationGate({"ndcg_at_5": 0.75, "rmse": 0.50})
    # If the user minimizes RMSE, they might need a custom check or negative metrics,
    # but the logic given is "metric >= min_val". If RMSE needs to be < 0.50, they should use negative RMSE.
    assert gate.evaluate({"ndcg_at_5": 0.80, "rmse": 0.60}) is True


def test_validation_gate_fail():
    gate = ValidationGate({"ndcg_at_5": 0.75, "precision_at_5": 0.80})
    assert gate.evaluate({"ndcg_at_5": 0.80, "precision_at_5": 0.70}) is False


def test_bias_gate_pass():
    gate = BiasGate(max_disparity=0.10)

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        json.dump(
            {
                "metrics": [
                    {
                        "name": "demographic_parity",
                        "is_biased": False,
                        "value": 0.05,
                        "sensitive_feature": "age",
                    },
                    {
                        "name": "equalized_odds",
                        "is_biased": True,
                        "value": 0.09,
                        "sensitive_feature": "location",
                    },
                ]
            },
            f,
        )
        temp_path = f.name

    try:
        assert gate.evaluate(temp_path) is True
    finally:
        Path(temp_path).unlink()


def test_bias_gate_fail():
    gate = BiasGate(max_disparity=0.10)

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        json.dump(
            {
                "metrics": [
                    {
                        "name": "equalized_odds",
                        "is_biased": True,
                        "value": 0.15,
                        "sensitive_feature": "group",
                    }
                ]
            },
            f,
        )
        temp_path = f.name

    try:
        assert gate.evaluate(temp_path) is False
    finally:
        Path(temp_path).unlink()


def test_bias_gate_missing_file():
    gate = BiasGate(max_disparity=0.10)
    assert gate.evaluate("/path/does/not/exist.json") is False
