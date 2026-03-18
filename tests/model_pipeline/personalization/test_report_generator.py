"""Tests for the sensitivity analysis report generator (Stories 5.1 & 5.2)."""

import json
import tempfile

import pandas as pd
import pytest

from model_pipeline.personalization.sensitivity.report_generator import (
    SensitivityReportGenerator,
)


@pytest.fixture()
def generator():
    return SensitivityReportGenerator(model_name="test-model")


@pytest.fixture()
def populated_generator(generator):
    """A report generator with all sections populated."""
    shap_imp = pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c"],
            "mean_abs_shap": [0.5, 0.3, 0.1],
            "rank": [1, 2, 3],
        }
    )
    generator.add_shap_results(
        global_importance=shap_imp,
        top_features=["feat_a", "feat_b", "feat_c"],
    )

    lime_imp = pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b"],
            "mean_abs_weight": [0.4, 0.2],
            "rank": [1, 2],
        }
    )
    generator.add_lime_results(
        aggregated_importance=lime_imp,
        consistency_check={"spearman_rho": 0.9, "top_5_overlap_pct": 80.0},
    )

    generator.add_segment_results(
        comparison_table=pd.DataFrame(
            {"segment": ["high_spender", "low_spender"], "top_1": ["feat_a", "feat_b"]}
        ),
        segment_count=2,
    )

    generator.add_hp_results(
        {
            "importance_ranking": [
                {"rank": 1, "param": "learning_rate", "abs_correlation": 0.72},
                {"rank": 2, "param": "max_depth", "abs_correlation": 0.45},
            ],
            "safe_ranges": [
                {
                    "name": "learning_rate",
                    "best_value": 0.05,
                    "lower_bound": 0.01,
                    "upper_bound": 0.15,
                },
            ],
        }
    )
    return generator


class TestReportGeneration:
    def test_empty_report_structure(self, generator):
        report = generator.generate()
        assert report["report_type"] == "sensitivity_analysis"
        assert report["model"] == "test-model"
        assert "generated_at" in report
        assert "sections" in report

    def test_populated_report_sections(self, populated_generator):
        report = populated_generator.generate()
        assert "shap" in report["sections"]
        assert "lime" in report["sections"]
        assert "segments" in report["sections"]
        assert "hyperparameters" in report["sections"]

    def test_to_json_valid(self, populated_generator):
        j = populated_generator.to_json()
        parsed = json.loads(j)
        assert parsed["model"] == "test-model"

    def test_to_markdown_contains_headers(self, populated_generator):
        md = populated_generator.to_markdown()
        assert "# Sensitivity Analysis Report" in md
        assert "## SHAP Feature Importance" in md
        assert "## LIME Analysis" in md
        assert "## Segment Analysis" in md
        assert "## Hyperparameter Sensitivity" in md

    def test_to_markdown_contains_data(self, populated_generator):
        md = populated_generator.to_markdown()
        assert "feat_a" in md
        assert "learning_rate" in md
        assert "0.72" in md

    def test_save_creates_files(self, populated_generator):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = populated_generator.save(tmpdir)
            assert paths["json"].exists()
            assert paths["markdown"].exists()
            content = json.loads(paths["json"].read_text())
            assert content["model"] == "test-model"

    def test_shap_section_contents(self, populated_generator):
        report = populated_generator.generate()
        shap = report["sections"]["shap"]
        assert shap["n_features_analyzed"] == 3
        assert shap["top_features"][0] == "feat_a"

    def test_lime_consistency_included(self, populated_generator):
        report = populated_generator.generate()
        lime = report["sections"]["lime"]
        assert lime["consistency_check"]["spearman_rho"] == 0.9

    def test_segment_count(self, populated_generator):
        report = populated_generator.generate()
        assert report["sections"]["segments"]["n_segments"] == 2

    def test_hp_safe_ranges_included(self, populated_generator):
        report = populated_generator.generate()
        hp = report["sections"]["hyperparameters"]
        assert len(hp["safe_ranges"]) == 1
        assert hp["safe_ranges"][0]["name"] == "learning_rate"
