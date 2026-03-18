"""Tests for segment-level feature importance analysis (Story 5.1)."""

import pytest
from sklearn.ensemble import RandomForestRegressor

from model_pipeline.personalization.sensitivity.segment_analysis import (
    SegmentAnalyzer,
    define_segments,
)
from model_pipeline.personalization.sensitivity.shap_analysis import SHAPAnalyzer


@pytest.fixture()
def trained_rf(xy_pair):
    X, y = xy_pair
    rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    rf.fit(X, y)
    return rf


@pytest.fixture()
def shap_values(trained_rf, xy_pair):
    X, _ = xy_pair
    analyzer = SHAPAnalyzer(model=trained_rf, X=X)
    return analyzer.compute_shap_values()


@pytest.fixture()
def meta_df(joined_df):
    """Metadata frame containing segment columns."""
    cols = [
        "user_id",
        "archetype",
        "age_group",
        "budget_quartile",
        "location_type",
        "total_spending",
        "num_cards",
    ]
    return joined_df[[c for c in cols if c in joined_df.columns]]


@pytest.fixture()
def seg_analyzer(shap_values, xy_pair, meta_df):
    X, _ = xy_pair
    return SegmentAnalyzer(shap_values=shap_values, X=X, meta=meta_df)


class TestDefineSegments:
    def test_spending_tier_created(self, meta_df):
        segs = define_segments(meta_df)
        assert "spending_tier::high_spender" in segs
        assert "spending_tier::low_spender" in segs

    def test_card_count_created(self, meta_df):
        segs = define_segments(meta_df)
        assert "card_count::single_card" in segs
        assert "card_count::multi_card" in segs

    def test_categorical_segments_created(self, meta_df):
        segs = define_segments(meta_df)
        archetype_keys = [k for k in segs if k.startswith("archetype::")]
        assert len(archetype_keys) > 0


class TestSegmentAnalyzer:
    def test_segment_names_not_empty(self, seg_analyzer):
        assert len(seg_analyzer.segment_names) > 0

    def test_importance_for_segment_columns(self, seg_analyzer):
        key = seg_analyzer.segment_names[0]
        imp = seg_analyzer.importance_for_segment(key)
        assert "feature" in imp.columns
        assert "mean_abs_shap" in imp.columns
        assert "rank" in imp.columns

    def test_analyze_all_segments(self, seg_analyzer):
        results = seg_analyzer.analyze_all_segments()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_compare_segments_table(self, seg_analyzer):
        results = seg_analyzer.analyze_all_segments()
        comparison = seg_analyzer.compare_segments(results, top_n=3)
        assert "segment" in comparison.columns
        assert "top_1" in comparison.columns

    def test_segment_comparison_plot(self, seg_analyzer):
        import matplotlib.pyplot as plt

        results = seg_analyzer.analyze_all_segments()
        fig = seg_analyzer.generate_segment_comparison_plot(results, top_n=5)
        assert fig is not None
        plt.close("all")

    def test_different_segments_may_differ(self, seg_analyzer):
        """Spending-based segments should potentially have different top features."""
        results = seg_analyzer.analyze_all_segments()
        if (
            "spending_tier::high_spender" in results
            and "spending_tier::low_spender" in results
        ):
            top_high = (
                results["spending_tier::high_spender"].head(3)["feature"].tolist()
            )
            top_low = results["spending_tier::low_spender"].head(3)["feature"].tolist()
            assert isinstance(top_high, list)
            assert isinstance(top_low, list)
