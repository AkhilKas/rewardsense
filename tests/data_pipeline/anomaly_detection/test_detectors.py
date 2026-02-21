"""
Unit tests for anomaly detection engine.

Acceptance criteria:
  - Anomalies detected and logged automatically
  - Configurable thresholds for each check
  - Historical anomaly tracking (via AnomalyReport.to_json)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.anomaly_detection.detectors import (
    Anomaly,
    AnomalyConfig,
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def detector():
    return AnomalyDetector()


@pytest.fixture
def strict_detector():
    """Detector with tight thresholds for testing."""
    cfg = AnomalyConfig(
        missing_warning_threshold=0.01,
        missing_critical_threshold=0.05,
        outlier_ratio_warning=0.01,
        outlier_ratio_critical=0.05,
    )
    return AnomalyDetector(config=cfg)


@pytest.fixture
def clean_transactions():
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i}" for i in range(n)],
            "user_id": [f"u{i % 20}" for i in range(n)],
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "category": np.random.choice(
                ["dining", "groceries", "gas", "travel", "online_shopping"], n
            ),
            "merchant": [f"merchant_{i % 30}" for i in range(n)],
            "mcc_code": np.random.choice([5812, 5411, 5541, 4511, 5691], n),
            "amount": np.random.normal(100, 40, n).clip(1),
            "card_used": np.random.choice(["Card A", "Card B", "Card C"], n),
        }
    )


@pytest.fixture
def dirty_transactions():
    """Transactions with deliberate quality issues."""
    n = 100
    amounts = np.random.normal(100, 40, n)
    amounts[:25] = np.nan  # 25% missing → CRITICAL
    categories = ["dining"] * 60 + [None] * 10 + ["groceries"] * 30

    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i}" for i in range(n)],
            "user_id": [f"u{i % 5}" for i in range(n)],
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "category": categories,
            "merchant": [f"m_{i}" for i in range(n)],
            "mcc_code": list(range(n)),
            "amount": amounts,
            "card_used": ["Card A"] * n,
        }
    )


# =====================================================================
# Anomaly / AnomalyReport data structures
# =====================================================================


class TestAnomalyDataStructures:
    def test_anomaly_to_dict(self):
        a = Anomaly(
            check_name="test",
            severity=AnomalySeverity.WARNING,
            message="Test message",
            dataset="transactions",
            column="amount",
        )
        d = a.to_dict()
        assert d["severity"] == "WARNING"
        assert d["check_name"] == "test"

    def test_report_summary(self):
        report = AnomalyReport(dataset="test")
        report.anomalies = [
            Anomaly("c1", AnomalySeverity.CRITICAL, "msg1"),
            Anomaly("c2", AnomalySeverity.WARNING, "msg2"),
            Anomaly("c3", AnomalySeverity.INFO, "msg3"),
            Anomaly("c4", AnomalySeverity.WARNING, "msg4"),
        ]
        assert report.summary == {"CRITICAL": 1, "WARNING": 2, "INFO": 1}
        assert report.has_critical is True
        assert report.max_severity == AnomalySeverity.CRITICAL

    def test_report_no_anomalies(self):
        report = AnomalyReport(dataset="clean")
        assert report.has_critical is False
        assert report.has_warnings is False
        assert report.max_severity == AnomalySeverity.INFO
        assert report.summary == {"CRITICAL": 0, "WARNING": 0, "INFO": 0}

    def test_report_to_json(self, tmp_path):
        report = AnomalyReport(dataset="test", total_checks=3)
        report.anomalies = [Anomaly("c1", AnomalySeverity.WARNING, "msg")]
        out = tmp_path / "report.json"
        report.to_json(out)

        loaded = json.loads(out.read_text())
        assert loaded["dataset"] == "test"
        assert loaded["anomaly_count"] == 1
        assert loaded["total_checks"] == 3

    def test_report_to_dict(self):
        report = AnomalyReport(dataset="test", total_checks=2)
        d = report.to_dict()
        assert "anomalies" in d
        assert "summary" in d
        assert d["dataset"] == "test"


# =====================================================================
# Missing value detection
# =====================================================================


class TestMissingValueDetection:
    def test_no_missing_values(self, detector, clean_transactions):
        anomalies = detector.detect_missing_values(
            clean_transactions, dataset="transactions"
        )
        assert len(anomalies) == 0

    def test_warning_level_missing(self, detector):
        n = 100
        df = pd.DataFrame(
            {
                "a": [1.0] * 90 + [np.nan] * 10,  # 10% missing
                "b": list(range(n)),
            }
        )
        anomalies = detector.detect_missing_values(df, dataset="test")
        assert len(anomalies) == 1
        assert anomalies[0].severity == AnomalySeverity.WARNING
        assert anomalies[0].column == "a"

    def test_critical_level_missing(self, detector):
        n = 100
        df = pd.DataFrame(
            {
                "a": [1.0] * 70 + [np.nan] * 30,  # 30% missing
                "b": list(range(n)),
            }
        )
        anomalies = detector.detect_missing_values(df, dataset="test")
        critical = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]
        assert len(critical) == 1

    def test_empty_dataframe(self, detector):
        df = pd.DataFrame({"a": pd.Series(dtype=float)})
        anomalies = detector.detect_missing_values(df)
        assert len(anomalies) == 0

    def test_custom_thresholds(self, strict_detector):
        _n = 100
        df = pd.DataFrame({"a": [1.0] * 97 + [np.nan] * 3})  # 3% missing
        anomalies = strict_detector.detect_missing_values(df)
        # 3% > 1% warning threshold for strict detector
        assert len(anomalies) == 1
        assert anomalies[0].severity == AnomalySeverity.WARNING


# =====================================================================
# Outlier detection — IQR
# =====================================================================


class TestOutlierDetectionIQR:
    def test_no_outliers_in_clean_data(self, detector, clean_transactions):
        anomalies = detector.detect_outliers_iqr(
            clean_transactions, columns=["amount"], dataset="transactions"
        )
        # Clean normally distributed data shouldn't trigger at default thresholds
        assert all(a.severity < AnomalySeverity.CRITICAL for a in anomalies)

    def test_detects_extreme_outliers(self, detector):
        np.random.seed(42)
        # Normal spread so IQR > 0, plus extreme outliers
        values = list(np.random.normal(100, 20, 90)) + [99999.0] * 10
        df = pd.DataFrame({"amount": values})
        anomalies = detector.detect_outliers_iqr(df, columns=["amount"])
        assert len(anomalies) >= 1
        assert anomalies[0].check_name == "outlier_iqr"

    def test_skips_non_numeric_columns(self, detector):
        df = pd.DataFrame({"name": ["a", "b", "c"], "val": [1, 2, 3]})
        anomalies = detector.detect_outliers_iqr(df, columns=["name"])
        assert len(anomalies) == 0

    def test_skips_constant_column(self, detector):
        df = pd.DataFrame({"val": [5.0] * 100})
        anomalies = detector.detect_outliers_iqr(df, columns=["val"])
        assert len(anomalies) == 0  # IQR = 0, skipped

    def test_auto_selects_numeric_columns(self, detector, clean_transactions):
        anomalies = detector.detect_outliers_iqr(clean_transactions)
        # Should only check numeric columns (amount, mcc_code)
        checked_cols = {a.column for a in anomalies}
        assert all(clean_transactions[c].dtype.kind in ("i", "f") for c in checked_cols)

    def test_details_contain_bounds(self, detector):
        values = [50.0] * 90 + [99999.0] * 10
        df = pd.DataFrame({"val": values})
        anomalies = detector.detect_outliers_iqr(df, columns=["val"])
        if anomalies:
            d = anomalies[0].details
            assert "q1" in d
            assert "q3" in d
            assert "lower_bound" in d
            assert "upper_bound" in d


# =====================================================================
# Outlier detection — Z-score
# =====================================================================


class TestOutlierDetectionZscore:
    def test_detects_zscore_outliers(self):
        """Use a strict detector so even a small outlier ratio triggers."""
        cfg = AnomalyConfig(outlier_ratio_warning=0.01)
        det = AnomalyDetector(config=cfg)
        np.random.seed(42)
        values = list(np.random.normal(100, 10, 950)) + [900] * 50
        df = pd.DataFrame({"amount": values})
        anomalies = det.detect_outliers_zscore(df, columns=["amount"])
        assert len(anomalies) >= 1
        assert anomalies[0].check_name == "outlier_zscore"

    def test_no_outliers_normal_data(self, detector):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.normal(100, 10, 200)})
        anomalies = detector.detect_outliers_zscore(df, columns=["val"])
        # Normal data rarely has >5% beyond 3 std
        assert len(anomalies) == 0

    def test_skips_constant_column(self, detector):
        df = pd.DataFrame({"val": [5.0] * 100})
        anomalies = detector.detect_outliers_zscore(df, columns=["val"])
        assert len(anomalies) == 0


# =====================================================================
# Schema validation
# =====================================================================


class TestSchemaValidation:
    def test_missing_required_column(self, detector):
        df = pd.DataFrame({"user_id": [1], "amount": [10.0]})
        anomalies = detector.detect_schema_violations(
            df,
            required_columns=["user_id", "amount", "category"],
            dataset="transactions",
        )
        missing = [a for a in anomalies if a.check_name == "schema_missing_column"]
        assert len(missing) == 1
        assert missing[0].column == "category"
        assert missing[0].severity == AnomalySeverity.CRITICAL

    def test_all_columns_present(self, detector, clean_transactions):
        anomalies = detector.detect_schema_violations(
            clean_transactions,
            required_columns=["transaction_id", "user_id", "amount"],
        )
        missing = [a for a in anomalies if a.check_name == "schema_missing_column"]
        assert len(missing) == 0

    def test_dtype_mismatch(self, detector):
        df = pd.DataFrame({"amount": ["not_a_number", "also_not"]})
        anomalies = detector.detect_schema_violations(
            df,
            expected_dtypes={"amount": "float"},
        )
        dtype_issues = [a for a in anomalies if a.check_name == "schema_dtype_mismatch"]
        assert len(dtype_issues) == 1

    def test_dtype_match(self, detector):
        df = pd.DataFrame({"amount": [1.0, 2.0]})
        anomalies = detector.detect_schema_violations(
            df,
            expected_dtypes={"amount": "float"},
        )
        assert len(anomalies) == 0

    def test_config_driven_schema(self):
        cfg = AnomalyConfig(
            required_columns={"my_data": ["col_a", "col_b"]},
            expected_dtypes={"my_data": {"col_a": "float"}},
        )
        det = AnomalyDetector(config=cfg)
        df = pd.DataFrame({"col_a": [1.0], "col_b": ["x"]})
        anomalies = det.detect_schema_violations(df, dataset="my_data")
        assert len(anomalies) == 0  # both columns present, col_a is float


# =====================================================================
# Distribution drift
# =====================================================================


class TestDistributionDrift:
    def test_no_drift_same_data(self, detector, clean_transactions):
        anomalies = detector.detect_distribution_drift(
            clean_transactions,
            clean_transactions,
            columns=["amount"],
            dataset="transactions",
        )
        assert len(anomalies) == 0

    def test_detects_drift(self, detector):
        np.random.seed(42)
        ref = pd.DataFrame({"val": np.random.normal(100, 10, 500)})
        cur = pd.DataFrame({"val": np.random.normal(200, 10, 500)})  # shifted
        anomalies = detector.detect_distribution_drift(ref, cur, columns=["val"])
        assert len(anomalies) >= 1
        assert anomalies[0].check_name == "distribution_drift"
        assert anomalies[0].details["ks_statistic"] > 0

    def test_empty_reference(self, detector):
        ref = pd.DataFrame({"val": pd.Series(dtype=float)})
        cur = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        anomalies = detector.detect_distribution_drift(ref, cur, columns=["val"])
        assert len(anomalies) == 0

    def test_auto_selects_shared_numeric_columns(self, detector):
        np.random.seed(42)
        ref = pd.DataFrame({"a": np.random.normal(0, 1, 100), "b": ["x"] * 100})
        cur = pd.DataFrame({"a": np.random.normal(0, 1, 100), "b": ["y"] * 100})
        anomalies = detector.detect_distribution_drift(ref, cur)
        # Only 'a' should be checked (numeric + shared)
        checked = {a.column for a in anomalies}
        assert "b" not in checked

    def test_drift_details_contain_stats(self, detector):
        np.random.seed(42)
        ref = pd.DataFrame({"val": np.random.normal(0, 1, 500)})
        cur = pd.DataFrame({"val": np.random.normal(5, 1, 500)})
        anomalies = detector.detect_distribution_drift(ref, cur, columns=["val"])
        assert len(anomalies) >= 1
        d = anomalies[0].details
        assert "ks_statistic" in d
        assert "p_value" in d
        assert "ref_mean" in d
        assert "cur_mean" in d


# =====================================================================
# run_all_checks
# =====================================================================


class TestRunAllChecks:
    def test_clean_data_no_anomalies(self, detector, clean_transactions):
        report = detector.run_all_checks(
            clean_transactions,
            dataset="transactions",
            # Exclude mcc_code — it's a categorical identifier, not continuous
            numeric_columns=["amount"],
        )
        assert isinstance(report, AnomalyReport)
        assert report.total_checks >= 4
        # Clean data should have no CRITICAL anomalies
        assert not report.has_critical

    def test_dirty_data_detects_anomalies(self, detector, dirty_transactions):
        report = detector.run_all_checks(dirty_transactions, dataset="transactions")
        assert len(report.anomalies) > 0
        assert report.has_critical  # 25% missing is CRITICAL

    def test_with_reference_df(self, detector, clean_transactions):
        np.random.seed(99)
        shifted = clean_transactions.copy()
        shifted["amount"] = shifted["amount"] + 5000  # massive shift
        report = detector.run_all_checks(
            shifted,
            dataset="transactions",
            reference_df=clean_transactions,
            numeric_columns=["amount"],
        )
        assert report.total_checks >= 5  # includes drift check
        drift_anomalies = [
            a for a in report.anomalies if a.check_name == "distribution_drift"
        ]
        assert len(drift_anomalies) >= 1

    def test_report_metadata(self, detector, clean_transactions):
        report = detector.run_all_checks(clean_transactions, dataset="txns")
        assert report.metadata["row_count"] == len(clean_transactions)
        assert report.metadata["column_count"] == len(clean_transactions.columns)
        assert "timestamp" in report.metadata


# =====================================================================
# Config
# =====================================================================


class TestAnomalyConfig:
    def test_default_values(self):
        cfg = AnomalyConfig()
        assert cfg.missing_warning_threshold == 0.05
        assert cfg.iqr_multiplier == 1.5
        assert cfg.zscore_threshold == 3.0

    def test_from_dict(self):
        cfg = AnomalyConfig.from_dict(
            {"missing_warning_threshold": 0.10, "iqr_multiplier": 2.0}
        )
        assert cfg.missing_warning_threshold == 0.10
        assert cfg.iqr_multiplier == 2.0

    def test_from_yaml(self, tmp_path):
        yaml_content = """
anomaly_detection:
  missing_warning_threshold: 0.08
  zscore_threshold: 2.5
  required_columns:
    transactions:
      - amount
      - user_id
"""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml_content)
        cfg = AnomalyConfig.from_yaml(cfg_path)
        assert cfg.missing_warning_threshold == 0.08
        assert cfg.zscore_threshold == 2.5
        assert cfg.required_columns["transactions"] == ["amount", "user_id"]
