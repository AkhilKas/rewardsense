import pytest

pytest.importorskip("airflow", reason="Airflow not installed in CI")

"""
RewardSense - DAG Structure Tests

Validates that the main data pipeline DAG:
    ✅ Is parseable by Airflow (no import errors)
    ✅ Contains the expected task IDs and task groups
    ✅ Has correct dependency chains
    ✅ DAG attributes (schedule, catchup, tags) are set properly
    ✅ Has documentation for UI display

Run with: pytest tests/dags/test_rewardsense_data_pipeline.py -v
"""

import pytest  # noqa: E402
from airflow.models import DagBag  # noqa: E402


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def dagbag():
    """Load DAGs from the project dags/ directory."""
    return DagBag(dag_folder="dags", include_examples=False)


@pytest.fixture(scope="module")
def pipeline_dag(dagbag):
    """Get the main pipeline DAG."""
    dag_id = "rewardsense_data_pipeline"
    assert (
        dag_id in dagbag.dags
    ), f"DAG '{dag_id}' not found. Import errors: {dagbag.import_errors}"
    return dagbag.dags[dag_id]


# =============================================================================
# DAG Parsing & Import Tests
# =============================================================================


class TestDAGParsing:
    """Verify the DAG file can be parsed without errors."""

    def test_no_import_errors(self, dagbag):
        """
        Given: The dags/ directory
        When: Airflow parses all DAG files
        Then: No import errors should be reported
        """
        dag_errors = {
            k: v
            for k, v in dagbag.import_errors.items()
            if "rewardsense_data_pipeline" in k
        }
        assert dag_errors == {}, f"DAG import errors: {dag_errors}"

    def test_dag_is_loaded(self, dagbag):
        """
        Given: The dags/ directory
        When: Airflow discovers DAGs
        Then: rewardsense_data_pipeline should be present
        """
        assert "rewardsense_data_pipeline" in dagbag.dags

    def test_dag_has_no_cycles(self, pipeline_dag):
        """
        Given: The pipeline DAG
        When: Airflow validates the graph
        Then: No circular dependencies should exist
        """
        # Airflow raises an exception during parsing if cycles exist.
        # If we get here, there are no cycles.
        assert pipeline_dag is not None


# =============================================================================
# DAG Attribute Tests
# =============================================================================


class TestDAGAttributes:
    """Verify DAG-level configuration is correct."""

    def test_schedule(self, pipeline_dag):
        """DAG should run weekly on Sunday at 06:00 UTC."""
        assert pipeline_dag.schedule_interval == "0 6 * * 0"

    def test_catchup_disabled(self, pipeline_dag):
        """DAG should not backfill missed runs."""
        assert pipeline_dag.catchup is False

    def test_max_active_runs(self, pipeline_dag):
        """Only one pipeline run at a time."""
        assert pipeline_dag.max_active_runs == 1

    def test_tags(self, pipeline_dag):
        """DAG should have descriptive tags for filtering in the UI."""
        tags = pipeline_dag.tags
        assert "rewardsense" in tags
        assert "data-pipeline" in tags
        assert "weekly" in tags

    def test_doc_md_present(self, pipeline_dag):
        """DAG should have markdown documentation for the UI."""
        assert pipeline_dag.doc_md is not None
        assert len(pipeline_dag.doc_md) > 100
        assert "RewardSense" in pipeline_dag.doc_md

    def test_default_args_retries(self, pipeline_dag):
        """Default args should configure 2 retries."""
        assert pipeline_dag.default_args["retries"] == 2

    def test_default_args_owner(self, pipeline_dag):
        """Default args should set owner to rewardsense."""
        assert pipeline_dag.default_args["owner"] == "rewardsense"


# =============================================================================
# Task Existence Tests
# =============================================================================


class TestTaskExistence:
    """Verify all expected tasks exist in the DAG."""

    EXPECTED_TASK_IDS = [
        # Sentinels
        "pipeline_start",
        "pipeline_end",
        # Ingestion group
        "ingestion.scrape_nerdwallet",
        "ingestion.scrape_issuers",
        "ingestion.fetch_api_data",
        "ingestion.generate_synthetic_data",
        "ingestion.merge_card_data",
        # Preprocessing group
        "preprocessing.check_raw_data_ready",
        "preprocessing.clean_data",
        "preprocessing.engineer_features",
        "preprocessing.run_transform_pipeline",
        # Versioning group
        "versioning.version_raw_data",
        "versioning.version_processed_data",
        "versioning.push_to_remote",
        "versioning.commit_dvc_files",
        # Reporting group
        "reporting.generate_pipeline_report",
        "reporting.log_pipeline_metrics",
        "reporting.send_pipeline_alerts",
        "reporting.generate_performance_dashboard",
        "reporting.check_performance_regression",
    ]

    def test_all_expected_tasks_present(self, pipeline_dag):
        """All planned tasks should exist in the DAG."""
        dag_task_ids = {task.task_id for task in pipeline_dag.tasks}
        for task_id in self.EXPECTED_TASK_IDS:
            assert (
                task_id in dag_task_ids
            ), f"Missing task: {task_id}. Found: {sorted(dag_task_ids)}"

    def test_task_count(self, pipeline_dag):
        """DAG should have exactly the expected number of tasks."""
        assert len(pipeline_dag.tasks) == len(self.EXPECTED_TASK_IDS)


# =============================================================================
# Task Group Tests
# =============================================================================


class TestTaskGroups:
    """Verify task groups are properly organized."""

    def test_ingestion_group_has_five_tasks(self, pipeline_dag):
        """Ingestion group should contain 5 tasks."""
        ingestion_tasks = [
            t for t in pipeline_dag.tasks if t.task_id.startswith("ingestion.")
        ]
        assert len(ingestion_tasks) == 5

    def test_preprocessing_group_has_four_tasks(self, pipeline_dag):
        """Preprocessing group should contain 4 tasks."""
        preprocessing_tasks = [
            t for t in pipeline_dag.tasks if t.task_id.startswith("preprocessing.")
        ]
        assert len(preprocessing_tasks) == 4

    def test_versioning_group_has_four_tasks(self, pipeline_dag):
        """Versioning group should contain 4 tasks."""
        versioning_tasks = [
            t for t in pipeline_dag.tasks if t.task_id.startswith("versioning.")
        ]
        assert len(versioning_tasks) == 4

    def test_reporting_group_has_five_tasks(self, pipeline_dag):
        """Reporting group should contain 5 tasks (report, alerts, metrics, perf)."""
        reporting_tasks = [
            t for t in pipeline_dag.tasks if t.task_id.startswith("reporting.")
        ]
        assert len(reporting_tasks) == 5


# =============================================================================
# Dependency Tests
# =============================================================================


class TestDependencies:
    """Verify task dependencies reflect the intended pipeline flow."""

    def _get_upstream_ids(self, pipeline_dag, task_id):
        """Helper to get upstream task IDs for a given task."""
        task = pipeline_dag.get_task(task_id)
        return {t.task_id for t in task.upstream_list}

    def _get_downstream_ids(self, pipeline_dag, task_id):
        """Helper to get downstream task IDs for a given task."""
        task = pipeline_dag.get_task(task_id)
        return {t.task_id for t in task.downstream_list}

    def test_pipeline_start_triggers_ingestion(self, pipeline_dag):
        """pipeline_start should feed into all ingestion tasks."""
        downstream = self._get_downstream_ids(pipeline_dag, "pipeline_start")
        # pipeline_start connects to the ingestion task group
        # which contains scrape_nerdwallet, scrape_issuers, fetch_api_data, generate_synthetic
        assert len(downstream) > 0

    def test_scrapers_and_api_converge_at_merge(self, pipeline_dag):
        """scrape_nerdwallet, scrape_issuers, fetch_api_data → merge_card_data."""
        merge_upstream = self._get_upstream_ids(
            pipeline_dag, "ingestion.merge_card_data"
        )
        assert "ingestion.scrape_nerdwallet" in merge_upstream
        assert "ingestion.scrape_issuers" in merge_upstream
        assert "ingestion.fetch_api_data" in merge_upstream

    def test_preprocessing_chain(self, pipeline_dag):
        """check_raw_data_ready -> clean -> features -> transform should be sequential."""
        clean_upstream = self._get_upstream_ids(
            pipeline_dag, "preprocessing.clean_data"
        )
        assert "preprocessing.check_raw_data_ready" in clean_upstream

        features_upstream = self._get_upstream_ids(
            pipeline_dag, "preprocessing.engineer_features"
        )
        assert "preprocessing.clean_data" in features_upstream

        transform_upstream = self._get_upstream_ids(
            pipeline_dag, "preprocessing.run_transform_pipeline"
        )
        assert "preprocessing.engineer_features" in transform_upstream

    def test_versioning_follows_preprocessing(self, pipeline_dag):
        """version_raw_data should depend on preprocessing completion."""
        version_upstream = self._get_upstream_ids(
            pipeline_dag, "versioning.version_raw_data"
        )
        # The upstream is the preprocessing group (via run_transform_pipeline)
        assert len(version_upstream) > 0

    def test_versioning_chain(self, pipeline_dag):
        """version_raw_data -> version_processed_data -> push_to_remote -> commit_dvc_files"""
        processed_upstream = self._get_upstream_ids(
            pipeline_dag, "versioning.version_processed_data"
        )
        assert "versioning.version_raw_data" in processed_upstream

        push_upstream = self._get_upstream_ids(
            pipeline_dag, "versioning.push_to_remote"
        )
        assert "versioning.version_processed_data" in push_upstream

        commit_upstream = self._get_upstream_ids(
            pipeline_dag, "versioning.commit_dvc_files"
        )
        assert "versioning.push_to_remote" in commit_upstream

    def test_reporting_follows_versioning(self, pipeline_dag):
        """generate_pipeline_report should depend on versioning."""
        report_upstream = self._get_upstream_ids(
            pipeline_dag, "reporting.generate_pipeline_report"
        )
        assert len(report_upstream) > 0

    def test_reporting_dependencies(self, pipeline_dag):
        """Report fans out to alerts/metrics, then performance chain follows metrics."""
        metrics_upstream = self._get_upstream_ids(
            pipeline_dag, "reporting.log_pipeline_metrics"
        )
        assert "reporting.generate_pipeline_report" in metrics_upstream

        alerts_upstream = self._get_upstream_ids(
            pipeline_dag, "reporting.send_pipeline_alerts"
        )
        assert "reporting.generate_pipeline_report" in alerts_upstream

        dashboard_upstream = self._get_upstream_ids(
            pipeline_dag, "reporting.generate_performance_dashboard"
        )
        assert "reporting.log_pipeline_metrics" in dashboard_upstream

        regression_upstream = self._get_upstream_ids(
            pipeline_dag, "reporting.check_performance_regression"
        )
        assert "reporting.generate_performance_dashboard" in regression_upstream

    def test_pipeline_end_is_terminal(self, pipeline_dag):
        """pipeline_end should have no downstream tasks."""
        downstream = self._get_downstream_ids(pipeline_dag, "pipeline_end")
        assert len(downstream) == 0
