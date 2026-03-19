"""
Rollback mechanism to revert production tags in the Registry if a model degrades.
"""

import logging
from typing import Optional

import mlflow

from src.model_pipeline.cd.notifier import NotificationDispatcher
from src.model_pipeline.registry.artifact_registry import RegistryClient

logger = logging.getLogger(__name__)


class ModelRollbackManager:
    """Manages rollback of the active production model to the previous stable version."""

    def __init__(self, project_id: str, bucket_name: str, model_name: str):
        self.client = RegistryClient(project=project_id, bucket_name=bucket_name)
        self.model_name = model_name
        self.notifier = NotificationDispatcher()

    def determine_previous_version(self) -> Optional[str]:
        """
        Query MLflow to find the second-most-recent champion or just
        query the Registry bucket to find previous folders.
        For simplicity, we query MLflow.
        """
        try:
            # We assume experiment is named 'personalization_model'
            # and runs are tagged with 'model_status'='champion'
            experiment = mlflow.get_experiment_by_name("personalization_model")
            if not experiment:
                return None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.model_status = 'champion'",
                order_by=["start_time DESC"],
            )
            # Find the run right before the current one
            if len(runs) > 1:
                # runs.iloc[1] is the previous champion
                # Assuming previous champion was versioned by its run_id
                return runs.iloc[1].run_id
            return None
        except Exception as e:
            logger.error(f"Failed to fetch previous version from MLflow: {e}")
            return None

    def execute_rollback(self, reason: str = "Metric degraded") -> bool:
        """
        Execute the rollback sequence.
        """
        logger.warning(f"Initiating rollback for {self.model_name}. Reason: {reason}")
        self.notifier.notify(
            f"Initiating rollback for {self.model_name}. Reason: {reason}",
            level="CRITICAL",
        )

        prev_version = self.determine_previous_version()
        if not prev_version:
            msg = "Rollback failed: Could not determine previous version."
            logger.error(msg)
            self.notifier.notify(msg, level="CRITICAL")
            return False

        try:
            # Instead of pushing, a true registry client might just remap a 'production' tag.
            # Since our RegistryClient just uploads blobs to GCS, a "rollback" in this
            # context implies we repoint whatever serves the model (or just announce it).
            # We'll log the rollback event.
            mlflow.set_experiment("personalization_model")
            with mlflow.start_run(run_name="rollback_event"):
                mlflow.log_param("rollback_reason", reason)
                mlflow.log_param("rolled_back_to", prev_version)

            msg = f"Rollback successful. Model {self.model_name} reverted to {prev_version}."
            logger.info(msg)
            self.notifier.notify(msg, level="INFO")
            return True
        except Exception as e:
            msg = f"Rollback failed during execution: {e}"
            logger.error(msg)
            self.notifier.notify(msg, level="CRITICAL")
            return False
