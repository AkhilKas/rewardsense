"""
Champion-Challenger logic for MLflow.
"""

import logging
from typing import Any, Optional

import mlflow

logger = logging.getLogger(__name__)


class ChampionChallenger:
    """Evaluates a challenger model against the current champion model in MLflow."""

    def __init__(
        self,
        experiment_name: str = "personalization_model",
        metric_to_compare: str = "ndcg_at_5",
        maximize: bool = True,
    ):
        self.experiment_name = experiment_name
        self.metric_to_compare = metric_to_compare
        self.maximize = maximize

    def get_champion_metric(self) -> Optional[float]:
        """Fetch the performance metric of the current champion from MLflow."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                return None

            # Find runs tagged as 'champion'
            runs: Any = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.model_status = 'champion'",
                order_by=[
                    f"metrics.{self.metric_to_compare} {'DESC' if self.maximize else 'ASC'}"
                ],
            )
            if runs.empty:
                return None
            return float(runs.iloc[0].get(f"metrics.{self.metric_to_compare}"))
        except Exception as e:
            logger.warning(f"Could not retrieve champion metric: {e}")
            return None

    def compare(self, challenger_metric: float) -> bool:
        """
        Compare the challenger against the champion.
        Returns True if the challenger should be promoted.
        """
        champion_metric = self.get_champion_metric()

        # If no champion exists, challenger wins by default
        if champion_metric is None:
            logger.info("No champion found. Challenger wins by default.")
            return True

        logger.info(
            f"Comparing Challenger ({challenger_metric}) vs Champion ({champion_metric}) "
            f"for metric '{self.metric_to_compare}' (maximize={self.maximize})"
        )

        if self.maximize:
            return challenger_metric >= champion_metric
        else:
            return challenger_metric <= champion_metric
