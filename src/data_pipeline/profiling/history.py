"""
Historical statistics storage and comparison.

Tracks statistics over time for trend analysis and drift detection.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class StatisticsHistory:
    """Store and retrieve historical statistics."""

    def __init__(self, history_dir: str = "data/profiling/history"):
        """
        Initialize statistics history tracker.

        Args:
            history_dir: Directory to store historical statistics
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized StatisticsHistory at: {self.history_dir}")

    def save_statistics(self, dataset_name: str, stats: Dict[str, Any]) -> Path:
        """
        Save statistics for a dataset.

        Args:
            dataset_name: Name of dataset (transactions, users, cards)
            stats: Statistics dictionary

        Returns:
            Path to saved file
        """

        # Save to timestamped file
        filename = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.history_dir / filename

        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"Saved {dataset_name} statistics to: {filepath}")

        # Also update latest
        latest_path = self.history_dir / f"{dataset_name}_latest.json"
        with open(latest_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        return filepath

    def load_latest(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load latest statistics for a dataset.

        Args:
            dataset_name: Name of dataset

        Returns:
            Statistics dictionary or None if not found
        """
        latest_path = self.history_dir / f"{dataset_name}_latest.json"

        if not latest_path.exists():
            logger.warning(f"No historical statistics found for {dataset_name}")
            return None

        with open(latest_path, "r") as f:
            stats = json.load(f)

        logger.info(f"Loaded latest statistics for {dataset_name}")
        return stats

    def load_history(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load all historical statistics for a dataset.

        Args:
            dataset_name: Name of dataset

        Returns:
            List of statistics dictionaries, sorted by timestamp
        """
        pattern = f"{dataset_name}_*.json"
        files = sorted(self.history_dir.glob(pattern))

        # Exclude '_latest.json'
        files = [f for f in files if not f.name.endswith("_latest.json")]

        history = []
        for filepath in files:
            with open(filepath, "r") as f:
                history.append(json.load(f))

        logger.info(f"Loaded {len(history)} historical records for {dataset_name}")
        return history

    def compare_with_previous(
        self, dataset_name: str, current_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare current statistics with previous run.

        Args:
            dataset_name: Name of dataset
            current_stats: Current statistics

        Returns:
            Dictionary of changes/deltas
        """
        previous = self.load_latest(dataset_name)

        if previous is None:
            logger.info(f"No previous statistics to compare for {dataset_name}")
            return {"status": "first_run"}

        changes = {
            "timestamp": current_stats.get("timestamp"),
            "previous_timestamp": previous.get("timestamp"),
            "changes": {},
        }

        # Compare total counts
        if "total_transactions" in current_stats and "total_transactions" in previous:
            delta = current_stats["total_transactions"] - previous["total_transactions"]
            changes["changes"]["total_transactions"] = {
                "current": current_stats["total_transactions"],
                "previous": previous["total_transactions"],
                "delta": delta,
                "pct_change": (
                    (delta / previous["total_transactions"] * 100)
                    if previous["total_transactions"] > 0
                    else 0
                ),
            }

        # Compare amount statistics
        if "amount" in current_stats and "amount" in previous:
            changes["changes"]["amount_mean"] = {
                "current": current_stats["amount"]["mean"],
                "previous": previous["amount"]["mean"],
                "delta": current_stats["amount"]["mean"] - previous["amount"]["mean"],
            }

        logger.info(f"Compared {dataset_name} with previous run")
        return changes
