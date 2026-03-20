"""
Automated CD Gates for Model Pipeline Deployment.
"""

import json
import logging
from pathlib import Path
from typing import Dict

from src.model_pipeline.registry.artifact_registry import RegistryClient

logger = logging.getLogger(__name__)


class ValidationGate:
    """Blocks deployment if performance metrics fail to meet thresholds."""

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def evaluate(self, metrics: Dict[str, float]) -> bool:
        """Evaluate if the provided metrics meet all thresholds."""
        for metric, min_val in self.thresholds.items():
            val = metrics.get(metric, 0.0)
            if val < min_val:
                logger.warning(f"Validation failed: {metric} ({val}) < {min_val}")
                return False
        return True


class BiasGate:
    """Blocks deployment if fairness disparity exceeds acceptable thresholds."""

    def __init__(self, max_disparity: float = 0.10):
        self.max_disparity = max_disparity

    def evaluate(self, report_path: str) -> bool:
        """Evaluate a generated JSON bias report to ensure disparities are within limit."""
        path = Path(report_path)
        if not path.exists():
            logger.error("Bias report missing, gate failed.")
            return False

        try:
            with open(path) as f:
                report = json.load(f)

            for metric in report.get("metrics", []):
                if metric.get("is_biased", False):
                    # Check if it strictly exceeds our max disparity threshold
                    if metric.get("value", 0.0) > self.max_disparity:
                        logger.warning(
                            f"Bias gate blocked: {metric['name']} across "
                            f"{metric['sensitive_feature']} has disparity "
                            f"{metric['value']} > {self.max_disparity}"
                        )
                        return False
            return True
        except Exception as e:
            logger.error(f"Error parsing bias report: {e}")
            return False


class RegistryGate:
    """Pushes successful models to the Artifact Registry (currently backed by GCS)."""

    def __init__(self, project_id: str, location: str, repository: str, model_name: str):
        self.client = RegistryClient(project=project_id, location=location, repository=repository)
        self.model_name = model_name

    def push(self, local_model_dir: str, version_tag: str) -> str:
        """Push model via client and return the URI."""
        logger.info(f"Pushing {self.model_name} (v{version_tag}) to registry")
        return self.client.push_model(
            local_dir=local_model_dir, model_name=self.model_name, version=version_tag
        )
