"""
RewardSense Model Pipeline Package.

Provides the model training, evaluation, and serving infrastructure
for the credit card recommendation system.
"""

import os
from typing import Dict, Any

__version__ = "0.1.0"


def get_config() -> Dict[str, Any]:
    """
    Return environment-aware configuration for the model pipeline.

    Reads EXECUTION_ENV to determine whether to use local or GCP paths
    for data, MLflow tracking, and artifact storage.

    Returns:
        Dict containing data_dir, mlflow_tracking_uri, artifact_store,
        model_registry, and execution_env keys.
    """
    execution_env = os.getenv("EXECUTION_ENV", "local")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    gcp_project = os.getenv("GCP_PROJECT_ID", "rewardsense")
    gcp_bucket = os.getenv("GCP_BUCKET_NAME", "rewardsense-dvc-store")

    if execution_env == "gcp":
        return {
            "execution_env": "gcp",
            "data_dir": f"gs://{gcp_bucket}/data/processed",
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "artifact_store": f"gs://{gcp_bucket}/mlflow-artifacts",
            "model_registry": f"gs://{gcp_bucket}/model-registry",
            "gcp_project": gcp_project,
            "gcp_bucket": gcp_bucket,
        }
    else:
        return {
            "execution_env": "local",
            "data_dir": os.getenv("LOCAL_DATA_DIR", "data/processed"),
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "artifact_store": os.getenv("LOCAL_ARTIFACT_STORE", "mlruns"),
            "model_registry": os.getenv("LOCAL_MODEL_REGISTRY", "models"),
            "gcp_project": gcp_project,
            "gcp_bucket": gcp_bucket,
        }
