"""
GCP Artifact Registry Client.

- Push, pull, and version model artifacts in GCP Artifact Registry. 
- Supports both Docker-format (containerized models) and generic-format (model binaries like .pkl, .pt, .joblib).

Versioning scheme: {model_name}-v{major}.{minor}.{patch}-{timestamp}
Example: personalization-model-v1.2.0-20260318T120000

Usage:
    from src.model_pipeline.registry.artifact_registry import RegistryClient

    client = RegistryClient(
        project="rewardsense-prod",
        location="us-central1",
        repository="rewardsense-models",
    )
    client.push_model("model.pkl", model_name="personalization", version="1.0.0")
    local = client.pull_model("personalization", version="1.0.0")
    versions = client.list_versions("personalization")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy GCS import — graceful fallback
# ---------------------------------------------------------------------------
try:
    from google.cloud import storage as gcs_storage

    GCS_AVAILABLE = True
except ImportError:
    gcs_storage = None  # type: ignore[assignment]
    GCS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default config (override via env vars or constructor)
# ---------------------------------------------------------------------------
DEFAULT_PROJECT = os.getenv("GCP_PROJECT", "rewardsense-prod")
DEFAULT_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
DEFAULT_REPOSITORY = os.getenv("GCP_MODEL_REPO", "rewardsense-models")
DEFAULT_BUCKET = os.getenv("GCP_MODEL_BUCKET", "rewardsense-model-artifacts")
LOCAL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", ".model_cache"))


class ModelVersion:
    """Represents a versioned model artifact."""

    def __init__(
        self,
        model_name: str,
        version: str,
        timestamp: str,
        sha256: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.version = version
        self.timestamp = timestamp
        self.sha256 = sha256
        self.metadata = metadata or {}

    @property
    def tag(self) -> str:
        return f"{self.model_name}-v{self.version}-{self.timestamp}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "timestamp": self.timestamp,
            "sha256": self.sha256,
            "tag": self.tag,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersion":
        return cls(
            model_name=d["model_name"],
            version=d["version"],
            timestamp=d["timestamp"],
            sha256=d["sha256"],
            metadata=d.get("metadata", {}),
        )


class RegistryClient:
    """GCP Artifact Registry client for model versioning.

    Uses GCS as the backing store for generic model artifacts.
    Each model is stored as:
        gs://{bucket}/{model_name}/v{version}/{artifact_file}
        gs://{bucket}/{model_name}/v{version}/manifest.json

    Parameters
    ----------
    project : str
        GCP project ID.
    location : str
        GCP region.
    repository : str
        Artifact Registry repository name.
    bucket : str
        GCS bucket for model binaries.
    local_cache : Path
        Local directory for caching pulled models.
    """

    def __init__(
        self,
        project: str = DEFAULT_PROJECT,
        location: str = DEFAULT_LOCATION,
        repository: str = DEFAULT_REPOSITORY,
        bucket: str = DEFAULT_BUCKET,
        local_cache: Optional[Path] = None,
    ) -> None:
        self.project = project
        self.location = location
        self.repository = repository
        self.bucket_name = bucket
        self.local_cache = local_cache or LOCAL_CACHE_DIR
        self.local_cache.mkdir(parents=True, exist_ok=True)
        self._gcs_client: Optional[Any] = None

        if GCS_AVAILABLE:
            try:
                self._gcs_client = gcs_storage.Client(project=project)
                logger.info("GCS client initialized for project '%s'", project)
            except Exception as e:
                logger.warning("GCS client init failed: %s. Using local-only mode.", e)

    # ------------------------------------------------------------------
    # File hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push_model(
        self,
        local_path: str | Path,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Push a model artifact to the registry.

        Parameters
        ----------
        local_path : str or Path
            Path to the model file (or directory) to upload.
        model_name : str
            Logical model name (e.g., "personalization").
        version : str
            Semantic version string (e.g., "1.0.0").
        metadata : dict, optional
            Additional metadata to store with the model version.

        Returns
        -------
        ModelVersion
            The created model version record.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {local_path}")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        # Compute hash
        if local_path.is_file():
            sha = self._compute_sha256(local_path)
        else:
            # For directories, hash a manifest of file hashes
            file_hashes = sorted(
                f"{self._compute_sha256(f)}:{f.relative_to(local_path)}"
                for f in local_path.rglob("*")
                if f.is_file()
            )
            sha = hashlib.sha256("\n".join(file_hashes).encode()).hexdigest()

        mv = ModelVersion(
            model_name=model_name,
            version=version,
            timestamp=ts,
            sha256=sha,
            metadata=metadata or {},
        )

        # --- GCS upload ---
        if self._gcs_client is not None:
            bucket = self._gcs_client.bucket(self.bucket_name)
            prefix = f"{model_name}/v{version}"

            if local_path.is_file():
                blob = bucket.blob(f"{prefix}/{local_path.name}")
                blob.upload_from_filename(str(local_path))
                logger.info(
                    "Uploaded %s → gs://%s/%s/%s",
                    local_path.name,
                    self.bucket_name,
                    prefix,
                    local_path.name,
                )
            else:
                for f in local_path.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(local_path)
                        blob = bucket.blob(f"{prefix}/{rel}")
                        blob.upload_from_filename(str(f))

            # Upload manifest
            manifest_blob = bucket.blob(f"{prefix}/manifest.json")
            manifest_blob.upload_from_string(
                json.dumps(mv.to_dict(), indent=2),
                content_type="application/json",
            )
            logger.info("Pushed %s to gs://%s/%s", mv.tag, self.bucket_name, prefix)
        else:
            logger.warning("GCS unavailable — saving locally only")

        # --- Local cache ---
        cache_dir = self.local_cache / model_name / f"v{version}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if local_path.is_file():
            shutil.copy2(local_path, cache_dir / local_path.name)
        else:
            if (cache_dir).exists():
                shutil.rmtree(cache_dir)
            shutil.copytree(local_path, cache_dir)

        manifest_path = cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(mv.to_dict(), indent=2))

        return mv

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull_model(
        self,
        model_name: str,
        version: str,
        force: bool = False,
    ) -> Path:
        """Pull a model artifact from the registry.

        Checks local cache first. If not cached (or force=True),
        downloads from GCS.

        Returns
        -------
        Path
            Local path to the model artifact directory.
        """
        cache_dir = self.local_cache / model_name / f"v{version}"
        manifest_path = cache_dir / "manifest.json"

        if manifest_path.exists() and not force:
            logger.info("Using cached model: %s v%s", model_name, version)
            return cache_dir

        # Download from GCS
        if self._gcs_client is None:
            if cache_dir.exists():
                return cache_dir
            raise RuntimeError(
                f"Model {model_name} v{version} not in local cache and "
                "GCS client is unavailable."
            )

        bucket = self._gcs_client.bucket(self.bucket_name)
        prefix = f"{model_name}/v{version}/"

        cache_dir.mkdir(parents=True, exist_ok=True)
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            raise FileNotFoundError(
                f"No artifacts found at gs://{self.bucket_name}/{prefix}"
            )

        for blob in blobs:
            rel = blob.name[len(prefix) :]
            if not rel:
                continue
            local_file = cache_dir / rel
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))

        logger.info("Pulled %s v%s → %s", model_name, version, cache_dir)
        return cache_dir

    # ------------------------------------------------------------------
    # List / Query
    # ------------------------------------------------------------------

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model in the registry."""
        versions = []

        # Check local cache
        local_model_dir = self.local_cache / model_name
        if local_model_dir.exists():
            for vdir in sorted(local_model_dir.iterdir()):
                manifest = vdir / "manifest.json"
                if manifest.exists():
                    data = json.loads(manifest.read_text())
                    versions.append(ModelVersion.from_dict(data))

        # Check GCS for additional versions
        if self._gcs_client is not None:
            bucket = self._gcs_client.bucket(self.bucket_name)
            prefix = f"{model_name}/"
            blobs = bucket.list_blobs(prefix=prefix, delimiter="/")

            # Collect version prefixes
            seen_versions = {v.version for v in versions}
            for page in blobs.pages:
                for pfx in page.prefixes:
                    # pfx looks like "model_name/v1.0.0/"
                    ver_str = pfx.rstrip("/").split("/")[-1].lstrip("v")
                    if ver_str not in seen_versions:
                        manifest_blob = bucket.blob(f"{pfx}manifest.json")
                        if manifest_blob.exists():
                            data = json.loads(manifest_blob.download_as_text())
                            versions.append(ModelVersion.from_dict(data))

        versions.sort(key=lambda v: (v.timestamp, v.version), reverse=True)
        return versions

    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the most recent version of a model."""
        versions = self.list_versions(model_name)
        return versions[0] if versions else None

    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a model version from local cache and GCS."""
        deleted = False

        # Local
        cache_dir = self.local_cache / model_name / f"v{version}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            deleted = True

        # GCS
        if self._gcs_client is not None:
            bucket = self._gcs_client.bucket(self.bucket_name)
            prefix = f"{model_name}/v{version}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            for blob in blobs:
                blob.delete()
            if blobs:
                deleted = True

        return deleted
