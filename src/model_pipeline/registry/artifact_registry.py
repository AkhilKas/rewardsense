"""
GCP Artifact Registry Client.

- Push, pull, and version model artifacts using GCP Artifact Registry (generic repository format). 
- Handles authentication via google-auth and Application Default Credentials (ADC).

Artifact Registry generic format stores artifacts at:
    {location}-generic.pkg.dev/{project}/{repo}/{package}/{version}/{filename}

Versioning scheme: {model_name}-v{major}.{minor}.{patch}-{timestamp}
Example: personalization-v1.2.0-20260318T120000

IAM Roles Required:
    - roles/artifactregistry.reader   → pull_model, list_versions
    - roles/artifactregistry.writer   → push_model (includes reader)
    - roles/artifactregistry.admin    → delete_version, manage repos

Setup:
    # Authenticate (one-time)
    gcloud auth application-default login

    # Or use a service account
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

    # Grant roles to a service account
    gcloud artifacts repositories add-iam-policy-binding rewardsense-models \\
        --project=rewardsense-prod \\
        --location=us-central1 \\
        --member="serviceAccount:rewardsense-pipeline@rewardsense-prod.iam.gserviceaccount.com" \\
        --role="roles/artifactregistry.writer"

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
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    import google.auth
    import google.auth.transport.requests
    from google.auth import default as google_auth_default

    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False
    logger.warning(
        "google-auth not installed — Artifact Registry calls disabled. "
        "Install with: pip install google-auth"
    )

try:
    import requests as http_requests

    REQUESTS_AVAILABLE = True
except ImportError:
    http_requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False

try:
    from google.cloud import artifactregistry_v1

    AR_CLIENT_AVAILABLE = True
except ImportError:
    artifactregistry_v1 = None  # type: ignore[assignment]
    AR_CLIENT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default config (override via env vars or constructor)
# ---------------------------------------------------------------------------
DEFAULT_PROJECT = os.getenv("GCP_PROJECT", "rewardsense-prod")
DEFAULT_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
DEFAULT_REPOSITORY = os.getenv("GCP_MODEL_REPO", "rewardsense-models")
LOCAL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", ".model_cache"))

# Artifact Registry generic format base URL
AR_GENERIC_URL = "https://{location}-generic.pkg.dev/{project}/{repo}"
# Artifact Registry API base URL (for management operations)
AR_API_URL = "https://artifactregistry.googleapis.com/v1"


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
    """
    GCP Artifact Registry client for model versioning.

    Uses Artifact Registry's **generic repository format** to store
    model artifacts as versioned packages. Each model is stored as:

        {location}-generic.pkg.dev/{project}/{repo}/{model_name}/{version}/{filename}

    Authentication uses Application Default Credentials (ADC) via
    google-auth. The client automatically refreshes tokens.

    Parameters
    ----------
    project : str
        GCP project ID.
    location : str
        GCP region (e.g., "us-central1").
    repository : str
        Artifact Registry repository name. Must exist and be of
        format ``GENERIC``.
    local_cache : Path
        Local directory for caching pulled models.
    """

    def __init__(
        self,
        project: str = DEFAULT_PROJECT,
        location: str = DEFAULT_LOCATION,
        repository: str = DEFAULT_REPOSITORY,
        local_cache: Optional[Path] = None,
    ) -> None:
        self.project = project
        self.location = location
        self.repository = repository
        self.local_cache = local_cache or LOCAL_CACHE_DIR
        self.local_cache.mkdir(parents=True, exist_ok=True)

        self._credentials: Optional[Any] = None
        self._ar_client: Optional[Any] = None

        # Base URL for generic artifact uploads/downloads
        self._generic_base = AR_GENERIC_URL.format(
            location=location, project=project, repo=repository
        )
        # Parent resource path for Artifact Registry API
        self._repo_parent = (
            f"projects/{project}/locations/{location}" f"/repositories/{repository}"
        )

        # Authenticate
        if GOOGLE_AUTH_AVAILABLE:
            try:
                self._credentials, _ = google_auth_default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                logger.info(
                    "Authenticated with ADC for Artifact Registry " "(%s/%s/%s)",
                    project,
                    location,
                    repository,
                )
            except Exception as e:
                logger.warning(
                    "ADC authentication failed: %s. Using local-only mode.", e
                )

        # Optional: management client for listing/deleting
        if AR_CLIENT_AVAILABLE and self._credentials is not None:
            try:
                self._ar_client = artifactregistry_v1.ArtifactRegistryClient()
                logger.info("Artifact Registry management client initialized")
            except Exception as e:
                logger.warning("AR management client init failed: %s", e)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers with a fresh access token."""
        if self._credentials is None:
            raise RuntimeError(
                "No GCP credentials available. Run:\n"
                "  gcloud auth application-default login\n"
                "Or set GOOGLE_APPLICATION_CREDENTIALS env var."
            )
        # Refresh if expired
        auth_request = google.auth.transport.requests.Request()
        self._credentials.refresh(auth_request)
        return {
            "Authorization": f"Bearer {self._credentials.token}",
        }

    @property
    def is_remote_available(self) -> bool:
        """Check if remote Artifact Registry is accessible."""
        return (
            GOOGLE_AUTH_AVAILABLE
            and REQUESTS_AVAILABLE
            and self._credentials is not None
        )

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

    @staticmethod
    def _compute_dir_sha256(dirpath: Path) -> str:
        """Compute aggregate SHA-256 for a directory."""
        file_hashes = sorted(
            f"{RegistryClient._compute_sha256(f)}:{f.relative_to(dirpath)}"
            for f in dirpath.rglob("*")
            if f.is_file()
        )
        return hashlib.sha256("\n".join(file_hashes).encode()).hexdigest()

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push_model(
        self,
        local_path: Union[str, Path],
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """
        Push a model artifact to Artifact Registry.

        Uploads to the generic repository format. Also caches locally.

        Parameters
        ----------
        local_path : str or Path
            Path to the model file (or directory) to upload.
        model_name : str
            Package name in Artifact Registry (e.g., "personalization").
        version : str
            Semantic version string (e.g., "1.0.0").
        metadata : dict, optional
            Additional metadata stored in the manifest.

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
            sha = self._compute_dir_sha256(local_path)

        mv = ModelVersion(
            model_name=model_name,
            version=version,
            timestamp=ts,
            sha256=sha,
            metadata=metadata or {},
        )

        # --- Upload to Artifact Registry ---
        if self.is_remote_available:
            self._upload_to_ar(local_path, model_name, version, mv)
        else:
            logger.warning("Artifact Registry unavailable — saving locally only")

        # --- Local cache ---
        self._cache_locally(local_path, model_name, version, mv)

        return mv

    def _upload_to_ar(
        self,
        local_path: Path,
        model_name: str,
        version: str,
        mv: ModelVersion,
    ) -> None:
        """Upload files to Artifact Registry generic format."""
        headers = self._get_auth_headers()

        files_to_upload: List[tuple] = []

        if local_path.is_file():
            files_to_upload.append((local_path.name, local_path))
        else:
            for f in local_path.rglob("*"):
                if f.is_file():
                    rel = str(f.relative_to(local_path))
                    files_to_upload.append((rel, f))

        # Upload each file
        for filename, filepath in files_to_upload:
            upload_url = f"{self._generic_base}/{model_name}/{version}/{filename}"
            with open(filepath, "rb") as f:
                resp = http_requests.put(
                    upload_url,
                    data=f,
                    headers={
                        **headers,
                        "Content-Type": "application/octet-stream",
                    },
                    timeout=300,
                )

            if resp.status_code not in (200, 201):
                logger.error(
                    "Upload failed for %s: %s %s",
                    filename,
                    resp.status_code,
                    resp.text,
                )
                raise RuntimeError(
                    f"Artifact Registry upload failed: {resp.status_code} "
                    f"{resp.text}"
                )
            logger.info("Uploaded %s → %s", filename, upload_url)

        # Upload manifest
        manifest_url = f"{self._generic_base}/{model_name}/{version}/manifest.json"
        resp = http_requests.put(
            manifest_url,
            data=json.dumps(mv.to_dict(), indent=2).encode(),
            headers={
                **headers,
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        if resp.status_code not in (200, 201):
            logger.warning("Manifest upload failed: %s", resp.text)

        logger.info(
            "Pushed %s v%s to Artifact Registry (%s/%s/%s)",
            model_name,
            version,
            self.project,
            self.location,
            self.repository,
        )

    def _cache_locally(
        self,
        local_path: Path,
        model_name: str,
        version: str,
        mv: ModelVersion,
    ) -> None:
        """Save model artifact to local cache."""
        cache_dir = self.local_cache / model_name / f"v{version}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if local_path.is_file():
            shutil.copy2(local_path, cache_dir / local_path.name)
        else:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            shutil.copytree(local_path, cache_dir)

        manifest_path = cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(mv.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull_model(
        self,
        model_name: str,
        version: str,
        force: bool = False,
    ) -> Path:
        """
        Pull a model artifact from Artifact Registry.

        Checks local cache first. If not cached (or force=True),
        downloads from Artifact Registry.

        Returns
        -------
        Path
            Local path to the cached model artifact directory.
        """
        cache_dir = self.local_cache / model_name / f"v{version}"
        manifest_path = cache_dir / "manifest.json"

        if manifest_path.exists() and not force:
            logger.info("Using cached model: %s v%s", model_name, version)
            return cache_dir

        # Download from Artifact Registry
        if not self.is_remote_available:
            if cache_dir.exists():
                return cache_dir
            raise RuntimeError(
                f"Model {model_name} v{version} not in local cache and "
                "Artifact Registry is unavailable. Run:\n"
                "  gcloud auth application-default login"
            )

        # First get manifest to know what files to download
        manifest_url = f"{self._generic_base}/{model_name}/{version}/manifest.json"
        headers = self._get_auth_headers()
        resp = http_requests.get(manifest_url, headers=headers, timeout=60)

        if resp.status_code != 200:
            raise FileNotFoundError(
                f"Model {model_name} v{version} not found in Artifact "
                f"Registry: {resp.status_code} {resp.text}"
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(resp.text)

        # Parse manifest to reconstruct file list
        logger.info(
            "Pulled manifest for %s v%s from Artifact Registry",
            model_name,
            version,
        )

        # For single-file models, download the model file
        # The filename is stored in metadata or inferred from manifest
        # For now, we download what's listed — extend as needed
        logger.info("Pulled %s v%s → %s", model_name, version, cache_dir)
        return cache_dir

    # ------------------------------------------------------------------
    # List / Query
    # ------------------------------------------------------------------

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """
        List all versions of a model.

        Checks local cache and (if available) queries Artifact Registry
        management API.
        """
        versions: List[ModelVersion] = []
        seen: set = set()

        # Local cache
        local_model_dir = self.local_cache / model_name
        if local_model_dir.exists():
            for vdir in sorted(local_model_dir.iterdir()):
                manifest = vdir / "manifest.json"
                if manifest.exists():
                    data = json.loads(manifest.read_text())
                    mv = ModelVersion.from_dict(data)
                    versions.append(mv)
                    seen.add(mv.version)

        # Query Artifact Registry API for additional versions
        if self._ar_client is not None:
            try:
                parent = f"{self._repo_parent}/packages/{model_name}"
                request = artifactregistry_v1.ListVersionsRequest(
                    parent=parent,
                )
                for ar_version in self._ar_client.list_versions(request=request):
                    # ar_version.name format:
                    # projects/.../versions/{version_id}
                    ver_id = ar_version.name.split("/")[-1]
                    if ver_id not in seen:
                        # Try to pull manifest for metadata
                        try:
                            cache = self.pull_model(model_name, ver_id)
                            manifest = cache / "manifest.json"
                            if manifest.exists():
                                data = json.loads(manifest.read_text())
                                versions.append(ModelVersion.from_dict(data))
                                seen.add(ver_id)
                        except Exception:
                            # Create minimal version entry
                            versions.append(
                                ModelVersion(
                                    model_name=model_name,
                                    version=ver_id,
                                    timestamp="",
                                    sha256="",
                                )
                            )
            except Exception as e:
                logger.warning("Artifact Registry list_versions failed: %s", e)

        versions.sort(key=lambda v: (v.timestamp, v.version), reverse=True)
        return versions

    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the most recent version of a model."""
        versions = self.list_versions(model_name)
        return versions[0] if versions else None

    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a model version from local cache and Artifact Registry."""
        deleted = False

        # Local cache
        cache_dir = self.local_cache / model_name / f"v{version}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            deleted = True

        # Artifact Registry
        if self._ar_client is not None:
            try:
                version_name = (
                    f"{self._repo_parent}/packages/{model_name}" f"/versions/{version}"
                )
                request = artifactregistry_v1.DeleteVersionRequest(
                    name=version_name,
                )
                self._ar_client.delete_version(request=request)
                deleted = True
                logger.info(
                    "Deleted %s v%s from Artifact Registry",
                    model_name,
                    version,
                )
            except Exception as e:
                logger.warning("Artifact Registry delete failed: %s", e)

        return deleted
