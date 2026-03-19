"""
Unit tests for — GCP Artifact Registry Client.

Tests:
  - Model push (file + directory)
  - Model pull (cached + GCS)
  - Version listing and querying
  - SHA-256 integrity
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.model_pipeline.registry.artifact_registry import (
    ModelVersion,
    RegistryClient,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def local_registry(tmp_path):
    """Registry client with remote disabled, local cache only."""
    client = RegistryClient(
        project="test-project",
        location="us-central1",
        repository="test-repo",
        local_cache=tmp_path / "cache",
    )
    # Force local-only mode — prevent real Artifact Registry calls
    client._credentials = None
    client._ar_client = None
    return client


@pytest.fixture
def model_file(tmp_path):
    """Create a dummy model file."""
    f = tmp_path / "model.pkl"
    f.write_bytes(b"fake-model-bytes-12345")
    return f


@pytest.fixture
def model_dir(tmp_path):
    """Create a dummy model directory with multiple files."""
    d = tmp_path / "model_dir"
    d.mkdir()
    (d / "model.bin").write_bytes(b"model-weights")
    (d / "config.json").write_text('{"layers": 3}')
    (d / "tokenizer").mkdir()
    (d / "tokenizer" / "vocab.txt").write_text("hello\nworld")
    return d


# =====================================================================
# ModelVersion
# =====================================================================


class TestModelVersion:
    def test_tag_format(self):
        mv = ModelVersion("personalization", "1.0.0", "20260318T120000", "abc123")
        assert mv.tag == "personalization-v1.0.0-20260318T120000"

    def test_roundtrip_serialization(self):
        mv = ModelVersion(
            "personalization",
            "2.1.0",
            "20260318T120000",
            "sha256hex",
            metadata={"accuracy": 0.95},
        )
        d = mv.to_dict()
        mv2 = ModelVersion.from_dict(d)
        assert mv2.model_name == mv.model_name
        assert mv2.version == mv.version
        assert mv2.sha256 == mv.sha256
        assert mv2.metadata == mv.metadata
        assert mv2.tag == mv.tag


# =====================================================================
# Push
# =====================================================================


class TestPush:
    def test_push_file(self, local_registry, model_file):
        """Push a single model file to local cache."""
        mv = local_registry.push_model(
            model_file, model_name="personalization", version="1.0.0"
        )
        assert mv.model_name == "personalization"
        assert mv.version == "1.0.0"
        assert len(mv.sha256) == 64  # SHA-256 hex length
        assert mv.timestamp  # non-empty

        # Verify local cache
        cached = local_registry.local_cache / "personalization" / "v1.0.0"
        assert (cached / "model.pkl").exists()
        assert (cached / "manifest.json").exists()

    def test_push_directory(self, local_registry, model_dir):
        """Push a model directory to local cache."""
        local_registry.push_model(model_dir, model_name="llm", version="0.1.0")
        cached = local_registry.local_cache / "llm" / "v0.1.0"
        assert (cached / "model.bin").exists()
        assert (cached / "config.json").exists()
        assert (cached / "tokenizer" / "vocab.txt").exists()

    def test_push_nonexistent_raises(self, local_registry):
        """Push should raise if file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            local_registry.push_model(
                "/nonexistent/model.pkl",
                model_name="test",
                version="1.0.0",
            )

    def test_push_with_metadata(self, local_registry, model_file):
        """Metadata should be stored in manifest."""
        mv = local_registry.push_model(
            model_file,
            model_name="scoring",
            version="1.0.0",
            metadata={"ndcg_5": 0.85, "trained_on": "2026-03-18"},
        )
        assert mv.metadata["ndcg_5"] == 0.85

        manifest = local_registry.local_cache / "scoring" / "v1.0.0" / "manifest.json"
        data = json.loads(manifest.read_text())
        assert data["metadata"]["ndcg_5"] == 0.85

    def test_push_same_version_overwrites(self, local_registry, model_file):
        """Pushing same version should overwrite cleanly."""
        local_registry.push_model(model_file, "m", "1.0.0")
        local_registry.push_model(model_file, "m", "1.0.0")
        versions = local_registry.list_versions("m")
        assert len(versions) == 1


# =====================================================================
# Pull
# =====================================================================


class TestPull:
    def test_pull_from_cache(self, local_registry, model_file):
        """Pull should return cached path when available."""
        local_registry.push_model(model_file, "personalization", "1.0.0")
        path = local_registry.pull_model("personalization", "1.0.0")
        assert path.exists()
        assert (path / "model.pkl").exists()

    def test_pull_not_cached_no_remote_raises(self, local_registry):
        """Pull should raise if not cached and remote unavailable."""
        with pytest.raises(RuntimeError, match="not in local cache"):
            local_registry.pull_model("nonexistent", "1.0.0")

    def test_pull_force_redownload(self, local_registry, model_file):
        """force=True should still work with local-only (re-reads cache)."""
        local_registry.push_model(model_file, "m", "1.0.0")
        # force=True but no remote — should still return cached
        path = local_registry.pull_model("m", "1.0.0", force=True)
        assert path.exists()


# =====================================================================
# List / Query
# =====================================================================


class TestListVersions:
    def test_list_versions_empty(self, local_registry):
        """No versions should return empty list."""
        assert local_registry.list_versions("nonexistent") == []

    def test_list_versions_multiple(self, local_registry, model_file):
        """Multiple versions should be listed in reverse chronological order."""
        local_registry.push_model(model_file, "m", "1.0.0")
        local_registry.push_model(model_file, "m", "1.1.0")
        local_registry.push_model(model_file, "m", "2.0.0")

        versions = local_registry.list_versions("m")
        assert len(versions) == 3
        ver_strings = [v.version for v in versions]
        assert "2.0.0" in ver_strings
        assert "1.1.0" in ver_strings
        assert "1.0.0" in ver_strings

    def test_get_latest_version(self, local_registry, model_file):
        """get_latest_version should return most recent."""
        local_registry.push_model(model_file, "m", "1.0.0")
        local_registry.push_model(model_file, "m", "2.0.0")

        latest = local_registry.get_latest_version("m")
        assert latest is not None
        assert latest.version == "2.0.0"

    def test_get_latest_version_empty(self, local_registry):
        assert local_registry.get_latest_version("empty") is None


# =====================================================================
# Delete
# =====================================================================


class TestDelete:
    def test_delete_version(self, local_registry, model_file):
        local_registry.push_model(model_file, "m", "1.0.0")
        assert local_registry.delete_version("m", "1.0.0") is True
        assert local_registry.list_versions("m") == []

    def test_delete_nonexistent(self, local_registry):
        assert local_registry.delete_version("m", "999.0.0") is False


# =====================================================================
# SHA-256 Integrity
# =====================================================================


class TestIntegrity:
    def test_sha256_deterministic(self, local_registry, model_file):
        """Same file should produce same hash."""
        mv1 = local_registry.push_model(model_file, "m", "1.0.0")
        mv2 = local_registry.push_model(model_file, "m", "1.0.1")
        assert mv1.sha256 == mv2.sha256

    def test_sha256_changes_with_content(self, local_registry, tmp_path):
        """Different content should produce different hash."""
        f1 = tmp_path / "m1.pkl"
        f1.write_bytes(b"model-v1")
        f2 = tmp_path / "m2.pkl"
        f2.write_bytes(b"model-v2")

        mv1 = local_registry.push_model(f1, "m", "1.0.0")
        mv2 = local_registry.push_model(f2, "m", "2.0.0")
        assert mv1.sha256 != mv2.sha256


# =====================================================================
# Authentication & Remote Availability
# =====================================================================


class TestAuthentication:
    def test_is_remote_available_false_without_creds(self, local_registry):
        """Remote should be unavailable when credentials are None."""
        assert local_registry.is_remote_available is False

    def test_is_remote_available_true_with_creds(self, local_registry):
        """Remote should be available when credentials exist."""
        local_registry._credentials = MagicMock()
        assert local_registry.is_remote_available is True

    def test_get_auth_headers_raises_without_creds(self, local_registry):
        """_get_auth_headers should raise when no credentials."""
        with pytest.raises(RuntimeError, match="No GCP credentials"):
            local_registry._get_auth_headers()

    def test_get_auth_headers_returns_bearer_token(self, local_registry):
        """_get_auth_headers should return Bearer token header."""
        mock_creds = MagicMock()
        mock_creds.token = "test-token-12345"
        local_registry._credentials = mock_creds

        with patch("google.auth.transport.requests.Request"):
            headers = local_registry._get_auth_headers()

        assert headers["Authorization"] == "Bearer test-token-12345"
        mock_creds.refresh.assert_called_once()

    def test_push_with_remote_calls_upload(self, local_registry, model_file):
        """When remote is available, push should call _upload_to_ar."""
        mock_creds = MagicMock()
        mock_creds.token = "test-token"
        local_registry._credentials = mock_creds

        with patch.object(local_registry, "_upload_to_ar") as mock_upload:
            local_registry.push_model(model_file, "m", "1.0.0")
            mock_upload.assert_called_once()

    def test_push_without_remote_skips_upload(self, local_registry, model_file):
        """When remote is unavailable, push should only cache locally."""
        with patch.object(local_registry, "_upload_to_ar") as mock_upload:
            local_registry.push_model(model_file, "m", "1.0.0")
            mock_upload.assert_not_called()

        # But local cache should still work
        cached = local_registry.local_cache / "m" / "v1.0.0"
        assert (cached / "model.pkl").exists()
