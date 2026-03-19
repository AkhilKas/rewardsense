"""Vertex AI Gemini client adapter for ExplanationGenerator."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class VertexGeminiClient:
    """Minimal Vertex AI Gemini REST client.

    This adapter is intentionally lightweight so it can be mocked in tests and
    swapped with alternative providers.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        timeout_sec: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location
        self.model = model
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.session = session or requests.Session()

        if not self.project_id:
            raise ValueError("project_id is required (or set GCP_PROJECT_ID)")

    def generate(self, system_message: str, user_message: str, **kwargs: Any) -> str:
        """Generate text via Vertex AI Gemini `generateContent` API."""
        model = kwargs.get("model", self.model)
        temperature = float(kwargs.get("temperature", self.temperature))

        token = self._get_access_token()
        endpoint = self._build_endpoint(model)

        payload: Dict[str, Any] = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_message}],
            },
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {"temperature": temperature},
        }

        response = self.session.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_sec,
        )

        if not response.ok:
            raise RuntimeError(
                f"Vertex Gemini request failed: {response.status_code} {response.text}"
            )

        body = response.json()
        candidates = body.get("candidates", [])
        if not candidates:
            raise RuntimeError("Vertex Gemini response missing candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        output = "".join(text_parts).strip()
        if not output:
            raise RuntimeError("Vertex Gemini response had empty text output")
        return output

    def _build_endpoint(self, model: str) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project_id}/locations/{self.location}/publishers/google/models/"
            f"{model}:generateContent"
        )

    @staticmethod
    def _get_access_token() -> str:
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise RuntimeError(
                "google-auth is required for VertexGeminiClient authentication"
            ) from exc

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())

        token = getattr(credentials, "token", None)
        if not token:
            raise RuntimeError("Failed to obtain GCP access token")
        return token
