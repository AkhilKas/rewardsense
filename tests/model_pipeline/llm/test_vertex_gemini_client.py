import pytest

from src.model_pipeline.llm.vertex_gemini_client import VertexGeminiClient


class _FakeResponse:
    def __init__(self, ok=True, status_code=200, body=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return self.response


def test_generate_happy_path(monkeypatch):
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"summary":"Use Amex Gold.","rationale":["4x dining"],"confidence":0.9}'
                        }
                    ]
                }
            }
        ]
    }
    session = _FakeSession(_FakeResponse(ok=True, body=body))

    monkeypatch.setattr(
        VertexGeminiClient,
        "_get_access_token",
        staticmethod(lambda: "fake-token"),
    )

    client = VertexGeminiClient(project_id="demo-project", session=session)
    output = client.generate("system", "user")

    assert "Use Amex Gold" in output
    assert session.calls
    assert session.calls[0]["headers"]["Authorization"] == "Bearer fake-token"


def test_generate_raises_on_http_error(monkeypatch):
    session = _FakeSession(_FakeResponse(ok=False, status_code=500, text="boom"))
    monkeypatch.setattr(
        VertexGeminiClient,
        "_get_access_token",
        staticmethod(lambda: "fake-token"),
    )

    client = VertexGeminiClient(project_id="demo-project", session=session)
    with pytest.raises(RuntimeError):
        client.generate("system", "user")


def test_get_access_token_without_google_auth(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("google.auth") or name == "google":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError):
        VertexGeminiClient._get_access_token()
