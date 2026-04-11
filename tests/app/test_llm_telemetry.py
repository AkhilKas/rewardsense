"""Tests for Story 4.3: LLM telemetry and prompt drift tracking."""

from __future__ import annotations

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("DB_PATH", ":memory:")

import pytest  # noqa: E402

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.app.db.database import Base  # noqa: E402
from src.app.db.models import LLMTelemetryEvent  # noqa: E402
from src.app.telemetry.llm_telemetry import (  # noqa: E402
    compute_quality_score,
    log_llm_telemetry,
)

import src.app.db.models  # noqa: F401, E402 — registers ORM models


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_log_llm_telemetry_persists(db_session):
    event = log_llm_telemetry(
        db_session,
        prompt_version_hash="abc123",
        model_name="gemini-2.5-flash",
        temperature=0.0,
        latency_ms=150.5,
        used_fallback=False,
        card_id="amex_gold",
    )

    assert event.id is not None
    assert event.prompt_version_hash == "abc123"
    assert event.model_name == "gemini-2.5-flash"
    assert event.latency_ms == 150.5
    assert event.used_fallback is False

    # Verify it's in the DB
    row = db_session.query(LLMTelemetryEvent).filter_by(id=event.id).first()
    assert row is not None
    assert row.card_id == "amex_gold"


def test_log_llm_telemetry_with_fallback(db_session):
    event = log_llm_telemetry(
        db_session,
        prompt_version_hash="def456",
        model_name="gemini-2.5-flash",
        temperature=0.0,
        latency_ms=50.0,
        used_fallback=True,
        fallback_reason="quality_filter_failed",
        output_quality_score=0.5,
    )

    assert event.used_fallback is True
    assert event.fallback_reason == "quality_filter_failed"
    assert event.output_quality_score == 0.5


def test_compute_quality_score():
    assert compute_quality_score({}) == 0.0
    assert compute_quality_score({"a": True, "b": True}) == 1.0
    assert compute_quality_score({"a": True, "b": False}) == 0.5
    assert compute_quality_score({"a": True, "b": True, "c": False, "d": True}) == 0.75


def test_prompt_hash_consistency():
    """Same prompt should produce the same hash."""
    from src.model_pipeline.llm.explanation_generator import _compute_prompt_hash

    h1 = _compute_prompt_hash("system msg", "user msg")
    h2 = _compute_prompt_hash("system msg", "user msg")
    h3 = _compute_prompt_hash("different system", "user msg")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16  # truncated SHA-256
