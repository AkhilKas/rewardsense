"""Shared fixtures for app-layer tests (auth, profile, personas)."""

from __future__ import annotations

import os

import pytest

# Ensure JWT_SECRET_KEY is set before any app module is imported in tests
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("DB_PATH", ":memory:")  # SQLite in-memory for tests

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("passlib")
pytest.importorskip("jose")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.app.db.database import Base, get_db  # noqa: E402
import src.app.db.models  # noqa: F401, E402 — registers ORM models with Base before create_all
from src.app.server import create_app  # noqa: E402


@pytest.fixture(scope="function")
def test_client():
    """TestClient backed by a fresh in-memory SQLite DB per test.

    StaticPool is required so that create_all and subsequent sessions all share
    the same underlying SQLite connection (each :memory: URL otherwise gets an
    independent, empty database).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client

    Base.metadata.drop_all(bind=engine)
