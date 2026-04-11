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

# ---------------------------------------------------------------------------
# Fixed 5-card test catalog — deterministic for all app tests.
# Matches the original hardcoded catalog so existing assertions hold.
# ---------------------------------------------------------------------------
from src.app.cards import catalog as _catalog_mod  # noqa: E402
from src.app.users.schemas import CardCatalogItem  # noqa: E402

_TEST_CARDS = [
    {
        "card_id": "chase_sapphire_preferred",
        "card_name": "Chase Sapphire Preferred",
        "issuer": "Chase",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 3.0, "travel": 2.0},
        },
        "key_benefits": ["3x dining", "2x travel", "1x everything else"],
    },
    {
        "card_id": "amex_gold",
        "card_name": "Amex Gold",
        "issuer": "American Express",
        "annual_fee": 250.0,
        "reward_rates": {
            "universal_base_rate": 1.0,
            "category_bonuses": {"dining": 4.0, "groceries": 4.0, "travel": 3.0},
        },
        "key_benefits": ["4x dining", "4x groceries", "3x travel"],
    },
    {
        "card_id": "citi_double_cash",
        "card_name": "Citi Double Cash",
        "issuer": "Citi",
        "annual_fee": 0.0,
        "reward_rates": {"universal_base_rate": 2.0},
        "key_benefits": ["2% cash back on everything"],
    },
    {
        "card_id": "capital_one_venture",
        "card_name": "Capital One Venture",
        "issuer": "Capital One",
        "annual_fee": 95.0,
        "reward_rates": {
            "universal_base_rate": 2.0,
            "category_bonuses": {"travel": 5.0},
        },
        "key_benefits": ["5x travel via Capital One", "2x everything else"],
    },
    {
        "card_id": "discover_it",
        "card_name": "Discover it Cash Back",
        "issuer": "Discover",
        "annual_fee": 0.0,
        "reward_rates": {"universal_base_rate": 1.0},
        "key_benefits": ["5% rotating categories", "1% everything else"],
    },
]

_TEST_CATALOG_BY_ID = {c["card_id"]: c for c in _TEST_CARDS}
_CARD_IMAGE_PATH = "/cards/{card_id}.svg"
_TEST_DISPLAY = [
    CardCatalogItem(
        card_id=c["card_id"],
        card_name=c["card_name"],
        issuer=c.get("issuer", "Unknown"),
        annual_fee=c["annual_fee"],
        reward_highlights=c.get("key_benefits", []),
        image_url=_CARD_IMAGE_PATH.format(card_id=c["card_id"]),
    )
    for c in _TEST_CARDS
]
_TEST_DISPLAY_BY_ID = {c.card_id: c for c in _TEST_DISPLAY}

# Patch the shared catalog module so every downstream import sees 5 test cards
_catalog_mod.CARD_CATALOG = _TEST_CARDS
_catalog_mod.CARD_CATALOG_BY_ID = _TEST_CATALOG_BY_ID
_catalog_mod.DISPLAY_CATALOG = _TEST_DISPLAY
_catalog_mod.DISPLAY_CATALOG_BY_ID = _TEST_DISPLAY_BY_ID

# Also patch the router's local aliases (imported at module load time)
import src.app.users.router as _router_mod  # noqa: E402

_router_mod._CATALOG = _TEST_DISPLAY
_router_mod._CATALOG_BY_ID = _TEST_DISPLAY_BY_ID


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
