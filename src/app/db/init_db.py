"""Create all DB tables on startup."""

from __future__ import annotations

from src.app.db.database import Base, engine
import src.app.db.models  # noqa: F401 — ensures all ORM models are registered before create_all


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)
