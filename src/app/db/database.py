"""SQLAlchemy engine and session factory for RewardSense application DB."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


def _build_database_url() -> str:
    """Return DATABASE_URL from env, falling back to local SQLite."""
    url = os.getenv("DATABASE_URL")
    if url:
        # Cloud SQL via pg8000/psycopg2 — normalise the common Heroku-style prefix
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url
    # Local development default
    db_path = os.getenv("DB_PATH", "./rewardsense.db")
    return f"sqlite:///{db_path}"


DATABASE_URL = _build_database_url()

_connect_args: dict = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=_connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
