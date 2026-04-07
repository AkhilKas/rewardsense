"""SQLAlchemy ORM models for RewardSense application DB.

Story 1.1 tables: users, auth_credentials.
Story 1.2 will add: user_settings, user_personas, saved_cards.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    credential = relationship(
        "AuthCredential", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class AuthCredential(Base):
    __tablename__ = "auth_credentials"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    password_hash = Column(String, nullable=False)

    user = relationship("User", back_populates="credential")
