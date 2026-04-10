"""
SQLAlchemy ORM models for RewardSense application DB.

Story 1.1 tables: users, auth_credentials.
Story 1.2 tables: user_settings, user_personas, saved_cards.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Float
)
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
        "AuthCredential",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    settings = relationship(
        "UserSettings",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    personas = relationship(
        "UserPersona",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    saved_cards = relationship(
        "SavedCard",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    transaction_logs = relationship(
    "TransactionLog",
    back_populates="user",
    cascade="all, delete-orphan",
    )
    recommendation_events = relationship(
    "RecommendationEvent",
    back_populates="user",
    cascade="all, delete-orphan",
    )


class AuthCredential(Base):
    __tablename__ = "auth_credentials"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    password_hash = Column(String, nullable=False)

    user = relationship("User", back_populates="credential")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    reward_preference = Column(String, nullable=False, default="cashback")
    transaction_logging_enabled = Column(Boolean, nullable=False, default=False)
    dark_mode = Column(Boolean, nullable=False, default=False)

    user = relationship("User", back_populates="settings")


class UserPersona(Base):
    __tablename__ = "user_personas"
    __table_args__ = (UniqueConstraint("user_id", "persona_key"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    persona_key = Column(String, nullable=False)

    user = relationship("User", back_populates="personas")


class SavedCard(Base):
    __tablename__ = "saved_cards"
    __table_args__ = (UniqueConstraint("user_id", "card_id"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    card_id = Column(String, nullable=False)
    added_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = relationship("User", back_populates="saved_cards")

class RecommendationEvent(Base):
    """
    Captures a recommendation request for later linkage to transactions/feedback.
    """
    __tablename__ = "recommendation_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    flow = Column(String, nullable=False)  # "portfolio" | "transaction"
    top_card_id = Column(String, nullable=True)
    top_card_name = Column(String, nullable=True)
    request_payload = Column(String, nullable=True)  # JSON string
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = relationship("User", back_populates="recommendation_events")
    transaction_logs = relationship(
        "TransactionLog",
        back_populates="recommendation_event",
    )

class TransactionLog(Base):
    """
    User-owned transaction log entry
    """
    __tablename__ = "transaction_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    merchant = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    chosen_card_id = Column(String, nullable=True)
    chosen_card_name = Column(String, nullable=True)
    reward_earned = Column(Float, nullable=False, default=0.0)
    estimated_savings = Column(Float, nullable=False, default=0.0)
    source_flow = Column(String, nullable=False, default="manual")  # "manual" | "portfolio" | "transaction"
    card_was_saved = Column(Boolean, nullable=False, default=False)
    recommendation_event_id = Column(
        Integer, ForeignKey("recommendation_events.id"), nullable=True
    )
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = relationship("User", back_populates="transaction_logs")
    recommendation_event = relationship(
        "RecommendationEvent", back_populates="transaction_logs"
    )