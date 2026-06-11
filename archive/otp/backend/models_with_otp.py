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
    Float,
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
    is_verified = Column(Boolean, nullable=False, default=False)
    email_otp_hash = Column(String, nullable=True)
    otp_expires_at = Column(DateTime(timezone=True), nullable=True)

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
    feedbacks = relationship(
        "FeedbackEvent",
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
    feedbacks = relationship(
        "FeedbackEvent",
        back_populates="recommendation_event",
    )
    llm_telemetry_events = relationship(
        "LLMTelemetryEvent",
        back_populates="recommendation_event",
    )


class FeedbackEvent(Base):
    """
    Captures like/dislike feedback on recommendation cards or explanations.
    Story 4.1 — data for future retraining and analytics, no live ranking change.
    """

    __tablename__ = "feedback_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    card_id = Column(String, nullable=False)
    recommendation_event_id = Column(
        Integer, ForeignKey("recommendation_events.id"), nullable=True
    )
    reaction = Column(String, nullable=False)  # "like" | "dislike"
    reason_tag = Column(String, nullable=True)  # optional reason tag
    target = Column(String, nullable=False, default="card")  # "card" | "explanation"
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = relationship("User", back_populates="feedbacks")
    recommendation_event = relationship(
        "RecommendationEvent", back_populates="feedbacks"
    )


class LLMTelemetryEvent(Base):
    """
    Per-explanation telemetry for prompt drift tracking and quality monitoring.
    Story 4.3 — internal-only, not exposed to frontend.
    """

    __tablename__ = "llm_telemetry_events"

    id = Column(Integer, primary_key=True, index=True)
    recommendation_event_id = Column(
        Integer, ForeignKey("recommendation_events.id"), nullable=True
    )
    card_id = Column(String, nullable=True)
    prompt_version_hash = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False)
    temperature = Column(Float, nullable=False, default=0.0)
    latency_ms = Column(Float, nullable=False)
    used_fallback = Column(Boolean, nullable=False, default=False)
    fallback_reason = Column(String, nullable=True)
    token_estimate = Column(Integer, nullable=True)
    cost_estimate_usd = Column(Float, nullable=True)
    output_quality_score = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    recommendation_event = relationship(
        "RecommendationEvent", back_populates="llm_telemetry_events"
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
    source_flow = Column(
        String, nullable=False, default="manual"
    )  # "manual" | "portfolio" | "transaction"
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
