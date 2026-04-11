"""Feedback persistence logic (Story 4.1)."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from src.app.db.models import FeedbackEvent
from src.app.feedback.schemas import FeedbackCreateRequest

logger = logging.getLogger(__name__)


def create_feedback(
    db: Session,
    user_id: int,
    req: FeedbackCreateRequest,
) -> FeedbackEvent:
    """Persist a feedback event and return it."""
    event = FeedbackEvent(
        user_id=user_id,
        card_id=req.card_id,
        recommendation_event_id=req.recommendation_event_id,
        reaction=req.reaction,
        reason_tag=req.reason_tag,
        target=req.target,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    logger.info(
        "Feedback recorded: id=%s user=%s card=%s reaction=%s",
        event.id,
        user_id,
        req.card_id,
        req.reaction,
    )
    return event
