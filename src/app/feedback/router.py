"""Feedback API endpoint (Story 4.1)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from src.app.auth.dependencies import get_current_user
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.feedback import service
from src.app.feedback.schemas import FeedbackCreateRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    payload: FeedbackCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """Record a like/dislike on a recommendation card or explanation."""
    event = service.create_feedback(db, current_user.id, payload)
    return FeedbackResponse(ok=True, feedback_id=event.id)
