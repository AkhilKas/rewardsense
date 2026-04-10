"""
FastAPI router for transaction ledger, summary, and export.

Endpoints:
    POST   /transactions          - log a transaction
    GET    /transactions          - paginated history
    GET    /summary               - aggregated summary
    GET    /transactions/export   - CSV or XLSX download
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from src.app.auth.dependencies import get_current_user
from src.app.db.database import get_db
from src.app.db.models import User
from src.app.transactions import service
from src.app.transactions.schemas import (
    TransactionCreateRequest,
    TransactionListResponse,
    TransactionResponse,
    TransactionSummaryResponse,
)

router = APIRouter(tags=["transactions"])


# ---------------------------------------------------------------------------
# Story 3.1 + 3.2: Create and list transactions
# ---------------------------------------------------------------------------


@router.post("/transactions", response_model=TransactionResponse, status_code=201)
def create_transaction(
    body: TransactionCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TransactionResponse:
    """Log a transaction. Requires transaction logging to be enabled in user settings."""
    return service.create_transaction(db, user, body)


@router.get("/transactions", response_model=TransactionListResponse)
def list_transactions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TransactionListResponse:
    """Get paginated transaction history for the authenticated user."""
    return service.list_transactions(db, user, page=page, page_size=page_size)


# ---------------------------------------------------------------------------
# Story 3.3: Summary
# ---------------------------------------------------------------------------


@router.get("/summary", response_model=TransactionSummaryResponse)
def get_summary(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TransactionSummaryResponse:
    """Aggregated spend/savings summary with chart-ready data."""
    return service.get_summary(db, user)


# ---------------------------------------------------------------------------
# Story 3.4: Export
# ---------------------------------------------------------------------------


@router.get("/transactions/export")
def export_transactions(
    format: str = Query(default="csv", pattern="^(csv|xlsx)$"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Download transaction history as CSV or XLSX."""
    if format == "xlsx":
        content = service.export_xlsx(db, user)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": "attachment; filename=rewardsense_transactions.xlsx"
            },
        )

    content = service.export_csv(db, user)
    return Response(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=rewardsense_transactions.csv"
        },
    )
