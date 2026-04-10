"""
Transaction ledger CRUD, summary aggregation, and export.

- Opt-in check before persisting.
- Full schema persistence.
- Aggregated summary computation.
- CSV/XLSX export.
"""

from __future__ import annotations

import csv
import io
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.app.db.models import SavedCard, TransactionLog, User
from src.app.transactions.schemas import (
    VALID_SOURCE_FLOWS,
    CardSummary,
    CategorySummary,
    TopInsight,
    TransactionCreateRequest,
    TransactionListResponse,
    TransactionResponse,
    TransactionSummaryResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Cards and their annual fees for fee-adjusted savings calculation.
# In production this would come from the card catalog; hardcoded here
# to avoid a circular import with the serving layer.
_CARD_ANNUAL_FEES: Dict[str, float] = {
    "amex_gold": 250.0,
    "chase_sapphire_preferred": 95.0,
    "capital_one_venture_x": 395.0,
    "citi_double_cash": 0.0,
    "blue_cash_preferred": 95.0,
    "capital_one_savor": 0.0,
    "chase_freedom_unlimited": 0.0,
    "wells_fargo_autograph": 0.0,
    "discover_it_cash_back": 0.0,
}


def _txn_to_response(txn: TransactionLog) -> TransactionResponse:
    return TransactionResponse(
        id=txn.id,
        merchant=txn.merchant,
        amount=txn.amount,
        category=txn.category,
        chosen_card_id=txn.chosen_card_id,
        chosen_card_name=txn.chosen_card_name,
        reward_earned=txn.reward_earned,
        estimated_savings=txn.estimated_savings,
        source_flow=txn.source_flow,
        card_was_saved=txn.card_was_saved,
        recommendation_event_id=txn.recommendation_event_id,
        timestamp=txn.timestamp.isoformat() if txn.timestamp else "",
    )


def _check_logging_enabled(user: User) -> None:
    """Raise 403 if the user has not opted in to transaction logging."""
    settings = user.settings
    if settings is None or not settings.transaction_logging_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Transaction logging is disabled. "
                "Enable it in your profile settings to start recording transactions."
            ),
        )


# ---------------------------------------------------------------------------
# Story 3.1 + 3.2: Create transaction
# ---------------------------------------------------------------------------


def create_transaction(
    db: Session,
    user: User,
    req: TransactionCreateRequest,
) -> TransactionResponse:
    """Persist a transaction log entry if opt-in is enabled."""
    _check_logging_enabled(user)

    if req.source_flow not in VALID_SOURCE_FLOWS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid source_flow '{req.source_flow}'. "
                f"Valid options: {sorted(VALID_SOURCE_FLOWS)}"
            ),
        )

    # Determine if the chosen card is currently in the user's saved wallet
    card_was_saved = False
    if req.chosen_card_id:
        saved = (
            db.query(SavedCard)
            .filter(
                SavedCard.user_id == user.id,
                SavedCard.card_id == req.chosen_card_id,
            )
            .first()
        )
        card_was_saved = saved is not None

    # Parse timestamp or default to now
    if req.timestamp:
        try:
            ts = datetime.fromisoformat(req.timestamp.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid timestamp format. Use ISO-8601.",
            )
    else:
        ts = datetime.now(timezone.utc)

    txn = TransactionLog(
        user_id=user.id,
        merchant=req.merchant,
        amount=req.amount,
        category=req.category.lower().strip(),
        chosen_card_id=req.chosen_card_id,
        chosen_card_name=req.chosen_card_name,
        reward_earned=req.reward_earned,
        estimated_savings=req.estimated_savings,
        source_flow=req.source_flow,
        card_was_saved=card_was_saved,
        recommendation_event_id=req.recommendation_event_id,
        timestamp=ts,
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)
    return _txn_to_response(txn)


# ---------------------------------------------------------------------------
# Story 3.1: List transactions (paginated)
# ---------------------------------------------------------------------------


def list_transactions(
    db: Session,
    user: User,
    page: int = 1,
    page_size: int = 20,
) -> TransactionListResponse:
    """Return paginated transaction history for the user."""
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    page_size = min(page_size, 100)

    total = (
        db.query(func.count(TransactionLog.id))
        .filter(TransactionLog.user_id == user.id)
        .scalar()
    ) or 0

    offset = (page - 1) * page_size
    rows = (
        db.query(TransactionLog)
        .filter(TransactionLog.user_id == user.id)
        .order_by(TransactionLog.timestamp.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return TransactionListResponse(
        transactions=[_txn_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


# ---------------------------------------------------------------------------
# Story 3.3: Summary
# ---------------------------------------------------------------------------


def get_summary(db: Session, user: User) -> TransactionSummaryResponse:
    """Compute aggregated summary from the user's transaction history."""
    rows: List[TransactionLog] = (
        db.query(TransactionLog).filter(TransactionLog.user_id == user.id).all()
    )

    if not rows:
        return TransactionSummaryResponse(
            spend_by_category=[],
            rewards_by_category=[],
            savings_by_card=[],
            total_spend=0.0,
            total_rewards=0.0,
            total_savings=0.0,
            fee_adjusted_savings=0.0,
            transaction_count=0,
            top_insights=[
                TopInsight(
                    label="No data yet",
                    value="Log some transactions to see your spending insights.",
                )
            ],
        )

    # --- Aggregate by category ---
    cat_agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"spend": 0.0, "reward": 0.0, "savings": 0.0, "count": 0}
    )
    for txn in rows:
        cat = txn.category or "other"
        cat_agg[cat]["spend"] += txn.amount
        cat_agg[cat]["reward"] += txn.reward_earned
        cat_agg[cat]["savings"] += txn.estimated_savings
        cat_agg[cat]["count"] += 1

    spend_by_category = [
        CategorySummary(
            category=cat,
            total_spend=round(vals["spend"], 2),
            total_reward=round(vals["reward"], 2),
            total_savings=round(vals["savings"], 2),
            transaction_count=int(vals["count"]),
        )
        for cat, vals in sorted(
            cat_agg.items(), key=lambda x: x[1]["spend"], reverse=True
        )
    ]

    # --- Aggregate by card ---
    card_agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "card_name": None,
            "spend": 0.0,
            "reward": 0.0,
            "savings": 0.0,
            "count": 0,
        }
    )
    for txn in rows:
        card_key = txn.chosen_card_id or "unknown"
        card_agg[card_key]["card_name"] = txn.chosen_card_name or card_key
        card_agg[card_key]["spend"] += txn.amount
        card_agg[card_key]["reward"] += txn.reward_earned
        card_agg[card_key]["savings"] += txn.estimated_savings
        card_agg[card_key]["count"] += 1

    savings_by_card = [
        CardSummary(
            card_id=cid if cid != "unknown" else None,
            card_name=vals["card_name"],
            total_spend=round(vals["spend"], 2),
            total_reward=round(vals["reward"], 2),
            total_savings=round(vals["savings"], 2),
            transaction_count=int(vals["count"]),
        )
        for cid, vals in sorted(
            card_agg.items(), key=lambda x: x[1]["savings"], reverse=True
        )
    ]

    # --- Totals ---
    total_spend = round(sum(t.amount for t in rows), 2)
    total_rewards = round(sum(t.reward_earned for t in rows), 2)
    total_savings = round(sum(t.estimated_savings for t in rows), 2)

    # Fee-adjusted: subtract annual fees (amortized monthly) for cards used
    total_annual_fees = sum(
        _CARD_ANNUAL_FEES.get(cid, 0.0) for cid in card_agg if cid != "unknown"
    )
    fee_adjusted_savings = round(total_savings - total_annual_fees, 2)

    # --- Top insights ---
    insights: List[TopInsight] = []
    if spend_by_category:
        top_cat = spend_by_category[0]
        insights.append(
            TopInsight(
                label="Top spending category",
                value=f"{top_cat.category}: ${top_cat.total_spend:,.2f}",
            )
        )
    if savings_by_card:
        top_card = savings_by_card[0]
        insights.append(
            TopInsight(
                label="Most savings from",
                value=f"{top_card.card_name}: ${top_card.total_savings:,.2f}",
            )
        )
    insights.append(
        TopInsight(
            label="Total transactions",
            value=str(len(rows)),
        )
    )
    if total_rewards > 0:
        reward_rate = (total_rewards / total_spend * 100) if total_spend > 0 else 0
        insights.append(
            TopInsight(
                label="Effective reward rate",
                value=f"{reward_rate:.2f}%",
            )
        )

    return TransactionSummaryResponse(
        spend_by_category=spend_by_category,
        rewards_by_category=spend_by_category,  # same shape, rewards included
        savings_by_card=savings_by_card,
        total_spend=total_spend,
        total_rewards=total_rewards,
        total_savings=total_savings,
        fee_adjusted_savings=fee_adjusted_savings,
        transaction_count=len(rows),
        top_insights=insights,
    )


# ---------------------------------------------------------------------------
# Story 3.4: Export
# ---------------------------------------------------------------------------

_EXPORT_COLUMNS = [
    "id",
    "merchant",
    "amount",
    "category",
    "chosen_card_id",
    "chosen_card_name",
    "reward_earned",
    "estimated_savings",
    "source_flow",
    "card_was_saved",
    "timestamp",
]


def _txn_to_export_row(txn: TransactionLog) -> List[Any]:
    return [
        txn.id,
        txn.merchant,
        txn.amount,
        txn.category,
        txn.chosen_card_id or "",
        txn.chosen_card_name or "",
        txn.reward_earned,
        txn.estimated_savings,
        txn.source_flow,
        txn.card_was_saved,
        txn.timestamp.isoformat() if txn.timestamp else "",
    ]


def export_csv(db: Session, user: User) -> str:
    """Export user's transactions as a CSV string."""
    rows = (
        db.query(TransactionLog)
        .filter(TransactionLog.user_id == user.id)
        .order_by(TransactionLog.timestamp.desc())
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_EXPORT_COLUMNS)
    for txn in rows:
        writer.writerow(_txn_to_export_row(txn))

    return output.getvalue()


def export_xlsx(db: Session, user: User) -> bytes:
    """Export user's transactions as XLSX bytes.

    Uses openpyxl. Falls back to CSV bytes if openpyxl is not installed.
    """
    try:
        from openpyxl import Workbook
    except ImportError:
        logger.warning("openpyxl not installed, falling back to CSV for XLSX export")
        return export_csv(db, user).encode("utf-8")

    rows = (
        db.query(TransactionLog)
        .filter(TransactionLog.user_id == user.id)
        .order_by(TransactionLog.timestamp.desc())
        .all()
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "Transactions"
    ws.append(_EXPORT_COLUMNS)
    for txn in rows:
        ws.append(_txn_to_export_row(txn))

    # Auto-width columns
    for col_cells in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
