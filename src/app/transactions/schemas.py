"""
Pydantic request/response shapes for transaction ledger, summary, and export.

- Opt-in transaction logging.
- Logged transaction schema.
- Summary page.
- Export.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Story 3.1 + 3.2: Transaction logging
# ---------------------------------------------------------------------------

VALID_SOURCE_FLOWS = {"manual", "portfolio", "transaction"}


class TransactionCreateRequest(BaseModel):
    """Create a new transaction log entry."""

    merchant: str = Field(..., min_length=1, max_length=200)
    amount: float = Field(..., gt=0)
    category: str = Field(..., min_length=1, max_length=50)
    chosen_card_id: Optional[str] = None
    chosen_card_name: Optional[str] = None
    reward_earned: float = Field(default=0.0, ge=0)
    estimated_savings: float = Field(default=0.0)
    source_flow: str = Field(default="manual")
    recommendation_event_id: Optional[int] = None
    timestamp: Optional[str] = None  # ISO-8601; defaults to now if omitted


class TransactionResponse(BaseModel):
    """Single transaction log entry returned to the client."""

    id: int
    merchant: str
    amount: float
    category: str
    chosen_card_id: Optional[str]
    chosen_card_name: Optional[str]
    reward_earned: float
    estimated_savings: float
    source_flow: str
    card_was_saved: bool
    recommendation_event_id: Optional[int]
    timestamp: str


class TransactionListResponse(BaseModel):
    """Paginated transaction history."""

    transactions: List[TransactionResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    total_rewards: float = Field(
        description="Sum of reward_earned across all logged transactions for this user.",
    )
    total_savings: float = Field(
        description="Sum of estimated_savings across all logged transactions for this user.",
    )


# ---------------------------------------------------------------------------
# Story 3.3: Summary
# ---------------------------------------------------------------------------


class CategorySummary(BaseModel):
    """Aggregated spend and rewards for a single category."""

    category: str
    total_spend: float
    total_reward: float
    total_savings: float
    transaction_count: int


class CardSummary(BaseModel):
    """Aggregated savings by card."""

    card_id: Optional[str]
    card_name: Optional[str]
    total_spend: float
    total_reward: float
    total_savings: float
    transaction_count: int


class TopInsight(BaseModel):
    """Short insight for the summary page."""

    label: str
    value: str


class TransactionSummaryResponse(BaseModel):
    """Chart-ready aggregates for the summary page (Story 3.3)."""

    spend_by_category: List[CategorySummary]
    rewards_by_category: List[CategorySummary]
    savings_by_card: List[CardSummary]
    total_spend: float
    total_rewards: float
    total_savings: float
    fee_adjusted_savings: float
    transaction_count: int
    top_insights: List[TopInsight]
