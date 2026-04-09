"""Pydantic request/response shapes for user profile and card wallet endpoints."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Valid persona keys — must match config/personas.yaml (added in Story 1.3)
# ---------------------------------------------------------------------------
VALID_PERSONAS = {"student", "traveler", "family", "cashback-focused"}

# Valid reward preferences
VALID_REWARD_PREFERENCES = {"cashback", "points", "miles"}


class UserProfileResponse(BaseModel):
    user_id: int
    email: str
    display_name: str
    personas: List[str]
    reward_preference: str
    transaction_logging_enabled: bool
    dark_mode: bool
    saved_card_ids: List[str]


class ProfilePatchRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    personas: Optional[List[str]] = None
    reward_preference: Optional[str] = None
    transaction_logging_enabled: Optional[bool] = None
    dark_mode: Optional[bool] = None


class CardListRequest(BaseModel):
    card_ids: List[str]


class CardDisplayInfo(BaseModel):
    """Normalized UI-ready card presentation shape.

    Used consistently across catalog, recommendations, wallet, and calculator
    so the frontend never has to assemble display fields from multiple places.
    """

    card_id: str
    card_name: str
    issuer: str
    annual_fee: float
    reward_highlights: List[str]
    image_url: Optional[str] = None


class CardCatalogItem(BaseModel):
    card_id: str
    card_name: str
    issuer: str
    annual_fee: float
    reward_highlights: List[str]
    image_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Recommendation request / response shapes (Story 1.3 + Story 2.1)
# ---------------------------------------------------------------------------


class ScoreBreakdown(BaseModel):
    """Per-card breakdown showing how the final reward was computed."""

    raw_reward_rate: float
    raw_reward_amount: float
    personalization_multiplier: float
    persona_category_boost: float
    persona_fee_penalty: float


class ScoredCard(BaseModel):
    card_id: Optional[str] = None
    card_name: str
    reward_amount: float
    annual_fee: float = 0.0
    rank: int
    persona_adjustments: Optional[Dict[str, float]] = None
    score_breakdown: Optional[ScoreBreakdown] = None
    persona_match_reason: Optional[str] = None
    projected_savings: Optional[float] = None
    card_display: Optional[CardDisplayInfo] = None


class PersonaRecommendResponse(BaseModel):
    top_card: Optional[ScoredCard] = None
    alternatives: List[ScoredCard] = []
    ranked: List[ScoredCard]
    best_card_id: Optional[str]
    is_personalized: bool = Field(
        description="True when the scorer applied ML personalization weights"
    )
    is_generic: bool = Field(
        description="True when the user has no saved cards and the curated catalog was used as fallback"
    )
    active_personas: List[str]
    persona_context: str


class PortfolioRecommendRequest(BaseModel):
    spending_categories: Optional[Dict[str, float]] = None
    monthly_spend: float = 0.0


class TransactionRecommendRequest(BaseModel):
    merchant: str
    amount: float
    category: Optional[str] = None


# ---------------------------------------------------------------------------
# Quick single-transaction recommendation (Story 2.2)
# ---------------------------------------------------------------------------


class QuickTransactionRequest(BaseModel):
    """``McDonald's, $15`` style quick entry."""

    merchant: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    category: Optional[str] = None
    date: Optional[str] = None  # ISO-8601 date string, e.g. "2026-04-09"


class QuickTransactionResponse(BaseModel):
    """Focused response for a single purchase."""

    top_card: Optional[ScoredCard] = None
    alternatives: List[ScoredCard] = []
    estimated_reward: float
    money_saved: float
    category_used: str
    is_personalized: bool
    has_saved_cards: bool
    active_personas: List[str]
    persona_context: str


# ---------------------------------------------------------------------------
# Savings calculator (Story 2.3)
# ---------------------------------------------------------------------------


class SavingsCalculatorRequest(BaseModel):
    """Spending profile for the savings calculator."""

    spending_by_category: Optional[Dict[str, float]] = None
    monthly_spend: float = 0.0


class CategorySavings(BaseModel):
    """Reward breakdown for a single category."""

    category: str
    monthly_spend: float
    reward_amount: float
    baseline_reward: float
    uplift: float


class CardSavingsDetail(BaseModel):
    """Full savings view for one card across all spending categories."""

    card_id: str
    card_name: str
    annual_fee: float
    card_display: Optional[CardDisplayInfo] = None
    categories: List[CategorySavings]
    monthly_reward_total: float
    annual_reward_total: float
    monthly_uplift_vs_baseline: float
    annual_uplift_vs_baseline: float
    net_annual_benefit: float = Field(
        description="annual_uplift_vs_baseline minus annual fee difference relative to baseline"
    )


class SavingsCalculatorResponse(BaseModel):
    """Category-by-category savings breakdown for every card in the wallet."""

    cards: List[CardSavingsDetail]
    baseline_card_id: Optional[str] = Field(
        description="Saved catch-all card used as baseline, or null for generic 1%"
    )
    baseline_card_name: str
    baseline_annual_fee: float
    spending_profile: Dict[str, float]
    total_monthly_spend: float
