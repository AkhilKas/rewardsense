"""
Credit Card Data Schemas for RewardSense.

Defines schemas for credit card data at different pipeline stages:
- CreditCardRaw: Data from API/scrapers (references existing CardOffer)
- CreditCardCleaned: After Story 3.1 cleaning
- CreditCardFeatures: After Story 3.2 feature engineering
"""

from typing import Optional, List
from pydantic import BaseModel, Field

# Import existing CardOffer model
from src.data_pipeline.api_fetcher.schema import CardOffer


# Alias for raw credit card data
CreditCardRaw = CardOffer


class CreditCardCleaned(BaseModel):
    """
    Credit card data after cleaning (Story 3.1).

    Guarantees from cleaning:
    - Deduplicated by card_id or (card_name, issuer)
    - Issuer names standardized (uppercase, no underscores)
    - Annual fees validated (0 <= fee < 1000)
    - Missing reward_rates imputed with default
    """

    # Core identity
    card_id: str = Field(..., description="Unique card identifier")
    card_name: str = Field(..., description="Official card name")
    issuer: str = Field(..., description="Standardized issuer name (uppercase)")
    source: str = Field(..., description="Data source")

    # Financial
    annual_fee: float = Field(
        ..., ge=0, lt=1000, description="Annual fee in USD (validated 0-1000)"
    )
    is_annual_fee_waived: Optional[bool] = Field(
        None, description="First year fee waived"
    )

    # Rewards
    reward_rates: dict = Field(
        default_factory=dict, description="Contains universal_base_rate (guaranteed)"
    )
    universal_cashback_percent: Optional[float] = Field(
        None, description="Base cashback rate"
    )

    # Card metadata
    network: Optional[str] = Field(None, description="Card network")
    currency: Optional[str] = Field(None, description="Reward currency type")
    is_business: Optional[bool] = Field(None, description="Business card flag")
    discontinued: Optional[bool] = Field(False, description="Discontinued flag")

    # Welcome bonus
    offers: List[dict] = Field(default_factory=list, description="Current offers")
    historical_offers: List[dict] = Field(
        default_factory=list, description="Historical offers"
    )

    # Benefits
    credits: List[dict] = Field(
        default_factory=list, description="Annual credits/benefits"
    )

    # Metadata
    last_updated: str = Field(..., description="Last update timestamp")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "card_id": "abc123",
                "card_name": "Chase Sapphire Reserve",
                "issuer": "CHASE",
                "source": "creditcardbonuses",
                "annual_fee": 550.0,
                "is_annual_fee_waived": False,
                "reward_rates": {"universal_base_rate": 1.0},
                "network": "VISA",
                "currency": "POINTS",
                "discontinued": False,
                "offers": [],
                "credits": [],
                "last_updated": "2026-02-17T12:00:00",
            }
        }


class CreditCardFeatures(BaseModel):
    """
    Credit card data after feature engineering (Story 3.2).

    Includes all cleaned fields plus engineered features:
    - Extracted reward rates
    - Welcome bonus calculations
    - Effective fee calculations
    - Net value metrics
    - One-hot encoded issuer/network
    """

    # Original fields (from cleaned)
    card_id: str
    card_name: str
    issuer: str
    annual_fee: float

    # Engineered: Reward rates
    base_reward_rate: float = Field(
        ..., description="Extracted from reward_rates.universal_base_rate"
    )
    cashback_rate: float = Field(..., description="Effective cashback rate")

    # Engineered: Welcome bonus
    welcome_bonus_spend_req: float = Field(0, description="Spend requirement for bonus")
    welcome_bonus_amount: float = Field(
        0, description="Bonus amount (points/miles/dollars)"
    )
    welcome_bonus_days: int = Field(90, description="Days to meet requirement")
    welcome_bonus_value_usd: float = Field(0, description="Bonus value in USD")
    welcome_bonus_roi: float = Field(0, description="Return on spend requirement")
    bonus_difficulty: str = Field(
        "none", description="Bonus difficulty: easy/medium/hard/none"
    )

    # Engineered: Credits/benefits
    annual_credits_value: float = Field(0, description="Total annual credits value")
    num_credits: int = Field(0, description="Number of credit benefits")
    has_credits: int = Field(0, description="Has credits flag (0/1)")

    # Engineered: Effective fees
    effective_annual_fee: float = Field(..., description="Annual fee minus credits")
    effective_fee_year1: float = Field(
        ..., description="Year 1 fee (considering waiver)"
    )
    net_annual_cost: float = Field(..., description="Net annual cost")

    # Engineered: Net value
    expected_annual_rewards: float = Field(..., description="Expected rewards per year")
    net_value_annual: float = Field(..., description="Net value: rewards - fees")
    net_value_year1: float = Field(..., description="First year net value (with bonus)")
    value_per_dollar: float = Field(..., description="Value per dollar spent")

    # Engineered: Status flags
    is_active: int = Field(1, description="Active card flag (0/1)")
    is_discontinued: int = Field(0, description="Discontinued flag (0/1)")
    is_premium: int = Field(0, description="Premium tier flag (fee >= $450)")
    is_mid_tier: int = Field(0, description="Mid tier flag ($95 <= fee < $450)")
    is_no_annual_fee: int = Field(0, description="No annual fee flag")
    is_business: int = Field(0, description="Business card flag")

    # Note: One-hot encoded fields (issuer_*, network_*, currency_*) are dynamic
    # and not enforced in schema to allow flexibility

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow dynamic one-hot encoded columns
