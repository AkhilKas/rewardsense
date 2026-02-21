"""
Transaction Data Schemas for RewardSense.

Defines schemas for transaction data at different pipeline stages:
- TransactionRaw: Initial generated/input data
- TransactionCleaned: After Story 3.1 cleaning
- TransactionFeatures: After Story 3.2 feature engineering (aggregated by user)
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from .validators import (
    validate_user_id_format,
    validate_transaction_id_format,
    validate_category,
    validate_mcc_code,
    validate_amount_positive,
)


class TransactionRaw(BaseModel):
    """
    Raw transaction data from generator or user input.

    Schema matches output from TransactionGenerator.generate():
    transaction_id, user_id, date, category, merchant, mcc_code, amount, card_used
    """

    transaction_id: str = Field(
        ..., description="Unique transaction identifier (txn_XXXXXXX)"
    )
    user_id: str = Field(..., description="User identifier (user_XXXX)")
    date: str = Field(..., description="Transaction date (YYYY-MM-DD or datetime)")
    category: str = Field(..., description="Spending category")
    merchant: str = Field(..., description="Merchant name")
    mcc_code: int = Field(..., description="Merchant Category Code (4 digits)")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    card_used: str = Field(..., description="Credit card used for transaction")

    # Validators
    _validate_user_id = field_validator("user_id")(
        lambda cls, v: validate_user_id_format(v)
    )
    _validate_transaction_id = field_validator("transaction_id")(
        lambda cls, v: validate_transaction_id_format(v)
    )
    _validate_category = field_validator("category")(
        lambda cls, v: validate_category(v)
    )
    _validate_mcc = field_validator("mcc_code")(lambda cls, v: validate_mcc_code(v))
    _validate_amount = field_validator("amount")(
        lambda cls, v: validate_amount_positive(v)
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "transaction_id": "txn_0000123",
                "user_id": "user_0001",
                "date": "2025-08-01",
                "category": "dining",
                "merchant": "Starbucks",
                "mcc_code": 5812,
                "amount": 16.09,
                "card_used": "Chase Sapphire Reserve",
            }
        }


class TransactionCleaned(TransactionRaw):
    """
    Transaction data after cleaning (Story 3.1).

    Guarantees from cleaning:
    - No negative amounts (removed)
    - No future dates (removed)
    - No invalid dates (removed)
    - Missing categories filled with 'unknown'
    - Suspicious flag added for amounts > $10,000
    """

    suspicious: bool = Field(
        False, description="Suspicious transaction flag (amount > $10,000)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "transaction_id": "txn_0000123",
                "user_id": "user_0001",
                "date": "2025-08-01",
                "category": "dining",
                "merchant": "Starbucks",
                "mcc_code": 5812,
                "amount": 16.09,
                "card_used": "Chase Sapphire Reserve",
                "suspicious": False,
            }
        }


class TransactionFeatures(BaseModel):
    """
    Aggregated transaction features per user (Story 3.2).

    Output from TransactionFeatureEngineer.engineer_features().
    One row per user with aggregated spending patterns.
    """

    user_id: str = Field(..., description="User identifier")

    # Spending by category (dynamic based on actual categories)
    # Common categories as examples:
    dining_total_spent: float = Field(0, description="Total dining spending")
    travel_total_spent: float = Field(0, description="Total travel spending")
    groceries_total_spent: float = Field(0, description="Total groceries spending")
    utilities_total_spent: float = Field(0, description="Total utilities spending")
    entertainment_total_spent: float = Field(
        0, description="Total entertainment spending"
    )

    # Transaction counts by category
    dining_txn_count: float = Field(0, description="Number of dining transactions")
    travel_txn_count: float = Field(0, description="Number of travel transactions")

    # Aggregated totals
    total_spending: float = Field(
        ..., ge=0, description="Total spending across all categories"
    )
    total_transactions: float = Field(..., ge=0, description="Total transaction count")
    spending_diversity: float = Field(0, description="Entropy of spending distribution")

    # Temporal patterns
    weekend_spending_ratio: float = Field(
        ..., ge=0, le=1, description="Proportion of weekend spending"
    )
    peak_spending_month: int = Field(
        ..., ge=1, le=12, description="Month with highest spending"
    )
    peak_spending_day: int = Field(
        ..., ge=0, le=6, description="Day of week with highest spending"
    )
    avg_transaction_amount: float = Field(..., description="Average transaction amount")
    transaction_amount_std: float = Field(
        ..., description="Std dev of transaction amounts"
    )
    median_transaction_amount: float = Field(
        ..., description="Median transaction amount"
    )
    total_spending_temporal: float = Field(
        ..., description="Total spending (from temporal calc)"
    )

    # Card usage patterns
    num_cards_used: int = Field(..., ge=1, description="Number of unique cards used")
    primary_card: Optional[str] = Field(None, description="Most frequently used card")
    card_switch_rate: float = Field(
        ..., ge=0, le=1, description="Card switching frequency"
    )

    # MCC patterns
    num_unique_mccs: int = Field(..., ge=1, description="Number of unique MCC codes")
    primary_mcc: Optional[int] = Field(None, description="Most common MCC code")
    avg_spending_per_mcc: float = Field(..., description="Average spending per MCC")

    # Merchant patterns
    num_unique_merchants: int = Field(
        ..., ge=1, description="Number of unique merchants"
    )
    favorite_merchant: Optional[str] = Field(None, description="Most frequent merchant")
    repeat_merchant_ratio: float = Field(
        ..., ge=0, le=1, description="Repeat merchant ratio"
    )

    # Suspicious (optional - from cleaning)
    num_suspicious: Optional[int] = Field(
        None, description="Number of suspicious transactions"
    )
    suspicious_rate: Optional[float] = Field(
        None, ge=0, le=1, description="Proportion of suspicious transactions"
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow dynamic category columns
