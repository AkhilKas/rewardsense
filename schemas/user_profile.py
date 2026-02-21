"""
User Profile Data Schemas for RewardSense.

Defines schemas for user profile data at different pipeline stages:
- UserProfileRaw: Initial generated/input data
- UserCardMapping: User-to-card ownership mapping
- UserProfileFeatures: After Story 3.2 feature engineering
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from .validators import (
    validate_user_id_format,
    validate_archetype,
    validate_redemption_preference,
)


class UserProfileRaw(BaseModel):
    """
    Raw user profile data from generator.

    Schema matches output from UserProfileGenerator.generate():
    user_id, archetype, monthly_budget, cards, redemption_preference, age_group, location_type
    """

    user_id: str = Field(..., description="User identifier (user_XXXX)")
    archetype: str = Field(..., description="User spending archetype")
    monthly_budget: float = Field(..., gt=0, description="Monthly spending budget")
    cards: str = Field(..., description="String representation of card list")
    redemption_preference: str = Field(..., description="Preferred redemption method")
    age_group: str = Field(
        ..., description="Age group (18-25, 26-35, 36-50, 51-65, 65+)"
    )
    location_type: str = Field(
        ..., description="Location type (urban, suburban, rural)"
    )

    # Validators
    _validate_user_id = field_validator("user_id")(
        lambda cls, v: validate_user_id_format(v)
    )
    _validate_archetype = field_validator("archetype")(
        lambda cls, v: validate_archetype(v)
    )
    _validate_redemption = field_validator("redemption_preference")(
        lambda cls, v: validate_redemption_preference(v)
    )

    @field_validator("age_group")
    @classmethod
    def validate_age_group(cls, v: str) -> str:
        """Validate age group."""
        valid = ["18-25", "26-35", "36-50", "51-65", "65+"]
        if v not in valid:
            raise ValueError(f"Invalid age_group: {v}. Valid: {valid}")
        return v

    @field_validator("location_type")
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate location type."""
        valid = ["urban", "suburban", "rural"]
        if v not in valid:
            raise ValueError(f"Invalid location_type: {v}. Valid: {valid}")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "user_id": "user_0001",
                "archetype": "high_roller",
                "monthly_budget": 13156.94,
                "cards": "['Chase Sapphire Reserve', 'Amex Platinum', 'Capital One Venture X']",
                "redemption_preference": "travel_portal",
                "age_group": "51-65",
                "location_type": "urban",
            }
        }


class UserCardMapping(BaseModel):
    """
    User-to-card ownership mapping.

    Schema matches output from UserProfileGenerator.generate_user_cards_mapping():
    user_id, card_id, redemption_preference
    """

    user_id: str = Field(..., description="User identifier")
    card_id: str = Field(..., description="Card name/identifier")
    redemption_preference: str = Field(
        ..., description="Redemption preference for this card"
    )

    # Validators
    _validate_user_id = field_validator("user_id")(
        lambda cls, v: validate_user_id_format(v)
    )
    _validate_redemption = field_validator("redemption_preference")(
        lambda cls, v: validate_redemption_preference(v)
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "user_id": "user_0001",
                "card_id": "Chase Sapphire Reserve",
                "redemption_preference": "travel_portal",
            }
        }


class UserProfileFeatures(BaseModel):
    """
    User profile data after feature engineering (Story 3.2).

    Includes all raw fields plus engineered features:
    - Parsed cards list
    - Encoded archetypes, age groups, locations
    - Budget features
    - Point valuations
    """

    # Original fields
    user_id: str
    archetype: str
    monthly_budget: float
    redemption_preference: str
    age_group: str
    location_type: str

    # Parsed from 'cards' string
    cards_list: List[str] = Field(..., description="Parsed list of cards")
    num_cards: int = Field(..., ge=1, description="Number of cards owned")

    # Budget features
    monthly_budget_log: float = Field(..., description="Log transform of budget")
    annual_budget: float = Field(..., description="Annual budget (monthly * 12)")
    budget_quartile: Optional[str] = Field(None, description="Budget quartile category")

    # Age features
    age_group_ordinal: int = Field(
        ..., ge=0, le=5, description="Ordinal age encoding (1-5)"
    )

    # Point valuation
    estimated_point_value: float = Field(
        ..., ge=0, le=1, description="Estimated point value in dollars"
    )

    # Note: One-hot encoded fields (archetype_*, age_*, location_*, redemption_*, budget_*)
    # are dynamic and not enforced to allow flexibility

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow dynamic one-hot encoded columns
