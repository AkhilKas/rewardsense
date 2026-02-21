"""
Feature Output Schemas and Metadata.

Defines schemas for feature metadata and feature registry.
Used for documenting what features are available and their properties.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class FeatureMetadata(BaseModel):
    """
    Metadata about a single feature.

    Used to document feature properties for ML pipeline and monitoring.
    """

    name: str = Field(..., description="Feature name (column name)")
    data_type: Literal["numeric", "categorical", "binary", "text"] = Field(
        ..., description="Feature data type"
    )
    description: str = Field(..., description="Human-readable description")
    source: Literal["credit_card", "transaction", "user_profile", "derived"] = Field(
        ..., description="Source dataset"
    )
    nullable: bool = Field(False, description="Can this feature be null?")
    required_for_ml: bool = Field(False, description="Required for ML models?")

    # Optional metadata
    min_value: Optional[float] = Field(
        None, description="Minimum expected value (for numeric)"
    )
    max_value: Optional[float] = Field(
        None, description="Maximum expected value (for numeric)"
    )
    categories: Optional[List[str]] = Field(
        None, description="Valid categories (for categorical)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "name": "net_value_annual",
                "data_type": "numeric",
                "description": "Net annual value (rewards - fees)",
                "source": "credit_card",
                "nullable": False,
                "required_for_ml": True,
                "min_value": -1000.0,
                "max_value": 5000.0,
            }
        }


class FeatureRegistry(BaseModel):
    """
    Registry of all features in the RewardSense system.

    Serves as documentation of available features for ML pipeline.
    Updated as new features are added.
    """

    version: str = Field(..., description="Schema version (semantic versioning)")

    credit_card_features: List[FeatureMetadata] = Field(
        default_factory=list, description="Features from credit card data"
    )

    transaction_features: List[FeatureMetadata] = Field(
        default_factory=list, description="Features from transaction data"
    )

    user_profile_features: List[FeatureMetadata] = Field(
        default_factory=list, description="Features from user profile data"
    )

    def get_feature(self, name: str) -> Optional[FeatureMetadata]:
        """Get feature metadata by name."""
        all_features = (
            self.credit_card_features
            + self.transaction_features
            + self.user_profile_features
        )

        for feat in all_features:
            if feat.name == name:
                return feat

        return None

    def get_features_by_type(self, data_type: str) -> List[FeatureMetadata]:
        """Get all features of a specific type."""
        all_features = (
            self.credit_card_features
            + self.transaction_features
            + self.user_profile_features
        )

        return [f for f in all_features if f.data_type == data_type]

    def get_required_features(self) -> List[FeatureMetadata]:
        """Get all features required for ML."""
        all_features = (
            self.credit_card_features
            + self.transaction_features
            + self.user_profile_features
        )

        return [f for f in all_features if f.required_for_ml]

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "version": "1.0.0",
                "credit_card_features": [],
                "transaction_features": [],
                "user_profile_features": [],
            }
        }
