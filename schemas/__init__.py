"""
RewardSense Data Schemas

Pydantic v2 schemas defining data contracts for all pipeline stages.

Schema Versions:
- Raw: Data from API/scrapers/generators (input)
- Cleaned: After Story 3.1 cleaning
- Features: After Story 3.2 feature engineering (output)

Usage:
    from schemas import TransactionRaw, TransactionCleaned, TransactionFeatures
    
    # Validate raw data
    txn = TransactionRaw(**data)
    
    # Validate cleaned data
    cleaned = TransactionCleaned(**cleaned_data)
    
    # Validate features
    features = TransactionFeatures(**feature_data)
"""

__version__ = "1.0.0"

# Credit card schemas
from .credit_card import CreditCardRaw, CreditCardCleaned, CreditCardFeatures

# Transaction schemas
from .transaction import TransactionRaw, TransactionCleaned, TransactionFeatures

# User profile schemas
from .user_profile import UserProfileRaw, UserCardMapping, UserProfileFeatures

# Feature metadata
from .features import FeatureMetadata, FeatureRegistry

# Validators
from .validators import (
    validate_user_id_format,
    validate_transaction_id_format,
    validate_category,
    validate_mcc_code,
    validate_amount_positive,
    validate_redemption_preference,
    validate_archetype,
)

__all__ = [
    # Version
    "__version__",
    # Credit card schemas
    "CreditCardRaw",
    "CreditCardCleaned",
    "CreditCardFeatures",
    # Transaction schemas
    "TransactionRaw",
    "TransactionCleaned",
    "TransactionFeatures",
    # User profile schemas
    "UserProfileRaw",
    "UserCardMapping",
    "UserProfileFeatures",
    # Feature metadata
    "FeatureMetadata",
    "FeatureRegistry",
    # Validators
    "validate_user_id_format",
    "validate_transaction_id_format",
    "validate_category",
    "validate_mcc_code",
    "validate_amount_positive",
    "validate_redemption_preference",
    "validate_archetype",
]
