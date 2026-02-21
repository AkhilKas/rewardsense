"""
Shared validation utilities for RewardSense schemas.

These validators are used across multiple schema definitions to ensure
consistent validation logic throughout the pipeline.
"""

from typing import Any


def validate_user_id_format(v: Any) -> str:
    """
    Validate user_id follows format: user_XXXX where X is a digit.

    Args:
        v: User ID value

    Returns:
        Validated user_id string

    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(v, str):
        raise ValueError(f"user_id must be string, got {type(v)}")

    if not v.startswith("user_"):
        raise ValueError(f'user_id must start with "user_", got: {v}')

    # Check suffix is numeric
    suffix = v.replace("user_", "")
    if not suffix.isdigit():
        raise ValueError(f"user_id suffix must be numeric, got: {v}")

    return v


def validate_transaction_id_format(v: Any) -> str:
    """
    Validate transaction_id follows format: txn_XXXXXXX.

    Args:
        v: Transaction ID value

    Returns:
        Validated transaction_id string

    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(v, str):
        raise ValueError(f"transaction_id must be string, got {type(v)}")

    if not v.startswith("txn_"):
        raise ValueError(f'transaction_id must start with "txn_", got: {v}')

    return v


def validate_category(v: Any) -> str:
    """
    Validate category against known spending categories.

    Args:
        v: Category value

    Returns:
        Validated category string

    Raises:
        ValueError: If category is unknown
    """
    from src.data_pipeline.generators.config import SPENDING_CATEGORIES

    valid_categories = list(SPENDING_CATEGORIES.keys()) + ["unknown", "other"]

    if v not in valid_categories:
        raise ValueError(
            f'Unknown category: {v}. Valid categories: {", ".join(valid_categories)}'
        )

    return v


def validate_mcc_code(v: Any) -> int:
    """
    Validate MCC code is a valid 4-digit integer.

    Args:
        v: MCC code value

    Returns:
        Validated MCC code

    Raises:
        ValueError: If MCC code is invalid
    """
    if not isinstance(v, int):
        try:
            v = int(v)
        except (ValueError, TypeError):
            raise ValueError(f"MCC code must be integer, got {type(v)}")

    if v < 1000 or v > 9999:
        raise ValueError(f"MCC code must be 4 digits (1000-9999), got: {v}")

    return v


def validate_amount_positive(v: Any) -> float:
    """
    Validate amount is positive.

    Args:
        v: Amount value

    Returns:
        Validated amount as float

    Raises:
        ValueError: If amount is not positive
    """
    if not isinstance(v, (int, float)):
        try:
            v = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Amount must be numeric, got {type(v)}")

    if v <= 0:
        raise ValueError(f"Amount must be positive, got: {v}")

    return float(v)


def validate_redemption_preference(v: Any) -> str:
    """
    Validate redemption preference against known options.

    Args:
        v: Redemption preference value

    Returns:
        Validated redemption preference

    Raises:
        ValueError: If preference is unknown
    """
    from src.data_pipeline.generators.config import REDEMPTION_PREFERENCES

    if v not in REDEMPTION_PREFERENCES:
        raise ValueError(
            f"Unknown redemption preference: {v}. "
            f'Valid options: {", ".join(REDEMPTION_PREFERENCES)}'
        )

    return v


def validate_archetype(v: Any) -> str:
    """
    Validate user archetype against known types.

    Args:
        v: Archetype value

    Returns:
        Validated archetype

    Raises:
        ValueError: If archetype is unknown
    """
    from src.data_pipeline.generators.config import SPENDING_ARCHETYPES

    valid_archetypes = [a.name for a in SPENDING_ARCHETYPES]

    if v not in valid_archetypes:
        raise ValueError(
            f'Unknown archetype: {v}. Valid archetypes: {", ".join(valid_archetypes)}'
        )

    return v
