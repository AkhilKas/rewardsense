"""
Data validation integration for RewardSense pipeline.

Provides functions to validate data at each pipeline stage using Great Expectations.
"""

import great_expectations as gx
import pandas as pd
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data using Great Expectations suites."""

    def __init__(self):
        """Initialize validator with GX context."""
        self.context = gx.get_context()
        logger.info("Initialized DataValidator")

    def validate_transactions(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate transaction data against expectations.

        Args:
            df: Transaction DataFrame

        Returns:
            Tuple of (validation_passed, validation_results)
        """
        logger.info(f"Validating {len(df)} transactions...")

        try:
            suite = self.context.get_expectation_suite("transactions_suite")

            batch = gx.core.batch.Batch(data=df)
            validator = gx.validator.validator.Validator(
                execution_engine=gx.execution_engine.PandasExecutionEngine(),
                batches=[batch],
                expectation_suite=suite,
            )

            results = validator.validate()

            success = results.success
            stats = results.statistics

            logger.info(f"Transaction validation: {'PASSED' if success else 'FAILED'}")
            logger.info(f"  Evaluated: {stats['evaluated_expectations']}")
            logger.info(f"  Successful: {stats['successful_expectations']}")
            logger.info(f"  Failed: {stats['unsuccessful_expectations']}")

            return success, results.to_json_dict()

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, {"error": str(e)}

    def validate_user_profiles(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate user profile data against expectations.

        Args:
            df: User profile DataFrame

        Returns:
            Tuple of (validation_passed, validation_results)
        """
        logger.info(f"Validating {len(df)} user profiles...")

        try:
            suite = self.context.get_expectation_suite("user_profiles_suite")

            batch = gx.core.batch.Batch(data=df)
            validator = gx.validator.validator.Validator(
                execution_engine=gx.execution_engine.PandasExecutionEngine(),
                batches=[batch],
                expectation_suite=suite,
            )

            results = validator.validate()

            success = results.success
            stats = results.statistics

            logger.info(f"User profile validation: {'PASSED' if success else 'FAILED'}")
            logger.info(f"  Evaluated: {stats['evaluated_expectations']}")
            logger.info(f"  Successful: {stats['successful_expectations']}")

            return success, results.to_json_dict()

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, {"error": str(e)}

    def validate_credit_cards(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate credit card data against expectations.

        Args:
            df: Credit card DataFrame

        Returns:
            Tuple of (validation_passed, validation_results)
        """
        logger.info(f"Validating {len(df)} credit cards...")

        try:
            suite = self.context.get_expectation_suite("credit_cards_suite")

            batch = gx.core.batch.Batch(data=df)
            validator = gx.validator.validator.Validator(
                execution_engine=gx.execution_engine.PandasExecutionEngine(),
                batches=[batch],
                expectation_suite=suite,
            )

            results = validator.validate()

            success = results.success
            stats = results.statistics

            logger.info(f"Credit card validation: {'PASSED' if success else 'FAILED'}")
            logger.info(f"  Evaluated: {stats['evaluated_expectations']}")

            return success, results.to_json_dict()

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, {"error": str(e)}


# Convenience function
def validate_all_data(
    transactions_df: pd.DataFrame = None,
    users_df: pd.DataFrame = None,
    cards_df: pd.DataFrame = None,
) -> Dict[str, bool]:
    """
    Validate all datasets.

    Args:
        transactions_df: Transaction DataFrame
        users_df: User profile DataFrame
        cards_df: Credit card DataFrame

    Returns:
        Dictionary of validation results per dataset
    """
    validator = DataValidator()
    results = {}

    if transactions_df is not None:
        success, _ = validator.validate_transactions(transactions_df)
        results["transactions"] = success

    if users_df is not None:
        success, _ = validator.validate_user_profiles(users_df)
        results["users"] = success

    if cards_df is not None:
        success, _ = validator.validate_credit_cards(cards_df)
        results["cards"] = success

    return results
