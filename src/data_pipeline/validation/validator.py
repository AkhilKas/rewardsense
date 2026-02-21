"""
Data validation integration for RewardSense pipeline.

Provides functions to validate data at each pipeline stage using Great Expectations.
"""

import logging
from typing import Any, Dict, Tuple

import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data using Great Expectations suites."""

    def __init__(self):
        """Initialize validator with an ephemeral GX context.

        Uses ``mode="ephemeral"`` so the validator works regardless of
        whether a ``great_expectations.yml`` is present on disk, and
        avoids v0.x-vs-v1.x config-migration issues.
        """
        self.context = gx.get_context(mode="ephemeral")
        logger.info("Initialized DataValidator (ephemeral context)")

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
            suite_name = "transactions_suite"
            suite = self.context.suites.add(gx.ExpectationSuite(name=suite_name))

            suite.add_expectation(
                gx.expectations.ExpectColumnToExist(column="transaction_id")
            )
            suite.add_expectation(gx.expectations.ExpectColumnToExist(column="user_id"))
            suite.add_expectation(gx.expectations.ExpectColumnToExist(column="amount"))
            suite.add_expectation(
                gx.expectations.ExpectColumnToExist(column="category")
            )
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column="transaction_id")
            )
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
            )

            data_source = self.context.data_sources.add_pandas(name="txn_source")
            data_asset = data_source.add_dataframe_asset(name="txn_asset")
            batch_definition = data_asset.add_batch_definition_whole_dataframe(
                "txn_batch"
            )
            batch_definition.get_batch(batch_parameters={"dataframe": df})

            validation_definition = self.context.validation_definitions.add(
                gx.ValidationDefinition(
                    name="txn_validation",
                    data=batch_definition,
                    suite=suite,
                )
            )

            results = validation_definition.run(batch_parameters={"dataframe": df})
            success = results.success

            stats = {
                "evaluated_expectations": len(results.results),
                "successful_expectations": sum(1 for r in results.results if r.success),
                "unsuccessful_expectations": sum(
                    1 for r in results.results if not r.success
                ),
            }

            logger.info(f"Transaction validation: {'PASSED' if success else 'FAILED'}")
            logger.info(f"  Evaluated: {stats['evaluated_expectations']}")
            logger.info(f"  Successful: {stats['successful_expectations']}")
            logger.info(f"  Failed: {stats['unsuccessful_expectations']}")

            return success, {"statistics": stats, "success": success}

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
            suite_name = "user_profiles_suite"
            suite = self.context.suites.add(gx.ExpectationSuite(name=suite_name))

            suite.add_expectation(gx.expectations.ExpectColumnToExist(column="user_id"))
            suite.add_expectation(
                gx.expectations.ExpectColumnToExist(column="archetype")
            )
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
            )

            data_source = self.context.data_sources.add_pandas(name="user_source")
            data_asset = data_source.add_dataframe_asset(name="user_asset")
            batch_definition = data_asset.add_batch_definition_whole_dataframe(
                "user_batch"
            )
            batch_definition.get_batch(batch_parameters={"dataframe": df})

            validation_definition = self.context.validation_definitions.add(
                gx.ValidationDefinition(
                    name="user_validation",
                    data=batch_definition,
                    suite=suite,
                )
            )

            results = validation_definition.run(batch_parameters={"dataframe": df})
            success = results.success

            stats = {
                "evaluated_expectations": len(results.results),
                "successful_expectations": sum(1 for r in results.results if r.success),
                "unsuccessful_expectations": sum(
                    1 for r in results.results if not r.success
                ),
            }

            logger.info(f"User profile validation: {'PASSED' if success else 'FAILED'}")
            logger.info(f"  Evaluated: {stats['evaluated_expectations']}")
            logger.info(f"  Successful: {stats['successful_expectations']}")

            return success, {"statistics": stats, "success": success}

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
            suite_name = "credit_cards_suite"
            suite = self.context.suites.add(gx.ExpectationSuite(name=suite_name))

            suite.add_expectation(
                gx.expectations.ExpectColumnToExist(column="card_name")
            )
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column="card_name")
            )

            data_source = self.context.data_sources.add_pandas(name="card_source")
            data_asset = data_source.add_dataframe_asset(name="card_asset")
            batch_definition = data_asset.add_batch_definition_whole_dataframe(
                "card_batch"
            )
            batch_definition.get_batch(batch_parameters={"dataframe": df})

            validation_definition = self.context.validation_definitions.add(
                gx.ValidationDefinition(
                    name="card_validation",
                    data=batch_definition,
                    suite=suite,
                )
            )

            results = validation_definition.run(batch_parameters={"dataframe": df})
            success = results.success

            stats = {
                "evaluated_expectations": len(results.results),
                "successful_expectations": sum(1 for r in results.results if r.success),
                "unsuccessful_expectations": sum(
                    1 for r in results.results if not r.success
                ),
            }

            logger.info(f"Credit card validation: {'PASSED' if success else 'FAILED'}")
            logger.info(f"  Evaluated: {stats['evaluated_expectations']}")

            return success, {"statistics": stats, "success": success}

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
