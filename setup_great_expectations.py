"""
Setup Great Expectations for RewardSense.
Simplified version compatible with GX 0.18.x
"""

import great_expectations as gx
import pandas as pd
import json
from pathlib import Path

print("=" * 70)
print("Setting Up Great Expectations for RewardSense")
print("=" * 70)

context = gx.get_context()

# =====================================================================
# Helper: Create Suite with In-Memory Validation
# =====================================================================


def create_suite_from_df(suite_name, df, expectations_func):
    """Create expectation suite from DataFrame."""
    try:
        # Create or get suite
        suite = context.add_or_update_expectation_suite(
            expectation_suite_name=suite_name
        )

        # Create validator with in-memory data
        batch = gx.core.batch.Batch(data=df)
        validator = gx.validator.validator.Validator(
            execution_engine=gx.execution_engine.PandasExecutionEngine(),
            batches=[batch],
            expectation_suite=suite,
        )

        # Add expectations
        expectations_func(validator)

        # Save suite
        context.save_expectation_suite(
            expectation_suite=validator.get_expectation_suite()
        )

        num_expectations = len(validator.get_expectation_suite().expectations)
        print(f"✅ Created {suite_name} with {num_expectations} expectations")
        return True

    except Exception as e:
        print(f"❌ Error creating {suite_name}: {e}")
        return False


# =====================================================================
# 1. Transaction Expectations
# =====================================================================

print("\n[1/3] Creating transaction expectations...")

df_txns = pd.read_csv("data/processed/current/synthetic/transactions.csv")


def add_transaction_expectations(validator):
    """Add all transaction expectations."""
    # Column presence
    validator.expect_table_columns_to_match_ordered_list(
        column_list=[
            "transaction_id",
            "user_id",
            "date",
            "category",
            "merchant",
            "mcc_code",
            "amount",
            "card_used",
        ]
    )

    # Not null
    validator.expect_column_values_to_not_be_null(column="transaction_id")
    validator.expect_column_values_to_not_be_null(column="user_id")
    validator.expect_column_values_to_not_be_null(column="amount")

    # Positive amounts
    validator.expect_column_values_to_be_between(
        column="amount", min_value=0, strictly=True
    )

    # Valid MCC codes
    validator.expect_column_values_to_be_between(
        column="mcc_code", min_value=1000, max_value=9999
    )

    # ID formats
    validator.expect_column_values_to_match_regex(
        column="transaction_id", regex=r"^txn_\d+$"
    )
    validator.expect_column_values_to_match_regex(
        column="user_id", regex=r"^user_\d{4}$"
    )

    # Valid categories
    from src.data_pipeline.generators.config import SPENDING_CATEGORIES

    valid_cats = list(SPENDING_CATEGORIES.keys()) + ["unknown"]
    validator.expect_column_values_to_be_in_set(column="category", value_set=valid_cats)


create_suite_from_df("transactions_suite", df_txns, add_transaction_expectations)

# =====================================================================
# 2. User Profile Expectations
# =====================================================================

print("\n[2/3] Creating user profile expectations...")

df_users = pd.read_csv("data/processed/current/synthetic/user_profiles.csv")


def add_user_expectations(validator):
    """Add all user profile expectations."""
    # Columns
    validator.expect_table_columns_to_match_ordered_list(
        column_list=[
            "user_id",
            "archetype",
            "monthly_budget",
            "cards",
            "redemption_preference",
            "age_group",
            "location_type",
        ]
    )

    # Not null and unique
    validator.expect_column_values_to_not_be_null(column="user_id")
    validator.expect_column_values_to_be_unique(column="user_id")

    # Positive budget
    validator.expect_column_values_to_be_between(
        column="monthly_budget", min_value=0, strictly=True
    )

    # Valid archetypes
    from src.data_pipeline.generators.config import SPENDING_ARCHETYPES

    valid_archetypes = [a.name for a in SPENDING_ARCHETYPES]
    validator.expect_column_values_to_be_in_set(
        column="archetype", value_set=valid_archetypes
    )

    # Valid age groups
    validator.expect_column_values_to_be_in_set(
        column="age_group", value_set=["18-25", "26-35", "36-50", "51-65", "65+"]
    )

    # Valid locations
    validator.expect_column_values_to_be_in_set(
        column="location_type", value_set=["urban", "suburban", "rural"]
    )

    # Valid redemptions
    from src.data_pipeline.generators.config import REDEMPTION_PREFERENCES

    validator.expect_column_values_to_be_in_set(
        column="redemption_preference", value_set=REDEMPTION_PREFERENCES
    )


create_suite_from_df("user_profiles_suite", df_users, add_user_expectations)

# =====================================================================
# 3. Credit Card Expectations
# =====================================================================

print("\n[3/3] Creating credit card expectations...")

# Check if file exists
cc_file = Path("data/processed/current/offers/creditcardbonuses_offers.json")
if not cc_file.exists():
    print(f"⚠️  Credit card data not found at {cc_file}")
    print("   Skipping credit card expectations")
else:
    with open(cc_file, "r") as f:
        cc_data = json.load(f)

    df_cards = pd.DataFrame(cc_data["offers"])

    def add_credit_card_expectations(validator):
        """Add all credit card expectations."""
        # Not null
        validator.expect_column_values_to_not_be_null(column="card_id")
        validator.expect_column_values_to_not_be_null(column="card_name")
        validator.expect_column_values_to_not_be_null(column="issuer")

        # Unique
        validator.expect_column_values_to_be_unique(column="card_id")

        # Valid annual fee range
        validator.expect_column_values_to_be_between(
            column="annual_fee", min_value=0, max_value=1000, mostly=0.95
        )

        # Has reward rates
        validator.expect_column_values_to_not_be_null(column="reward_rates")

    create_suite_from_df("credit_cards_suite", df_cards, add_credit_card_expectations)

# =====================================================================
# Summary
# =====================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

suites = context.list_expectation_suite_names()
print(f"\n✅ Total expectation suites: {len(suites)}")
for suite in suites:
    print(f"   - {suite}")

print("\n" + "=" * 70)
print("Great Expectations Setup Complete!")
print("=" * 70)
print("\nNext steps:")
print("  1. Build data docs: great_expectations docs build")
print("  2. Create checkpoints for validation")
print("  3. Integrate into pipeline")
