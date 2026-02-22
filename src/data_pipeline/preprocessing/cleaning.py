"""
RewardSense - Data Cleaning Module (Enhanced)

Story 3.1: Comprehensive data cleaning for credit card and transaction data.

This module provides:
- Credit card data cleaning (deduplication, name normalization, validation)
- Transaction data cleaning (invalid removal, MCC validation, suspicious flagging)
- Cleaning reports with before/after metrics

All functions are idempotent and log cleaning steps.
"""

import re
import logging
from typing import Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (Story 3.3 will move these to YAML)
# =============================================================================


@dataclass
class CleaningConfig:
    """Configuration for cleaning thresholds and rules."""

    # Credit card thresholds
    max_annual_fee: float = 1000.0
    min_annual_fee: float = 0.0

    # Transaction thresholds
    min_transaction_amount: float = 0.0
    suspicious_amount_threshold: float = 10000.0

    # Valid MCC codes (from generators/config.py)
    valid_mcc_codes: Set[int] = field(
        default_factory=lambda: {
            # Groceries
            5411,
            5422,
            5441,
            5451,
            5462,
            # Dining
            5812,
            5813,
            5814,
            # Gas
            5541,
            5542,
            # Travel
            3000,
            3001,
            4511,
            4722,
            7011,
            7512,
            # Online shopping
            5691,
            5699,
            5944,
            5945,
            5947,
            # Streaming
            4899,
            5815,
            5816,
            5818,
            # Utilities
            4900,
            4814,
            4816,
            # Insurance
            6300,
            6381,
            5960,
            # Entertainment
            7832,
            7922,
            7941,
            7991,
            7999,
            # Drugstore
            5912,
            # Home improvement
            5200,
            5211,
            5231,
            5251,
            5261,
            # Transit
            4111,
            4112,
            4121,
            4131,
            # Other
            5999,
        }
    )

    # Issuer name standardization mapping
    issuer_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            "AMEX": "AMERICAN EXPRESS",
            "AMERICANEXPRESS": "AMERICAN EXPRESS",
            "BOFA": "BANK OF AMERICA",
            "BOA": "BANK OF AMERICA",
            "CAPITALONE": "CAPITAL ONE",
            "CAP ONE": "CAPITAL ONE",
            "USBANK": "US BANK",
            "U.S. BANK": "US BANK",
            "WELLSFARGO": "WELLS FARGO",
        }
    )


# Default configuration instance
DEFAULT_CONFIG = CleaningConfig()


# =============================================================================
# Text Normalization Utilities
# =============================================================================


def clean_card_name(name: Any) -> str:
    """
    Normalize credit card names for consistent deduplication.

    Handles:
    - Trademark symbols (®, ™, ℠, ©)
    - Extra whitespace
    - Inconsistent casing

    Args:
        name: Raw card name (string or any type)

    Returns:
        Cleaned, normalized card name

    Examples:
        >>> clean_card_name("Chase Sapphire Preferred®")
        "Chase Sapphire Preferred"
        >>> clean_card_name("AMEX  Gold   Card™")
        "Amex Gold Card"
    """
    if pd.isna(name) or name is None:
        return ""

    name = str(name)

    # Remove trademark and copyright symbols
    name = re.sub(r"[®™℠©]", "", name)

    # Remove "Credit Card" suffix (redundant)
    name = re.sub(r"\s*credit\s*card\s*$", "", name, flags=re.IGNORECASE)

    # Normalize whitespace (multiple spaces, tabs, newlines -> single space)
    name = re.sub(r"\s+", " ", name).strip()

    # Title case for consistency (but preserve known acronyms)
    name = name.title()

    # Fix common acronyms that get mangled by title()
    acronym_fixes = {
        "Amex": "AMEX",
        "Aaa": "AAA",
        "Ihg": "IHG",
        "Usb": "USB",
    }
    for wrong, correct in acronym_fixes.items():
        name = name.replace(wrong, correct)

    return name


def standardize_issuer_name(
    issuer: Any, aliases: Optional[Dict[str, str]] = None
) -> str:
    """
    Standardize issuer names for consistent grouping.

    Args:
        issuer: Raw issuer name
        aliases: Optional mapping of aliases to canonical names

    Returns:
        Standardized issuer name (uppercase, no underscores)

    Examples:
        >>> standardize_issuer_name("chase_bank")
        "CHASE BANK"
        >>> standardize_issuer_name("Amex", {"AMEX": "AMERICAN EXPRESS"})
        "AMERICAN EXPRESS"
    """
    if pd.isna(issuer) or issuer is None:
        return "UNKNOWN"

    # Convert to string, uppercase, replace underscores
    issuer = str(issuer).upper().replace("_", " ").strip()

    # Remove extra whitespace
    issuer = re.sub(r"\s+", " ", issuer)

    # Apply aliases if provided
    if aliases:
        # Check exact match first
        if issuer in aliases:
            return aliases[issuer]
        # Check without spaces
        issuer_no_space = issuer.replace(" ", "")
        if issuer_no_space in aliases:
            return aliases[issuer_no_space]

    return issuer


def normalize_welcome_bonus(bonus: Any) -> Dict[str, Any]:
    """
    Parse and normalize welcome bonus text to structured format.

    Handles various formats:
    - "60,000 points after $4,000 spend in 3 months"
    - "$750 bonus"
    - "60000 miles"

    Args:
        bonus: Raw welcome bonus text

    Returns:
        Dict with keys: amount, unit, spend_requirement, time_days, raw_text
    """
    result = {
        "amount": None,
        "unit": "points",  # default
        "spend_requirement": None,
        "time_days": None,
        "raw_text": str(bonus) if bonus else None,
    }

    if pd.isna(bonus) or bonus is None:
        return result

    text = str(bonus).lower()

    # FIRST: Check for cash bonus format ($XXX) at start of string
    cash_bonus_match = re.search(r"^\s*\$\s*([\d,]+)", text)
    if cash_bonus_match:
        result["amount"] = int(cash_bonus_match.group(1).replace(",", ""))
        result["unit"] = "dollars"
    else:
        # Extract bonus amount (handles commas)
        amount_match = re.search(r"([\d,]+)\s*(points?|miles?|dollars?)", text)
        if amount_match:
            amount_str = amount_match.group(1).replace(",", "")
            result["amount"] = int(amount_str)

            unit = amount_match.group(2) or "points"
            if "mile" in unit:
                result["unit"] = "miles"
            elif "dollar" in unit:
                result["unit"] = "dollars"
            else:
                result["unit"] = "points"

    # Extract cash bonus format ($XXX)
    cash_match = re.search(r"\$\s*([\d,]+)", text)
    if cash_match and result["amount"] is None:
        result["amount"] = int(cash_match.group(1).replace(",", ""))
        result["unit"] = "dollars"

    # Extract spend requirement
    spend_match = re.search(r"(?:after|spend|spending)\s*\$?\s*([\d,]+)", text)
    if spend_match:
        result["spend_requirement"] = int(spend_match.group(1).replace(",", ""))

    # Extract time limit
    time_match = re.search(r"(\d+)\s*(month|day|week)", text)
    if time_match:
        value = int(time_match.group(1))
        unit = time_match.group(2)
        if "month" in unit:
            result["time_days"] = value * 30
        elif "week" in unit:
            result["time_days"] = value * 7
        else:
            result["time_days"] = value

    return result


# =============================================================================
# Credit Card Data Cleaning
# =============================================================================


def clean_credit_card_data(
    df: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean credit card data with comprehensive normalization and validation.

    Cleaning steps:
    1. Normalize card names (remove ®™, standardize whitespace)
    2. Standardize issuer names (uppercase, apply aliases)
    3. Deduplicate by card_name + issuer (or card_id if available)
    4. Handle missing reward_rates (impute default)
    5. Validate annual fee ranges
    6. Parse and normalize welcome bonus text
    7. Remove discontinued cards (optional flag)

    Args:
        df: Raw credit card DataFrame
        config: Cleaning configuration (uses defaults if None)

    Returns:
        Tuple of (cleaned DataFrame, cleaning report dict)

    Example:
        >>> df_clean, report = clean_credit_card_data(raw_cards_df)
        >>> print(f"Removed {report['dedup_removed']} duplicates")
    """
    if config is None:
        config = DEFAULT_CONFIG

    df = df.copy()
    report: Dict[str, Any] = {"steps": []}

    report["initial_count"] = len(df)
    logger.info(f"Starting credit card cleaning: {len(df)} records")

    # -------------------------------------------------------------------------
    # Step 1: Normalize card names
    # -------------------------------------------------------------------------
    if "card_name" in df.columns:
        df["card_name_original"] = df["card_name"]  # Keep original for reference
        df["card_name"] = df["card_name"].apply(clean_card_name)

        # Create normalized version for deduplication (lowercase, no spaces)
        df["_card_name_normalized"] = (
            df["card_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        )
        report["steps"].append("card_name_normalized")
        logger.info("Step 1: Card names normalized")
    elif "name" in df.columns:
        # Handle API format where field is 'name' not 'card_name'
        df["card_name_original"] = df["name"]
        df["card_name"] = df["name"].apply(clean_card_name)
        df["_card_name_normalized"] = (
            df["card_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        )
        report["steps"].append("card_name_normalized_from_name")
        logger.info("Step 1: Card names normalized (from 'name' column)")

    # -------------------------------------------------------------------------
    # Step 2: Standardize issuer names
    # -------------------------------------------------------------------------
    if "issuer" in df.columns:
        df["issuer_original"] = df["issuer"]
        df["issuer"] = df["issuer"].apply(
            lambda x: standardize_issuer_name(x, config.issuer_aliases)
        )
        report["unique_issuers"] = int(df["issuer"].nunique())
        report["steps"].append("issuer_standardized")
        logger.info(
            f"Step 2: Issuer names standardized ({report['unique_issuers']} unique)"
        )

    # -------------------------------------------------------------------------
    # Step 3: Deduplication
    # -------------------------------------------------------------------------
    before_dedup = len(df)

    # Only use card_id if there are actual duplicate card_ids
    use_card_id = False
    if "card_id" in df.columns and df["card_id"].notna().any():
        if df["card_id"].duplicated().any():
            use_card_id = True

    if use_card_id:
        df = df.drop_duplicates(subset=["card_id"], keep="first")
        report["dedup_key"] = "card_id"
    elif "_card_name_normalized" in df.columns and "issuer" in df.columns:
        # Dedupe by normalized name + issuer
        df = df.drop_duplicates(
            subset=["_card_name_normalized", "issuer"], keep="first"
        )
        report["dedup_key"] = "card_name_normalized+issuer"
    elif "card_name" in df.columns and "issuer" in df.columns:
        df = df.drop_duplicates(subset=["card_name", "issuer"], keep="first")
        report["dedup_key"] = "card_name+issuer"
    else:
        report["dedup_key"] = "none"

    report["dedup_removed"] = before_dedup - len(df)
    report["after_dedup"] = len(df)
    report["steps"].append("deduplicated")
    logger.info(f"Step 3: Deduplication removed {report['dedup_removed']} records")

    # -------------------------------------------------------------------------
    # Step 4: Handle missing reward_rates
    # -------------------------------------------------------------------------
    if "reward_rates" in df.columns:
        missing_reward = int(df["reward_rates"].isna().sum())

        # Also check for empty dicts
        def is_empty_reward(x):
            if pd.isna(x):
                return True
            if isinstance(x, dict) and len(x) == 0:
                return True
            return False

        empty_reward = df["reward_rates"].apply(is_empty_reward).sum()

        # Impute with default
        default_rate = {"universal_base_rate": 1.0}
        df["reward_rates"] = df["reward_rates"].apply(
            lambda x: x if not is_empty_reward(x) else default_rate
        )

        report["missing_reward_rates"] = int(missing_reward)
        report["empty_reward_rates"] = int(empty_reward)
        report["steps"].append("reward_rates_imputed")
        logger.info(
            f"Step 4: Imputed {missing_reward + empty_reward} missing/empty reward_rates"
        )
    else:
        report["missing_reward_rates"] = 0

    # -------------------------------------------------------------------------
    # Step 5: Validate annual fee
    # -------------------------------------------------------------------------
    if "annual_fee" in df.columns:
        df["annual_fee"] = pd.to_numeric(df["annual_fee"], errors="coerce")

        # Flag invalid fees
        invalid_mask = (
            (df["annual_fee"] < config.min_annual_fee)
            | (df["annual_fee"] >= config.max_annual_fee)
            | df["annual_fee"].isna()
        )

        report["invalid_annual_fees"] = int(invalid_mask.sum())

        # Remove invalid (or just flag - configurable)
        before_fee = len(df)
        df = df.loc[~invalid_mask].copy()
        report["annual_fee_removed"] = before_fee - len(df)
        report["steps"].append("annual_fee_validated")
        logger.info(
            f"Step 5: Removed {report['annual_fee_removed']} invalid annual fees"
        )
    else:
        report["invalid_annual_fees"] = 0
        report["annual_fee_removed"] = 0

    # -------------------------------------------------------------------------
    # Step 6: Parse welcome bonus (if string format)
    # -------------------------------------------------------------------------
    if "welcome_bonus" in df.columns:
        # Check if it's string format that needs parsing
        sample = df["welcome_bonus"].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], str):
            parsed = df["welcome_bonus"].apply(normalize_welcome_bonus)
            df["welcome_bonus_parsed"] = parsed
            df["welcome_bonus_amount"] = parsed.apply(lambda x: x.get("amount"))
            df["welcome_bonus_unit"] = parsed.apply(lambda x: x.get("unit"))
            df["welcome_bonus_spend_req"] = parsed.apply(
                lambda x: x.get("spend_requirement")
            )
            report["steps"].append("welcome_bonus_parsed")
            logger.info("Step 6: Welcome bonus text parsed")

    # -------------------------------------------------------------------------
    # Step 7: Flag discontinued cards
    # -------------------------------------------------------------------------
    if "discontinued" in df.columns:
        df["is_active"] = (~df["discontinued"].fillna(False)).astype(int)
        discontinued_count = df["discontinued"].fillna(False).sum()
        report["discontinued_cards"] = int(discontinued_count)
        report["steps"].append("discontinued_flagged")
        logger.info(f"Step 7: Flagged {discontinued_count} discontinued cards")

    # -------------------------------------------------------------------------
    # Cleanup: Remove temporary columns
    # -------------------------------------------------------------------------
    temp_cols = ["_card_name_normalized"]
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors="ignore")

    # Final report
    report["final_count"] = len(df)
    report["total_removed"] = report["initial_count"] - report["final_count"]

    logger.info(
        f"Credit card cleaning complete: {report['initial_count']} → {report['final_count']} records"
    )
    logger.info(f"Cleaning report: {report}")

    return df.reset_index(drop=True), report


# =============================================================================
# Transaction Data Cleaning
# =============================================================================


def clean_transaction_data(
    df: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
    validate_mcc: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean transaction data with validation and suspicious pattern detection.

    Cleaning steps:
    1. Remove negative amounts
    2. Remove invalid/future dates
    3. Handle missing categories (impute 'unknown')
    4. Validate MCC codes against known codes
    5. Flag suspicious high-value transactions
    6. Detect potential duplicate transactions

    Args:
        df: Raw transaction DataFrame
        config: Cleaning configuration (uses defaults if None)
        validate_mcc: Whether to validate MCC codes (default: True)

    Returns:
        Tuple of (cleaned DataFrame, cleaning report dict)
    """
    if config is None:
        config = DEFAULT_CONFIG

    df = df.copy()
    report: Dict[str, Any] = {"steps": []}

    report["initial_count"] = len(df)
    logger.info(f"Starting transaction cleaning: {len(df)} records")

    now = pd.Timestamp.now()

    # -------------------------------------------------------------------------
    # Step 1: Validate and clean amounts
    # -------------------------------------------------------------------------
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        # Remove negative amounts
        neg_mask = df["amount"] < config.min_transaction_amount
        report["negative_amounts"] = int(neg_mask.sum())

        before = len(df)
        df = df.loc[~neg_mask].copy()
        report["negative_amounts_removed"] = before - len(df)
        report["steps"].append("negative_amounts_removed")
        logger.info(
            f"Step 1: Removed {report['negative_amounts_removed']} negative amounts"
        )
    else:
        report["negative_amounts"] = 0
        report["negative_amounts_removed"] = 0

    # -------------------------------------------------------------------------
    # Step 2: Validate dates
    # -------------------------------------------------------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Invalid dates (unparseable)
        invalid_mask = df["date"].isna()
        report["invalid_dates"] = int(invalid_mask.sum())

        # Future dates
        future_mask = df["date"] > now
        report["future_dates"] = int(future_mask.sum())

        # Very old dates (> 10 years ago might be data errors)
        old_threshold = now - pd.Timedelta(days=365 * 10)
        old_mask = df["date"] < old_threshold
        report["very_old_dates"] = int(old_mask.sum())

        # Remove invalid and future (keep old with flag)
        drop_mask = invalid_mask | future_mask
        before = len(df)
        df = df.loc[~drop_mask].copy()
        report["dates_removed"] = before - len(df)

        # Flag very old
        if "date" in df.columns and len(df) > 0:
            df["is_old_transaction"] = (df["date"] < old_threshold).astype(int)

        report["steps"].append("dates_validated")
        logger.info(f"Step 2: Removed {report['dates_removed']} invalid/future dates")
    else:
        report["invalid_dates"] = 0
        report["future_dates"] = 0
        report["dates_removed"] = 0

    # -------------------------------------------------------------------------
    # Step 3: Handle missing categories
    # -------------------------------------------------------------------------
    if "category" in df.columns:
        missing_cat = int(df["category"].isna().sum())

        # Also check for empty strings
        empty_cat = int((df["category"] == "").sum())

        # Impute with 'unknown'
        df["category"] = df["category"].fillna("unknown")
        df.loc[df["category"] == "", "category"] = "unknown"

        # Standardize category names (lowercase, strip)
        df["category"] = df["category"].str.lower().str.strip()

        report["missing_categories"] = missing_cat + empty_cat
        report["unique_categories"] = int(df["category"].nunique())
        report["steps"].append("categories_cleaned")
        logger.info(
            f"Step 3: Imputed {report['missing_categories']} missing categories"
        )
    else:
        report["missing_categories"] = 0

    # -------------------------------------------------------------------------
    # Step 4: Validate MCC codes
    # -------------------------------------------------------------------------
    if validate_mcc and "mcc_code" in df.columns:
        df["mcc_code"] = pd.to_numeric(df["mcc_code"], errors="coerce")

        # Check against valid codes
        df["mcc_valid"] = df["mcc_code"].isin(config.valid_mcc_codes)

        invalid_mcc = (~df["mcc_valid"]).sum()
        report["invalid_mcc_codes"] = int(invalid_mcc)

        # Don't remove, just flag (MCC might be valid but not in our list)
        report["steps"].append("mcc_validated")
        logger.info(f"Step 4: Found {invalid_mcc} MCC codes not in standard list")
    else:
        report["invalid_mcc_codes"] = 0

    # -------------------------------------------------------------------------
    # Step 5: Flag suspicious transactions
    # -------------------------------------------------------------------------
    if "amount" in df.columns:
        # High value flag
        df["suspicious_high_amount"] = (
            df["amount"] > config.suspicious_amount_threshold
        ).astype(int)
        report["suspicious_high_amounts"] = int(df["suspicious_high_amount"].sum())

        # Round number flag (exact $100, $500, etc. might be refunds or adjustments)
        df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)

        # Combined suspicious flag
        df["suspicious"] = df["suspicious_high_amount"]

        report["steps"].append("suspicious_flagged")
        logger.info(
            f"Step 5: Flagged {report['suspicious_high_amounts']} suspicious transactions"
        )
    else:
        report["suspicious_high_amounts"] = 0

    # -------------------------------------------------------------------------
    # Step 6: Detect potential duplicates
    # -------------------------------------------------------------------------
    if all(c in df.columns for c in ["user_id", "date", "amount", "merchant"]):
        # Exact duplicates on key fields
        dup_mask = df.duplicated(
            subset=["user_id", "date", "amount", "merchant"], keep="first"
        )
        df["is_potential_duplicate"] = dup_mask.astype(int)
        report["potential_duplicates"] = int(dup_mask.sum())
        report["steps"].append("duplicates_flagged")
        logger.info(
            f"Step 6: Flagged {report['potential_duplicates']} potential duplicates"
        )
    else:
        report["potential_duplicates"] = 0

    # Final report
    report["final_count"] = len(df)
    report["total_removed"] = report["initial_count"] - report["final_count"]

    logger.info(
        f"Transaction cleaning complete: {report['initial_count']} → {report['final_count']} records"
    )
    logger.info(f"Cleaning report: {report}")

    return df.reset_index(drop=True), report


# =============================================================================
# User Profile Data Cleaning
# =============================================================================


def clean_user_profile_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean user profile data.

    User profiles from synthetic generator are already clean,
    but this handles edge cases and validates structure.

    Args:
        df: User profile DataFrame

    Returns:
        Tuple of (cleaned DataFrame, cleaning report dict)
    """
    df = df.copy()
    report: Dict[str, Any] = {"steps": []}

    report["initial_count"] = len(df)
    logger.info(f"Starting user profile cleaning: {len(df)} records")

    # Validate user_id uniqueness
    if "user_id" in df.columns:
        dup_users = df["user_id"].duplicated().sum()
        if dup_users > 0:
            df = df.drop_duplicates(subset=["user_id"], keep="first")
            report["duplicate_users_removed"] = int(dup_users)
        else:
            report["duplicate_users_removed"] = 0
        report["steps"].append("user_id_deduplicated")

    # Validate monthly_budget
    if "monthly_budget" in df.columns:
        df["monthly_budget"] = pd.to_numeric(df["monthly_budget"], errors="coerce")
        invalid_budget = df["monthly_budget"].isna().sum()
        df["monthly_budget"] = df["monthly_budget"].fillna(
            df["monthly_budget"].median()
        )
        report["invalid_budgets_imputed"] = int(invalid_budget)
        report["steps"].append("budget_validated")

    # Validate archetype
    if "archetype" in df.columns:
        valid_archetypes = {
            "young_professional",
            "suburban_family",
            "frequent_traveler",
            "budget_conscious",
            "high_roller",
            "minimal_user",
            "category_specialist",
        }
        invalid_archetype = ~df["archetype"].isin(valid_archetypes)
        report["invalid_archetypes"] = int(invalid_archetype.sum())
        # Don't remove, just report
        report["steps"].append("archetype_validated")

    report["final_count"] = len(df)
    logger.info(f"User profile cleaning complete: {len(df)} records")

    return df.reset_index(drop=True), report


# =============================================================================
# Convenience Functions
# =============================================================================


def clean_all_data(
    credit_cards_df: Optional[pd.DataFrame] = None,
    transactions_df: Optional[pd.DataFrame] = None,
    users_df: Optional[pd.DataFrame] = None,
    config: Optional[CleaningConfig] = None,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Dict[str, Any],
]:
    """
    Clean all datasets in one call.

    Args:
        credit_cards_df: Credit card data (optional)
        transactions_df: Transaction data (optional)
        users_df: User profile data (optional)
        config: Cleaning configuration

    Returns:
        Tuple of (clean_cards, clean_transactions, clean_users, combined_report)
    """
    combined_report = {}

    clean_cards = None
    clean_transactions = None
    clean_users = None

    if credit_cards_df is not None:
        clean_cards, cards_report = clean_credit_card_data(credit_cards_df, config)
        combined_report["credit_cards"] = cards_report

    if transactions_df is not None:
        clean_transactions, txn_report = clean_transaction_data(transactions_df, config)
        combined_report["transactions"] = txn_report

    if users_df is not None:
        clean_users, users_report = clean_user_profile_data(users_df)
        combined_report["users"] = users_report

    return clean_cards, clean_transactions, clean_users, combined_report


# =============================================================================
# Module Info
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("RewardSense Data Cleaning Module (Enhanced)")
    print("Story 3.1: Comprehensive Data Cleaning")
    print("=" * 70)
    print("\nFeatures:")
    print("  ✅ Card name normalization (removes ®™, standardizes whitespace)")
    print("  ✅ Issuer name standardization (aliases, uppercase)")
    print("  ✅ Smart deduplication (by card_id or normalized name+issuer)")
    print("  ✅ Welcome bonus text parsing")
    print("  ✅ MCC code validation")
    print("  ✅ Suspicious transaction flagging")
    print("  ✅ Configurable thresholds")
    print("  ✅ Comprehensive cleaning reports")
    print("\nUsage:")
    print("  from data_pipeline.preprocessing.cleaning import clean_all_data")
    print("  cards, txns, users, report = clean_all_data(cards_df, txns_df, users_df)")
    print("=" * 70)
