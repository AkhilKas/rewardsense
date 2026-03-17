"""
Core reward calculation logic for RewardSense.

Calculates expected reward value for a (card, transaction) pair.
Handles base rates, category bonuses, rotating bonuses, annual fee amortization,
welcome bonus eligibility, statement credits, and edge cases.
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


# Quarter mapping: month -> quarter label
_MONTH_TO_QUARTER = {
    1: "Q1",
    2: "Q1",
    3: "Q1",
    4: "Q2",
    5: "Q2",
    6: "Q2",
    7: "Q3",
    8: "Q3",
    9: "Q3",
    10: "Q4",
    11: "Q4",
    12: "Q4",
}


class RewardCalculator:
    """
    Calculates reward value for credit card transactions.

    Supports:
    - Universal base reward rates
    - Category-specific bonus rates
    - Rotating quarterly bonus categories
    - Annual fee amortization
    - Foreign transaction fees
    - Welcome bonus eligibility checks
    - Statement credit offsets
    - Spending caps (via external tracker)
    """

    def __init__(
        self,
        amortize_annual_fee: bool = False,
        amortization_period_months: int = 12,
        default_reward_rate: float = 1.0,
        include_statement_credits: bool = False,
    ):
        """
        Initialize reward calculator.

        Args:
            amortize_annual_fee: If True, subtract amortized annual fee from rewards
            amortization_period_months: Period for fee amortization (default: 12 months)
            default_reward_rate: Default rate if card has no reward_rates (default: 1.0%)
            include_statement_credits: If True, factor statement credits into reward calculations
        """
        self.amortize_annual_fee = amortize_annual_fee
        self.amortization_period_months = amortization_period_months
        self.default_reward_rate = default_reward_rate
        self.include_statement_credits = include_statement_credits

        logger.info(
            f"Initialized RewardCalculator (amortize_fee={amortize_annual_fee}, "
            f"statement_credits={include_statement_credits})"
        )

    def calculate_reward(
        self, card: Dict[str, Any], transaction: Dict[str, Any]
    ) -> float:
        """
        Calculate reward value for a transaction with a given card.

        Args:
            card: Credit card dict with reward_rates, annual_fee, etc.
            transaction: Transaction dict with amount, category, mcc_code, etc.

        Returns:
            Reward value in dollars
        """
        amount = float(transaction.get("amount", 0))

        # Edge case: zero amount
        if amount == 0:
            return 0.0

        # Extract reward rate (handles base, category bonuses, rotating bonuses)
        reward_rate = self._extract_reward_rate(card, transaction)

        # Calculate base reward (rate as percentage)
        base_reward = amount * (reward_rate / 100)

        # Handle foreign transaction fee
        if transaction.get("is_foreign", False):
            foreign_fee_pct = card.get("foreign_transaction_fee_pct", 0)
            foreign_fee = amount * (foreign_fee_pct / 100)
            base_reward -= foreign_fee

        # Amortize annual fee if configured
        if self.amortize_annual_fee:
            annual_fee = float(card.get("annual_fee", 0))
            monthly_fee_impact = annual_fee / self.amortization_period_months
            fee_impact_per_transaction = monthly_fee_impact / 30
            base_reward -= fee_impact_per_transaction

        return base_reward

    def calculate_reward_with_credits(
        self, card: Dict[str, Any], transaction: Dict[str, Any]
    ) -> float:
        """
        Calculate reward value including statement credit offsets.

        Statement credits (e.g. "$10/month dining credit") add effective value
        on top of the base reward rate for matching categories.

        Args:
            card: Credit card dict with statement_credits field
            transaction: Transaction dict

        Returns:
            Effective reward value in dollars (base reward + credit value)
        """
        base_reward = self.calculate_reward(card, transaction)

        if not self.include_statement_credits:
            return base_reward

        category = transaction.get("category", "general")
        credits = card.get("statement_credits", {})

        credit_info = credits.get(category)
        if credit_info is None:
            return base_reward

        credit_amount = float(credit_info.get("amount", 0))
        frequency = credit_info.get("frequency", "monthly")

        # Normalize credit to a per-transaction estimate
        # Monthly credit spread across assumed ~4 transactions in that category
        if frequency == "monthly":
            per_txn_credit = credit_amount / 4
        elif frequency == "quarterly":
            per_txn_credit = credit_amount / 12
        elif frequency == "annual":
            per_txn_credit = credit_amount / 48
        else:
            per_txn_credit = credit_amount / 4

        # Cap credit at the transaction amount (can't earn more credit than you spend)
        per_txn_credit = min(per_txn_credit, float(transaction.get("amount", 0)))

        return base_reward + per_txn_credit

    def is_welcome_bonus_eligible(
        self, card: Dict[str, Any], user_status: Dict[str, Any]
    ) -> bool:
        """
        Check if a user is eligible for a card's welcome bonus.

        Eligibility requires:
        - Card has a welcome bonus defined
        - User hasn't already received the bonus
        - User is still within the qualifying time window

        Args:
            card: Credit card dict with welcome_bonus field
            user_status: Dict with user_id, card_tenure_days, total_spent_on_card,
                         and optionally welcome_bonus_received

        Returns:
            True if user is eligible for welcome bonus
        """
        welcome_bonus = card.get("welcome_bonus")
        if not welcome_bonus:
            return False

        # Already received the bonus
        if user_status.get("welcome_bonus_received", False):
            return False

        # Check if still within the qualifying time window
        tenure_days = user_status.get("card_tenure_days", 0)
        days_to_complete = welcome_bonus.get("days_to_complete", 90)

        if tenure_days > days_to_complete:
            return False

        # Check if spend requirement already met (bonus would have been triggered)
        spend_requirement = float(welcome_bonus.get("spend_requirement", 0))
        total_spent = float(user_status.get("total_spent_on_card", 0))

        if spend_requirement > 0 and total_spent >= spend_requirement:
            # Met the requirement but within window and not received → still eligible
            # (bonus just hasn't posted yet, or this is a check before it posts)
            return True

        return True

    def _extract_reward_rate(
        self, card: Dict[str, Any], transaction: Dict[str, Any]
    ) -> float:
        """
        Extract the applicable reward rate for this transaction.

        Priority order:
        1. Rotating quarterly bonus (if active for this quarter + category)
        2. Category-specific bonus
        3. Universal base rate
        4. Default fallback

        Args:
            card: Credit card dict
            transaction: Transaction dict

        Returns:
            Reward rate as percentage
        """
        reward_rates = card.get("reward_rates")

        if reward_rates is None or not isinstance(reward_rates, dict):
            logger.warning(
                f"Card {card.get('card_id')} missing reward_rates, using default"
            )
            return self.default_reward_rate

        category = transaction.get("category", "general")
        base_rate = float(
            reward_rates.get("universal_base_rate", self.default_reward_rate)
        )

        # 1. Check rotating quarterly bonuses
        rotating_bonuses = reward_rates.get("rotating_bonuses")
        if rotating_bonuses and isinstance(rotating_bonuses, dict):
            txn_date = transaction.get("date")
            if txn_date is not None:
                if isinstance(txn_date, datetime):
                    quarter = _MONTH_TO_QUARTER[txn_date.month]
                else:
                    quarter = None

                if quarter and quarter in rotating_bonuses:
                    bonus_info = rotating_bonuses[quarter]
                    bonus_categories = bonus_info.get("categories", [])
                    if category in bonus_categories:
                        return float(bonus_info.get("rate", base_rate))

        # 2. Check category-specific bonuses
        category_bonuses = reward_rates.get("category_bonuses")
        if category_bonuses and isinstance(category_bonuses, dict):
            if category in category_bonuses:
                return float(category_bonuses[category])

        # 3. Universal base rate
        return base_rate
