"""
Spending cap tracker for RewardSense.

Tracks cumulative spending per card per category against quarterly/annual caps.
Used by the scoring engine to determine if bonus rates still apply.
"""

import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SpendingCapTracker:
    """
    Tracks spending against category caps for a given user.
    
    Credit cards often cap bonus rewards (e.g., "5% on groceries up to
    $6,000/quarter"). This tracker maintains cumulative spend per
    (card, category) pair so the scoring engine knows when a cap is hit.
    """

    def __init__(self, user_id: str):
        """
        Initialize tracker for a specific user.
        
        Args:
            user_id: The user whose spending is being tracked
        """
        self.user_id = user_id
        # Nested dict: {(card_id, category): cumulative_amount}
        self._spending: Dict[Tuple[str, str], float] = defaultdict(float)
        logger.info(f"Initialized SpendingCapTracker for user={user_id}")

    def record_transaction(self, card_id: str, category: str, amount: float) -> None:
        """
        Record a transaction against a card+category bucket.
        
        Args:
            card_id: Credit card identifier
            category: Spending category (e.g., 'dining', 'gas')
            amount: Transaction amount in dollars
        """
        if amount < 0:
            logger.warning(f"Negative amount {amount} for {card_id}/{category}, treating as 0")
            amount = 0.0

        key = (card_id, category)
        self._spending[key] += amount
        logger.debug(
            f"Recorded ${amount:.2f} for {card_id}/{category}. "
            f"Cumulative: ${self._spending[key]:.2f}"
        )

    def get_remaining_cap(self, card_id: str, category: str,
                          cap: float, spent_so_far: Optional[float] = None) -> float:
        """
        Calculate remaining spend before a bonus cap is hit.
        
        Args:
            card_id: Credit card identifier
            category: Spending category
            cap: The spending cap amount (e.g., 6000.0 for $6k quarterly cap)
            spent_so_far: If provided, use this as cumulative spend instead
                          of internal tracking. Useful when loading from external data.
        
        Returns:
            Remaining dollars before cap is reached. Never negative.
        """
        if spent_so_far is not None:
            spent = spent_so_far
        else:
            spent = self._spending.get((card_id, category), 0.0)

        remaining = max(0.0, cap - spent)
        return remaining

    def get_spent(self, card_id: str, category: str) -> float:
        """
        Get cumulative amount spent for a card+category.
        
        Args:
            card_id: Credit card identifier
            category: Spending category
        
        Returns:
            Cumulative spend in dollars
        """
        return self._spending.get((card_id, category), 0.0)

    def is_cap_reached(self, card_id: str, category: str, cap: float) -> bool:
        """
        Check if a spending cap has been reached.
        
        Args:
            card_id: Credit card identifier
            category: Spending category
            cap: The cap amount
        
        Returns:
            True if cumulative spend >= cap
        """
        return self.get_spent(card_id, category) >= cap

    def reset(self, card_id: Optional[str] = None, category: Optional[str] = None) -> None:
        """
        Reset tracked spending. Useful for quarterly cap resets.
        
        Args:
            card_id: If provided with category, reset only that bucket.
                     If None, reset everything.
            category: Category to reset (requires card_id).
        """
        if card_id and category:
            key = (card_id, category)
            self._spending[key] = 0.0
            logger.info(f"Reset spending for {card_id}/{category}")
        else:
            self._spending.clear()
            logger.info(f"Reset all spending for user={self.user_id}")