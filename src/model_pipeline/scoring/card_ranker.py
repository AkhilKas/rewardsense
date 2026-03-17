"""
Card ranking module for RewardSense.

Ranks scored cards by reward value with deterministic tie-breaking:
1. Highest reward amount
2. Lowest annual fee
3. Alphabetical card_id
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CardRanker:
    """
    Ranks scored credit cards with deterministic tie-breaking.
    
    Sort order:
        1. reward_amount descending (higher reward wins)
        2. annual_fee ascending (lower fee wins on tie)
        3. card_id ascending (alphabetical for full determinism)
    """

    def __init__(self):
        logger.info("Initialized CardRanker")

    def rank(self, scored_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank scored cards and assign rank positions.
        
        Args:
            scored_cards: List of dicts with at least card_id, reward_amount, annual_fee
        
        Returns:
            New list sorted best-to-worst, each dict augmented with 'rank' (1-indexed)
        """
        if not scored_cards:
            return []

        ranked = sorted(
            scored_cards,
            key=lambda c: (
                -c.get('reward_amount', 0),   # higher reward first
                c.get('annual_fee', 0),        # lower fee first
                c.get('card_id', ''),          # alphabetical fallback
            ),
        )

        # Assign rank positions (1-indexed)
        for i, card in enumerate(ranked):
            card = dict(card)       # shallow copy so we don't mutate input
            card['rank'] = i + 1
            ranked[i] = card

        return ranked

    def get_best_card(self, scored_cards: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Convenience method: return the top-ranked card or None.
        
        Args:
            scored_cards: List of scored card dicts
        
        Returns:
            Top-ranked card dict with 'rank' == 1, or None if empty
        """
        ranked = self.rank(scored_cards)
        return ranked[0] if ranked else None