"""
Transaction scoring module for RewardSense.

Scores credit cards against transactions using the RewardCalculator,
and supports batch scoring across multiple transactions for portfolio optimization.
"""

import logging
from typing import Dict, Any, List, Optional

from src.model_pipeline.scoring.reward_calculator import RewardCalculator
from src.model_pipeline.scoring.card_ranker import CardRanker
from src.model_pipeline.scoring.spending_cap_tracker import SpendingCapTracker

logger = logging.getLogger(__name__)


class TransactionScorer:
    """
    Scores credit cards against transactions.

    Uses RewardCalculator to compute reward values, then packages
    results with metadata for ranking and downstream consumption.
    Optionally enforces spending caps via SpendingCapTracker.
    """

    def __init__(
        self,
        calculator: Optional[RewardCalculator] = None,
        cap_tracker: Optional[SpendingCapTracker] = None,
    ):
        """
        Initialize scorer with an optional custom RewardCalculator and cap tracker.

        Args:
            calculator: RewardCalculator instance. If None, uses default.
            cap_tracker: SpendingCapTracker instance. If None, caps are not enforced.
        """
        self.calculator = calculator or RewardCalculator()
        self.cap_tracker = cap_tracker
        self._ranker = CardRanker()
        logger.info("Initialized TransactionScorer")

    def score_card(
        self, card: Dict[str, Any], transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a single card against a single transaction.

        If a cap_tracker is set and the card defines category_caps,
        checks whether the bonus category cap has been exceeded.
        If so, scores using a base-rate-only version of the card.
        After scoring, records the transaction in the tracker.

        Args:
            card: Credit card dict
            transaction: Transaction dict

        Returns:
            Dict with card_id, card_name, reward_amount, reward_rate, annual_fee
        """
        amount = float(transaction.get("amount", 0))
        category = transaction.get("category", "general")
        card_id = card.get("card_id", "")

        scoring_card = card

        # Check cap enforcement
        if self.cap_tracker is not None:
            reward_rates = card.get("reward_rates") or {}
            caps = reward_rates.get("category_caps", {})
            cap = caps.get(category)

            if cap is not None:
                remaining = self.cap_tracker.get_remaining_cap(card_id, category, cap)
                if remaining <= 0 or remaining < amount:
                    # Cap exceeded — strip category bonus so calculator uses base rate
                    scoring_card = dict(card)
                    base_rates = {
                        "universal_base_rate": reward_rates.get(
                            "universal_base_rate", 1.0
                        )
                    }
                    scoring_card["reward_rates"] = base_rates

        reward_amount = self.calculator.calculate_reward(scoring_card, transaction)
        reward_rate = (reward_amount / amount * 100) if amount > 0 else 0.0

        # Record spend in tracker
        if self.cap_tracker is not None and amount > 0:
            self.cap_tracker.record_transaction(card_id, category, amount)

        return {
            "card_id": card_id,
            "card_name": card.get("card_name", ""),
            "reward_amount": reward_amount,
            "reward_rate": reward_rate,
            "annual_fee": card.get("annual_fee", 0),
        }

    def score_portfolio(
        self, portfolio: List[Dict[str, Any]], transaction: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Score every card in a portfolio against one transaction.

        Args:
            portfolio: List of credit card dicts
            transaction: Transaction dict

        Returns:
            List of scored card dicts (unranked)
        """
        if not portfolio:
            return []

        return [self.score_card(card, transaction) for card in portfolio]

    def score_batch(
        self, portfolio: List[Dict[str, Any]], transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score a portfolio against multiple transactions.

        Args:
            portfolio: List of credit card dicts
            transactions: List of transaction dicts

        Returns:
            List of batch result dicts, one per transaction. Each contains:
            - transaction: the original transaction
            - scores: list of scored card dicts
            - best_card_id: card_id of the top-ranked card (or None)
        """
        if not transactions:
            return []

        results = []
        for txn in transactions:
            scored = self.score_portfolio(portfolio, txn)
            best = self._ranker.get_best_card(scored)

            results.append(
                {
                    "transaction": txn,
                    "scores": scored,
                    "best_card_id": best["card_id"] if best else None,
                }
            )

        logger.info(
            f"Batch scored {len(transactions)} transactions against {len(portfolio)} cards"
        )
        return results
