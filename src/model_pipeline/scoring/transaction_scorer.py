"""
Transaction scoring module for RewardSense.

Scores credit cards against transactions using the RewardCalculator,
and supports batch scoring across multiple transactions for portfolio optimization.
"""

import logging
from typing import Dict, Any, List, Optional

from src.model_pipeline.scoring.reward_calculator import RewardCalculator
from src.model_pipeline.scoring.card_ranker import CardRanker

logger = logging.getLogger(__name__)


class TransactionScorer:
    """
    Scores credit cards against transactions.

    Uses RewardCalculator to compute reward values, then packages
    results with metadata for ranking and downstream consumption.
    """

    def __init__(self, calculator: Optional[RewardCalculator] = None):
        """
        Initialize scorer with an optional custom RewardCalculator.

        Args:
            calculator: RewardCalculator instance. If None, uses default.
        """
        self.calculator = calculator or RewardCalculator()
        self._ranker = CardRanker()
        logger.info("Initialized TransactionScorer")

    def score_card(
        self, card: Dict[str, Any], transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a single card against a single transaction.

        Args:
            card: Credit card dict
            transaction: Transaction dict

        Returns:
            Dict with card_id, card_name, reward_amount, reward_rate, annual_fee
        """
        amount = float(transaction.get("amount", 0))
        reward_amount = self.calculator.calculate_reward(card, transaction)
        reward_rate = (reward_amount / amount * 100) if amount > 0 else 0.0

        return {
            "card_id": card.get("card_id"),
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
