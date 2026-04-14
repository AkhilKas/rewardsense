"""
Integration layer between the deterministic scoring engine (Epic 2)
and the ML personalization model (Epic 3).

Story 3.5 — Scoring Engine + Personalization Model Integration.

Flow:
1. Score all cards in a user's portfolio via ``TransactionScorer``
2. Load the user's feature vector and run the personalization model
3. Adjust each card's ``reward_amount`` by the user's predicted point
   valuation (a multiplicative weight)
4. Re-rank the adjusted scores via ``CardRanker``
5. Fall back to unpersonalized scores for cold-start users
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.model_pipeline.scoring.card_ranker import CardRanker
from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

# Default point-value multiplier for cold-start users (no feature data).
# Must be 1.0 because TransactionScorer.calculate_reward already returns
# dollar amounts (amount * rate / 100). A value of 0.01 would incorrectly
# divide by 100 again, making a $150 reward appear as $1.50.
# When the ML model is loaded it returns a user-specific multiplier != 1.0
# to weight point-earning cards relative to cashback cards based on how
# much that user actually values transferable points.
DEFAULT_POINT_VALUE = 1.0


class PersonalizedScorer:
    """Score and rank cards with per-user personalization.

    Parameters
    ----------
    model : sklearn-compatible estimator or None
        Fitted personalization model with a ``predict`` method.
        If None, the scorer operates in unpersonalized (fallback) mode.
    scorer : TransactionScorer or None
        If None, a default ``TransactionScorer`` is created.
    ranker : CardRanker or None
        If None, a default ``CardRanker`` is created.
    default_point_value : float
        Fallback multiplier for cold-start users.
    """

    def __init__(
        self,
        model: Any = None,
        scorer: Optional[TransactionScorer] = None,
        ranker: Optional[CardRanker] = None,
        default_point_value: float = DEFAULT_POINT_VALUE,
    ) -> None:
        self.model = model
        self.scorer = scorer or TransactionScorer()
        self.ranker = ranker or CardRanker()
        self.default_point_value = default_point_value

    @classmethod
    def from_artifact(
        cls,
        model_path: str | Path,
        **kwargs: Any,
    ) -> "PersonalizedScorer":
        """Load a personalization model from a joblib artifact.

        Parameters
        ----------
        model_path : str or Path
            Path to the ``.joblib`` model file.
        **kwargs
            Forwarded to ``__init__``.
        """
        model = joblib.load(model_path)
        logger.info("Loaded personalization model from {}", model_path)
        return cls(model=model, **kwargs)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def score(
        self,
        portfolio: List[Dict[str, Any]],
        transaction: Dict[str, Any],
        user_features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Score a portfolio for a single transaction with personalization.

        Parameters
        ----------
        portfolio : list of dict
            Credit card dicts (as expected by ``TransactionScorer``).
        transaction : dict
            Transaction dict with ``amount``, ``category``, etc.
        user_features : pd.DataFrame or None
            Single-row DataFrame with the user's feature vector.
            If None, cold-start fallback is used.

        Returns
        -------
        dict with keys:
            ``ranked``  – list of scored+ranked card dicts
            ``best_card_id`` – card_id of the top pick (or None)
            ``point_value`` – the personalization multiplier used
            ``is_personalized`` – whether the model was used
        """
        raw_scores = self.scorer.score_portfolio(portfolio, transaction)

        point_value, is_personalized = self._get_point_value(user_features)

        adjusted = self._apply_personalization(raw_scores, point_value)

        ranked = self.ranker.rank(adjusted)

        return {
            "ranked": ranked,
            "best_card_id": ranked[0]["card_id"] if ranked else None,
            "point_value": point_value,
            "is_personalized": is_personalized,
        }

    def score_batch(
        self,
        portfolio: List[Dict[str, Any]],
        transactions: List[Dict[str, Any]],
        user_features: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Score a portfolio against multiple transactions.

        Personalization weight is computed once and applied to every
        transaction (it's a user-level attribute, not transaction-level).
        """
        point_value, is_personalized = self._get_point_value(user_features)

        results: List[Dict[str, Any]] = []
        for txn in transactions:
            raw_scores = self.scorer.score_portfolio(portfolio, txn)
            adjusted = self._apply_personalization(raw_scores, point_value)
            ranked = self.ranker.rank(adjusted)
            results.append(
                {
                    "transaction": txn,
                    "ranked": ranked,
                    "best_card_id": ranked[0]["card_id"] if ranked else None,
                    "point_value": point_value,
                    "is_personalized": is_personalized,
                }
            )
        return results

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _get_point_value(
        self, user_features: Optional[pd.DataFrame]
    ) -> tuple[float, bool]:
        """Predict the user's point valuation or return the default.

        Returns ``(point_value, is_personalized)``.
        """
        if user_features is None or self.model is None:
            return self.default_point_value, False

        try:
            prediction = self.model.predict(user_features)
            value = float(np.atleast_1d(prediction)[0])
            if np.isnan(value) or value <= 0:
                logger.warning("Invalid prediction {}, falling back to default", value)
                return self.default_point_value, False
            return value, True
        except Exception as exc:
            logger.warning("Personalization prediction failed: {}", exc)
            return self.default_point_value, False

    @staticmethod
    def _apply_personalization(
        scores: List[Dict[str, Any]],
        point_value: float,
    ) -> List[Dict[str, Any]]:
        """Multiply each card's reward_amount by the point-value weight.

        The adjustment scales raw dollar-rewards into user-perceived
        value.  Relative ordering can change when different users
        value points differently (e.g. a travel-focused user values
        transferable points higher than flat cashback).
        """
        adjusted: List[Dict[str, Any]] = []
        for s in scores:
            entry = dict(s)
            entry["raw_reward_amount"] = entry.get("reward_amount", 0)
            entry["reward_amount"] = entry["raw_reward_amount"] * point_value
            adjusted.append(entry)
        return adjusted
