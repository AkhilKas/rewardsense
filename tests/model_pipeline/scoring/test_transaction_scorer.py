"""
TDD Tests for TransactionScorer & CardRanker - Story 2.2

Tests written BEFORE implementation.
Covers: single card scoring, multi-card ranking, tie-breaking,
batch scoring, and portfolio-level optimization.
"""

import pytest
from datetime import datetime


# ── Single Card Scoring ──────────────────────────────────────────────

class TestTransactionScorer:
    """Test scoring a single card against a transaction."""

    def test_score_single_card(self):
        """Score one card for one transaction returns a ScoredCard result."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

        card = {
            'card_id': 'card_001',
            'card_name': 'Simple Cash Card',
            'reward_rates': {'universal_base_rate': 2.0},
            'annual_fee': 0
        }

        transaction = {
            'amount': 100.0,
            'category': 'dining',
            'merchant': 'Starbucks',
            'mcc_code': 5812
        }

        result = scorer.score_card(card, transaction)

        assert result['card_id'] == 'card_001'
        assert result['reward_amount'] == 2.0  # 2% of $100
        assert 'reward_rate' in result

    def test_score_card_includes_metadata(self):
        """Scored result includes card name and annual fee for downstream use."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

        card = {
            'card_id': 'card_002',
            'card_name': 'Premium Card',
            'reward_rates': {'universal_base_rate': 1.0},
            'annual_fee': 550
        }

        transaction = {
            'amount': 50.0,
            'category': 'travel',
            'merchant': 'United Airlines',
            'mcc_code': 3000
        }

        result = scorer.score_card(card, transaction)

        assert result['card_name'] == 'Premium Card'
        assert result['annual_fee'] == 550

    def test_score_portfolio(self):
        """Score all cards in a user's portfolio against one transaction."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

        portfolio = [
            {
                'card_id': 'card_001',
                'card_name': 'Card A',
                'reward_rates': {'universal_base_rate': 1.0},
                'annual_fee': 0
            },
            {
                'card_id': 'card_002',
                'card_name': 'Card B',
                'reward_rates': {
                    'universal_base_rate': 1.0,
                    'category_bonuses': {'dining': 3.0}
                },
                'annual_fee': 95
            },
            {
                'card_id': 'card_003',
                'card_name': 'Card C',
                'reward_rates': {'universal_base_rate': 2.0},
                'annual_fee': 0
            },
        ]

        transaction = {
            'amount': 100.0,
            'category': 'dining',
            'merchant': 'Restaurant',
            'mcc_code': 5812
        }

        results = scorer.score_portfolio(portfolio, transaction)

        assert len(results) == 3
        # Each result should have card_id and reward_amount
        card_ids = [r['card_id'] for r in results]
        assert 'card_001' in card_ids
        assert 'card_002' in card_ids
        assert 'card_003' in card_ids

    def test_score_portfolio_empty(self):
        """Empty portfolio returns empty list."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

        transaction = {
            'amount': 100.0,
            'category': 'dining',
            'merchant': 'Restaurant',
            'mcc_code': 5812
        }

        results = scorer.score_portfolio([], transaction)
        assert results == []


# ── Card Ranking ─────────────────────────────────────────────────────

class TestCardRanker:
    """Test ranking scored cards with tie-breaking logic."""

    def test_rank_by_reward_amount(self):
        """Cards ranked by reward amount descending."""
        from src.model_pipeline.scoring.card_ranker import CardRanker

        ranker = CardRanker()

        scored_cards = [
            {'card_id': 'card_A', 'card_name': 'A', 'reward_amount': 1.0, 'annual_fee': 0},
            {'card_id': 'card_B', 'card_name': 'B', 'reward_amount': 3.0, 'annual_fee': 95},
            {'card_id': 'card_C', 'card_name': 'C', 'reward_amount': 2.0, 'annual_fee': 0},
        ]

        ranked = ranker.rank(scored_cards)

        assert ranked[0]['card_id'] == 'card_B'  # $3.00 reward
        assert ranked[1]['card_id'] == 'card_C'  # $2.00 reward
        assert ranked[2]['card_id'] == 'card_A'  # $1.00 reward

    def test_rank_includes_position(self):
        """Each ranked result includes its rank position (1-indexed)."""
        from src.model_pipeline.scoring.card_ranker import CardRanker

        ranker = CardRanker()

        scored_cards = [
            {'card_id': 'card_A', 'card_name': 'A', 'reward_amount': 5.0, 'annual_fee': 0},
            {'card_id': 'card_B', 'card_name': 'B', 'reward_amount': 2.0, 'annual_fee': 0},
        ]

        ranked = ranker.rank(scored_cards)

        assert ranked[0]['rank'] == 1
        assert ranked[1]['rank'] == 2

    def test_tiebreak_by_lower_annual_fee(self):
        """When reward amounts are equal, prefer lower annual fee."""
        from src.model_pipeline.scoring.card_ranker import CardRanker

        ranker = CardRanker()

        scored_cards = [
            {'card_id': 'card_A', 'card_name': 'A', 'reward_amount': 3.0, 'annual_fee': 250},
            {'card_id': 'card_B', 'card_name': 'B', 'reward_amount': 3.0, 'annual_fee': 0},
            {'card_id': 'card_C', 'card_name': 'C', 'reward_amount': 3.0, 'annual_fee': 95},
        ]

        ranked = ranker.rank(scored_cards)

        # Same reward → lower annual fee wins
        assert ranked[0]['card_id'] == 'card_B'  # $0 fee
        assert ranked[1]['card_id'] == 'card_C'  # $95 fee
        assert ranked[2]['card_id'] == 'card_A'  # $250 fee

    def test_tiebreak_deterministic_by_card_id(self):
        """When reward AND annual fee are equal, sort by card_id for determinism."""
        from src.model_pipeline.scoring.card_ranker import CardRanker

        ranker = CardRanker()

        scored_cards = [
            {'card_id': 'card_Z', 'card_name': 'Z', 'reward_amount': 2.0, 'annual_fee': 0},
            {'card_id': 'card_A', 'card_name': 'A', 'reward_amount': 2.0, 'annual_fee': 0},
        ]

        ranked = ranker.rank(scored_cards)

        # Same reward, same fee → alphabetical card_id
        assert ranked[0]['card_id'] == 'card_A'
        assert ranked[1]['card_id'] == 'card_Z'

    def test_rank_single_card(self):
        """Single card portfolio returns rank 1."""
        from src.model_pipeline.scoring.card_ranker import CardRanker

        ranker = CardRanker()

        scored_cards = [
            {'card_id': 'card_A', 'card_name': 'A', 'reward_amount': 1.5, 'annual_fee': 0},
        ]

        ranked = ranker.rank(scored_cards)

        assert len(ranked) == 1
        assert ranked[0]['rank'] == 1

    def test_rank_empty_returns_empty(self):
        """Empty input returns empty list."""
        from src.model_pipeline.scoring.card_ranker import CardRanker

        ranker = CardRanker()
        ranked = ranker.rank([])
        assert ranked == []

    def test_get_best_card(self):
        """Convenience method returns the top-ranked card."""
        from src.model_pipeline.scoring.card_ranker import CardRanker

        ranker = CardRanker()

        scored_cards = [
            {'card_id': 'card_A', 'card_name': 'A', 'reward_amount': 1.0, 'annual_fee': 0},
            {'card_id': 'card_B', 'card_name': 'B', 'reward_amount': 5.0, 'annual_fee': 95},
        ]

        best = ranker.get_best_card(scored_cards)

        assert best['card_id'] == 'card_B'
        assert best['rank'] == 1

    def test_get_best_card_empty_returns_none(self):
        """get_best_card on empty list returns None."""
        from src.model_pipeline.scoring.card_ranker import CardRanker

        ranker = CardRanker()
        assert ranker.get_best_card([]) is None


# ── Batch Scoring ────────────────────────────────────────────────────

class TestBatchScoring:
    """Test scoring multiple transactions at once."""

    def test_batch_score_multiple_transactions(self):
        """Score a portfolio against multiple transactions."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

        portfolio = [
            {
                'card_id': 'card_001',
                'card_name': 'Base Card',
                'reward_rates': {'universal_base_rate': 1.0},
                'annual_fee': 0
            },
            {
                'card_id': 'card_002',
                'card_name': 'Dining Card',
                'reward_rates': {
                    'universal_base_rate': 1.0,
                    'category_bonuses': {'dining': 4.0}
                },
                'annual_fee': 0
            },
        ]

        transactions = [
            {'amount': 50.0, 'category': 'dining', 'merchant': 'Chipotle', 'mcc_code': 5812},
            {'amount': 80.0, 'category': 'gas', 'merchant': 'Shell', 'mcc_code': 5541},
            {'amount': 200.0, 'category': 'travel', 'merchant': 'Delta', 'mcc_code': 3000},
        ]

        batch_results = scorer.score_batch(portfolio, transactions)

        # One result set per transaction
        assert len(batch_results) == 3

        # Each result set has scores for all cards in portfolio
        for result in batch_results:
            assert len(result['scores']) == 2
            assert 'best_card_id' in result
            assert 'transaction' in result

    def test_batch_score_returns_best_card_per_transaction(self):
        """Each batch result identifies the best card for that transaction."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

        portfolio = [
            {
                'card_id': 'base_card',
                'card_name': 'Base',
                'reward_rates': {'universal_base_rate': 1.5},
                'annual_fee': 0
            },
            {
                'card_id': 'dining_card',
                'card_name': 'Dining',
                'reward_rates': {
                    'universal_base_rate': 1.0,
                    'category_bonuses': {'dining': 5.0}
                },
                'annual_fee': 0
            },
        ]

        transactions = [
            {'amount': 100.0, 'category': 'dining', 'merchant': 'Olive Garden', 'mcc_code': 5812},
            {'amount': 100.0, 'category': 'gas', 'merchant': 'BP', 'mcc_code': 5541},
        ]

        batch_results = scorer.score_batch(portfolio, transactions)

        # Dining → dining_card wins (5% vs 1.5%)
        assert batch_results[0]['best_card_id'] == 'dining_card'

        # Gas → base_card wins (1.5% vs 1.0%)
        assert batch_results[1]['best_card_id'] == 'base_card'

    def test_batch_score_empty_transactions(self):
        """Empty transaction list returns empty results."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

        portfolio = [
            {
                'card_id': 'card_001',
                'card_name': 'Card',
                'reward_rates': {'universal_base_rate': 1.0},
                'annual_fee': 0
            }
        ]

        batch_results = scorer.score_batch(portfolio, [])
        assert batch_results == []

    def test_batch_score_empty_portfolio(self):
        """Empty portfolio returns results with no scores."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

        transactions = [
            {'amount': 100.0, 'category': 'dining', 'merchant': 'Test', 'mcc_code': 5812},
        ]

        batch_results = scorer.score_batch([], transactions)

        assert len(batch_results) == 1
        assert batch_results[0]['scores'] == []
        assert batch_results[0]['best_card_id'] is None


# ── End-to-End: Scorer + Ranker ──────────────────────────────────────

class TestScorerRankerIntegration:
    """Test TransactionScorer + CardRanker working together."""

    def test_score_and_rank_full_flow(self):
        """Score a portfolio then rank — top card is correct."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer
        from src.model_pipeline.scoring.card_ranker import CardRanker

        scorer = TransactionScorer()
        ranker = CardRanker()

        portfolio = [
            {
                'card_id': 'sapphire',
                'card_name': 'Chase Sapphire Reserve',
                'reward_rates': {
                    'universal_base_rate': 1.0,
                    'category_bonuses': {'dining': 3.0, 'travel': 3.0}
                },
                'annual_fee': 550
            },
            {
                'card_id': 'freedom',
                'card_name': 'Chase Freedom Unlimited',
                'reward_rates': {'universal_base_rate': 1.5},
                'annual_fee': 0
            },
            {
                'card_id': 'gold',
                'card_name': 'Amex Gold',
                'reward_rates': {
                    'universal_base_rate': 1.0,
                    'category_bonuses': {'dining': 4.0, 'groceries': 4.0}
                },
                'annual_fee': 250
            },
        ]

        # Dining transaction
        transaction = {
            'amount': 75.0,
            'category': 'dining',
            'merchant': 'Nobu',
            'mcc_code': 5812
        }

        scored = scorer.score_portfolio(portfolio, transaction)
        ranked = ranker.rank(scored)

        # Amex Gold: 4% on dining = $3.00
        # Sapphire Reserve: 3% on dining = $2.25
        # Freedom Unlimited: 1.5% base = $1.125
        assert ranked[0]['card_id'] == 'gold'
        assert ranked[1]['card_id'] == 'sapphire'
        assert ranked[2]['card_id'] == 'freedom'

    def test_score_and_rank_non_bonus_category(self):
        """For a non-bonus category, highest base rate wins."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer
        from src.model_pipeline.scoring.card_ranker import CardRanker

        scorer = TransactionScorer()
        ranker = CardRanker()

        portfolio = [
            {
                'card_id': 'sapphire',
                'card_name': 'Chase Sapphire Reserve',
                'reward_rates': {
                    'universal_base_rate': 1.0,
                    'category_bonuses': {'dining': 3.0, 'travel': 3.0}
                },
                'annual_fee': 550
            },
            {
                'card_id': 'double_cash',
                'card_name': 'Citi Double Cash',
                'reward_rates': {'universal_base_rate': 2.0},
                'annual_fee': 0
            },
        ]

        # Utilities — no bonus category
        transaction = {
            'amount': 150.0,
            'category': 'utilities',
            'merchant': 'Electric Co',
            'mcc_code': 4900
        }

        scored = scorer.score_portfolio(portfolio, transaction)
        ranked = ranker.rank(scored)

        # Double Cash: 2% = $3.00
        # Sapphire: 1% = $1.50
        assert ranked[0]['card_id'] == 'double_cash'