"""
Scoring engine validation runner for RewardSense.

Runs golden test cases, computes accuracy metrics, benchmarks throughput,
and logs everything to MLflow under the 'reward-scoring' experiment.

Usage:
    python -m src.model_pipeline.scoring.scoring_validator
"""

import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from src.model_pipeline.scoring.reward_calculator import RewardCalculator
from src.model_pipeline.scoring.transaction_scorer import TransactionScorer
from src.model_pipeline.scoring.card_ranker import CardRanker
from src.model_pipeline.scoring.spending_cap_tracker import SpendingCapTracker

logger = logging.getLogger(__name__)

# ── Golden Test Dataset ──────────────────────────────────────────────
# Duplicated from test file so validator can run independently.
# Format: (test_id, card, transaction, expected_reward)

GOLDEN_CASES: List[Tuple[str, Dict, Dict, float]] = [
    # ── Base Rate Cards (1-7) ────────────────────────────────────
    (
        "base_1pct_100",
        {
            "card_id": "g01",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Starbucks",
            "mcc_code": 5812,
        },
        1.0,
    ),
    (
        "base_1.5pct_80",
        {
            "card_id": "g02",
            "reward_rates": {"universal_base_rate": 1.5},
            "annual_fee": 0,
        },
        {"amount": 80.0, "category": "gas", "merchant": "Shell", "mcc_code": 5541},
        1.2,
    ),
    (
        "base_2pct_250",
        {
            "card_id": "g03",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {
            "amount": 250.0,
            "category": "groceries",
            "merchant": "Whole Foods",
            "mcc_code": 5411,
        },
        5.0,
    ),
    (
        "base_1pct_travel_500",
        {
            "card_id": "g04",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
        },
        {"amount": 500.0, "category": "travel", "merchant": "Delta", "mcc_code": 3000},
        5.0,
    ),
    (
        "base_2pct_small_15",
        {
            "card_id": "g05",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {
            "amount": 15.0,
            "category": "online_shopping",
            "merchant": "Amazon",
            "mcc_code": 5964,
        },
        0.3,
    ),
    (
        "base_1pct_utilities",
        {
            "card_id": "g06",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
        },
        {
            "amount": 158.15,
            "category": "utilities",
            "merchant": "Electric Co",
            "mcc_code": 4900,
        },
        1.5815,
    ),
    (
        "base_1pct_entertainment",
        {
            "card_id": "g07",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
        },
        {
            "amount": 51.19,
            "category": "entertainment",
            "merchant": "Ticketmaster",
            "mcc_code": 7922,
        },
        0.5119,
    ),
    # ── Category Bonus Cards (8-17) ──────────────────────────────
    (
        "cat_3x_dining",
        {
            "card_id": "g08",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "category": "dining", "merchant": "Nobu", "mcc_code": 5812},
        3.0,
    ),
    (
        "cat_4x_dining_75",
        {
            "card_id": "g09",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 4.0, "groceries": 4.0},
            },
            "annual_fee": 250,
        },
        {
            "amount": 75.0,
            "category": "dining",
            "merchant": "Olive Garden",
            "mcc_code": 5812,
        },
        3.0,
    ),
    (
        "cat_4x_groceries",
        {
            "card_id": "g10",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 4.0, "groceries": 4.0},
            },
            "annual_fee": 250,
        },
        {
            "amount": 200.0,
            "category": "groceries",
            "merchant": "Trader Joes",
            "mcc_code": 5411,
        },
        8.0,
    ),
    (
        "cat_3x_travel_283",
        {
            "card_id": "g11",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 3.0},
            },
            "annual_fee": 550,
        },
        {
            "amount": 283.54,
            "category": "travel",
            "merchant": "United Airlines",
            "mcc_code": 3000,
        },
        8.5062,
    ),
    (
        "cat_fallback_base",
        {
            "card_id": "g12",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "category": "gas", "merchant": "BP", "mcc_code": 5541},
        1.0,
    ),
    (
        "cat_5x_travel_1000",
        {
            "card_id": "g13",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"travel": 5.0},
            },
            "annual_fee": 0,
        },
        {
            "amount": 1000.0,
            "category": "travel",
            "merchant": "Marriott",
            "mcc_code": 7011,
        },
        50.0,
    ),
    (
        "cat_2x_gas_45",
        {
            "card_id": "g14",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"gas": 2.0},
            },
            "annual_fee": 0,
        },
        {"amount": 45.0, "category": "gas", "merchant": "Exxon", "mcc_code": 5541},
        0.9,
    ),
    (
        "cat_3x_streaming",
        {
            "card_id": "g15",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"streaming": 3.0},
            },
            "annual_fee": 0,
        },
        {
            "amount": 15.99,
            "category": "streaming",
            "merchant": "Netflix",
            "mcc_code": 4899,
        },
        0.4797,
    ),
    (
        "cat_6x_groceries_150",
        {
            "card_id": "g16",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"groceries": 6.0},
            },
            "annual_fee": 95,
        },
        {
            "amount": 150.0,
            "category": "groceries",
            "merchant": "Kroger",
            "mcc_code": 5411,
        },
        9.0,
    ),
    (
        "cat_multi_picks_correct",
        {
            "card_id": "g17",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 5.0, "gas": 2.0},
            },
            "annual_fee": 0,
        },
        {"amount": 60.0, "category": "travel", "merchant": "Hilton", "mcc_code": 7011},
        3.0,
    ),
    # ── Rotating Quarterly Bonuses (18-24) ───────────────────────
    (
        "rot_q1_gas_active",
        {
            "card_id": "g18",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {
                    "Q1": {"categories": ["gas", "streaming"], "rate": 5.0}
                },
            },
            "annual_fee": 0,
        },
        {
            "amount": 40.0,
            "category": "gas",
            "merchant": "Shell",
            "mcc_code": 5541,
            "date": datetime(2025, 2, 10),
        },
        2.0,
    ),
    (
        "rot_q1_streaming_active",
        {
            "card_id": "g19",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {
                    "Q1": {"categories": ["gas", "streaming"], "rate": 5.0}
                },
            },
            "annual_fee": 0,
        },
        {
            "amount": 15.0,
            "category": "streaming",
            "merchant": "Hulu",
            "mcc_code": 4899,
            "date": datetime(2025, 3, 1),
        },
        0.75,
    ),
    (
        "rot_q1_inactive_in_q2",
        {
            "card_id": "g20",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q1": {"categories": ["gas"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 40.0,
            "category": "gas",
            "merchant": "Shell",
            "mcc_code": 5541,
            "date": datetime(2025, 5, 15),
        },
        0.4,
    ),
    (
        "rot_q3_dining_active",
        {
            "card_id": "g21",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q3": {"categories": ["dining"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Chipotle",
            "mcc_code": 5812,
            "date": datetime(2025, 8, 20),
        },
        5.0,
    ),
    (
        "rot_q4_online_active",
        {
            "card_id": "g22",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {
                    "Q4": {"categories": ["online_shopping"], "rate": 5.0}
                },
            },
            "annual_fee": 0,
        },
        {
            "amount": 300.0,
            "category": "online_shopping",
            "merchant": "Amazon",
            "mcc_code": 5964,
            "date": datetime(2025, 11, 28),
        },
        15.0,
    ),
    (
        "rot_q2_groceries_active",
        {
            "card_id": "g23",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q2": {"categories": ["groceries"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 120.0,
            "category": "groceries",
            "merchant": "Safeway",
            "mcc_code": 5411,
            "date": datetime(2025, 4, 5),
        },
        6.0,
    ),
    (
        "rot_non_matching_cat",
        {
            "card_id": "g24",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q1": {"categories": ["gas"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Diner",
            "mcc_code": 5812,
            "date": datetime(2025, 1, 15),
        },
        1.0,
    ),
    # ── Foreign Transaction Fees (25-28) ─────────────────────────
    (
        "ftf_net_negative",
        {
            "card_id": "g25",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 3.0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Foreign",
            "mcc_code": 5812,
            "is_foreign": True,
        },
        -1.0,
    ),
    (
        "ftf_no_fee_card",
        {
            "card_id": "g26",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 0.0,
        },
        {
            "amount": 200.0,
            "category": "travel",
            "merchant": "Hotel",
            "mcc_code": 7011,
            "is_foreign": True,
        },
        4.0,
    ),
    (
        "ftf_domestic_not_applied",
        {
            "card_id": "g27",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 3.0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Local Diner",
            "mcc_code": 5812,
            "is_foreign": False,
        },
        2.0,
    ),
    (
        "ftf_cancels_bonus",
        {
            "card_id": "g28",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 3.0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Paris Bistro",
            "mcc_code": 5812,
            "is_foreign": True,
        },
        0.0,
    ),
    # ── Edge Cases (29-34) ───────────────────────────────────────
    (
        "zero_amount",
        {
            "card_id": "g29",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"amount": 0.0, "category": "dining", "merchant": "Test", "mcc_code": 5812},
        0.0,
    ),
    (
        "tiny_amount_penny",
        {
            "card_id": "g30",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"amount": 0.01, "category": "dining", "merchant": "Penny", "mcc_code": 5812},
        0.0002,
    ),
    (
        "small_amount_1.50",
        {
            "card_id": "g31",
            "reward_rates": {"universal_base_rate": 1.5},
            "annual_fee": 0,
        },
        {
            "amount": 1.50,
            "category": "online_shopping",
            "merchant": "Amazon",
            "mcc_code": 5964,
        },
        0.0225,
    ),
    (
        "large_5000",
        {
            "card_id": "g32",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {
            "amount": 5000.0,
            "category": "travel",
            "merchant": "Cruise",
            "mcc_code": 4411,
        },
        100.0,
    ),
    (
        "missing_rates_fallback",
        {"card_id": "g33", "annual_fee": 0},
        {"amount": 100.0, "category": "dining", "merchant": "Test", "mcc_code": 5812},
        1.0,
    ),
    (
        "empty_rates_fallback",
        {"card_id": "g34", "reward_rates": {}, "annual_fee": 0},
        {"amount": 100.0, "category": "dining", "merchant": "Test", "mcc_code": 5812},
        1.0,
    ),
    # ── Real-World Card Approximations (35-42) ───────────────────
    (
        "csr_dining_50",
        {
            "card_id": "g35",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 3.0},
            },
            "annual_fee": 550,
        },
        {
            "amount": 50.0,
            "category": "dining",
            "merchant": "Restaurant",
            "mcc_code": 5812,
        },
        1.5,
    ),
    (
        "csr_travel_400",
        {
            "card_id": "g36",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 3.0},
            },
            "annual_fee": 550,
        },
        {
            "amount": 400.0,
            "category": "travel",
            "merchant": "American Airlines",
            "mcc_code": 3000,
        },
        12.0,
    ),
    (
        "csr_general_200",
        {
            "card_id": "g37",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 3.0},
            },
            "annual_fee": 550,
        },
        {
            "amount": 200.0,
            "category": "utilities",
            "merchant": "Electric Co",
            "mcc_code": 4900,
        },
        2.0,
    ),
    (
        "amex_gold_dining_120",
        {
            "card_id": "g38",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 4.0, "groceries": 4.0},
            },
            "annual_fee": 250,
        },
        {"amount": 120.0, "category": "dining", "merchant": "Per Se", "mcc_code": 5812},
        4.8,
    ),
    (
        "amex_gold_groceries_95",
        {
            "card_id": "g39",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 4.0, "groceries": 4.0},
            },
            "annual_fee": 250,
        },
        {
            "amount": 95.0,
            "category": "groceries",
            "merchant": "Whole Foods",
            "mcc_code": 5411,
        },
        3.8,
    ),
    (
        "venture_x_travel_300",
        {
            "card_id": "g40",
            "reward_rates": {
                "universal_base_rate": 2.0,
                "category_bonuses": {"travel": 5.0},
            },
            "annual_fee": 395,
        },
        {"amount": 300.0, "category": "travel", "merchant": "Hyatt", "mcc_code": 7011},
        15.0,
    ),
    (
        "double_cash_88",
        {
            "card_id": "g41",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"amount": 88.0, "category": "dining", "merchant": "Panera", "mcc_code": 5812},
        1.76,
    ),
    (
        "cff_rotating_q3",
        {
            "card_id": "g42",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q3": {"categories": ["gas"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 50.0,
            "category": "gas",
            "merchant": "Mobil",
            "mcc_code": 5541,
            "date": datetime(2025, 7, 4),
        },
        2.5,
    ),
    # ── Complex Combinations (43-47) ─────────────────────────────
    (
        "rotating_overrides_category",
        {
            "card_id": "g43",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
                "rotating_bonuses": {"Q1": {"categories": ["dining"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Place",
            "mcc_code": 5812,
            "date": datetime(2025, 2, 14),
        },
        5.0,
    ),
    (
        "category_when_rot_inactive",
        {
            "card_id": "g44",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
                "rotating_bonuses": {"Q1": {"categories": ["dining"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Place",
            "mcc_code": 5812,
            "date": datetime(2025, 6, 14),
        },
        3.0,
    ),
    (
        "ftf_with_travel_bonus",
        {
            "card_id": "g45",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"travel": 5.0},
            },
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 3.0,
        },
        {
            "amount": 200.0,
            "category": "travel",
            "merchant": "Tokyo Hotel",
            "mcc_code": 7011,
            "is_foreign": True,
        },
        4.0,
    ),
    (
        "no_date_no_rotating",
        {
            "card_id": "g46",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q1": {"categories": ["gas"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "category": "gas", "merchant": "Shell", "mcc_code": 5541},
        1.0,
    ),
    (
        "missing_category_in_txn",
        {
            "card_id": "g47",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "merchant": "Mystery", "mcc_code": 9999},
        1.0,
    ),
    # ── Welcome Bonus Eligibility (48-52) ────────────────────────
    # These test is_welcome_bonus_eligible — not reward amounts.
    # We encode: expected = 1.0 for eligible, 0.0 for ineligible.
    # The validator runs these through a separate eligibility check.
    (
        "wb_new_user_eligible",
        {
            "card_id": "g48",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
            "welcome_bonus": {
                "amount": 60000,
                "spend_requirement": 4000,
                "days_to_complete": 90,
                "currency": "POINTS",
            },
        },
        {
            "user_status": {
                "user_id": "u1",
                "card_tenure_days": 0,
                "total_spent_on_card": 0,
            }
        },
        1.0,
    ),  # eligible
    (
        "wb_already_received",
        {
            "card_id": "g49",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
            "welcome_bonus": {
                "amount": 60000,
                "spend_requirement": 4000,
                "days_to_complete": 90,
            },
        },
        {
            "user_status": {
                "user_id": "u1",
                "card_tenure_days": 200,
                "total_spent_on_card": 5000,
                "welcome_bonus_received": True,
            }
        },
        0.0,
    ),  # ineligible
    (
        "wb_past_time_window",
        {
            "card_id": "g50",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
            "welcome_bonus": {
                "amount": 50000,
                "spend_requirement": 3000,
                "days_to_complete": 90,
            },
        },
        {
            "user_status": {
                "user_id": "u1",
                "card_tenure_days": 120,
                "total_spent_on_card": 1000,
            }
        },
        0.0,
    ),  # past 90-day window
    (
        "wb_met_spend_not_received",
        {
            "card_id": "g51",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
            "welcome_bonus": {
                "amount": 75000,
                "spend_requirement": 5000,
                "days_to_complete": 90,
            },
        },
        {
            "user_status": {
                "user_id": "u1",
                "card_tenure_days": 60,
                "total_spent_on_card": 6000,
            }
        },
        1.0,
    ),  # met spend, within window, not received yet → eligible
    (
        "wb_no_bonus_defined",
        {
            "card_id": "g52",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
        },
        {
            "user_status": {
                "user_id": "u1",
                "card_tenure_days": 0,
                "total_spent_on_card": 0,
            }
        },
        0.0,
    ),  # no welcome_bonus field → ineligible
    # ── Statement Credits (53-56) ────────────────────────────────
    # These test calculate_reward_with_credits.
    # We encode the expected effective reward value.
    (
        "sc_dining_credit_applied",
        {
            "card_id": "g53",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 250,
            "statement_credits": {"dining": {"amount": 10.0, "frequency": "monthly"}},
        },
        {
            "amount": 50.0,
            "category": "dining",
            "merchant": "Restaurant",
            "mcc_code": 5812,
            "_test_type": "statement_credit",
        },
        3.0,
    ),  # base reward 0.5 + credit 2.5 (10/4)
    (
        "sc_no_matching_category",
        {
            "card_id": "g54",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 250,
            "statement_credits": {"dining": {"amount": 10.0, "frequency": "monthly"}},
        },
        {
            "amount": 100.0,
            "category": "gas",
            "merchant": "Shell",
            "mcc_code": 5541,
            "_test_type": "statement_credit",
        },
        1.0,
    ),  # no matching credit → just base reward
    (
        "sc_quarterly_credit",
        {
            "card_id": "g55",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
            "statement_credits": {"travel": {"amount": 50.0, "frequency": "quarterly"}},
        },
        {
            "amount": 200.0,
            "category": "travel",
            "merchant": "Airline",
            "mcc_code": 3000,
            "_test_type": "statement_credit",
        },
        6.1667,
    ),  # base 2.0 + credit 50/12 ≈ 4.1667
    (
        "sc_annual_credit",
        {
            "card_id": "g56",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 550,
            "statement_credits": {"travel": {"amount": 300.0, "frequency": "annual"}},
        },
        {
            "amount": 500.0,
            "category": "travel",
            "merchant": "Hotel",
            "mcc_code": 7011,
            "_test_type": "statement_credit",
        },
        16.25,
    ),  # base 10.0 + credit 300/48 = 6.25
    # ── Spending Cap Enforcement (57-61) ─────────────────────────
    # These test TransactionScorer with cap_tracker.
    # Format: card has category_caps, transaction includes _test_type and _pre_spent.
    (
        "cap_within_limit",
        {
            "card_id": "g57",
            "card_name": "Capped Grocery",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"groceries": 6.0},
                "category_caps": {"groceries": 6000.0},
            },
            "annual_fee": 95,
        },
        {
            "amount": 100.0,
            "category": "groceries",
            "merchant": "Kroger",
            "mcc_code": 5411,
            "_test_type": "cap_check",
            "_pre_spent": 0.0,
        },
        6.0,
    ),  # under cap → 6%
    (
        "cap_exceeded",
        {
            "card_id": "g58",
            "card_name": "Capped Grocery",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"groceries": 6.0},
                "category_caps": {"groceries": 6000.0},
            },
            "annual_fee": 95,
        },
        {
            "amount": 100.0,
            "category": "groceries",
            "merchant": "Kroger",
            "mcc_code": 5411,
            "_test_type": "cap_check",
            "_pre_spent": 6000.0,
        },
        1.0,
    ),  # over cap → falls to 1% base
    (
        "cap_nearly_full",
        {
            "card_id": "g59",
            "card_name": "Capped Grocery",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"groceries": 4.0},
                "category_caps": {"groceries": 500.0},
            },
            "annual_fee": 0,
        },
        {
            "amount": 200.0,
            "category": "groceries",
            "merchant": "Safeway",
            "mcc_code": 5411,
            "_test_type": "cap_check",
            "_pre_spent": 400.0,
        },
        2.0,
    ),  # remaining $100 < txn $200 → base rate 1%
    (
        "cap_other_category_unaffected",
        {
            "card_id": "g60",
            "card_name": "Multi Bonus Capped",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"groceries": 4.0, "dining": 3.0},
                "category_caps": {"groceries": 6000.0},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Chipotle",
            "mcc_code": 5812,
            "_test_type": "cap_check",
            "_pre_spent_category": "groceries",
            "_pre_spent": 6000.0,
        },
        3.0,
    ),  # groceries capped, but dining has no cap → 3%
    (
        "cap_no_caps_defined",
        {
            "card_id": "g61",
            "card_name": "No Cap Card",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"groceries": 4.0},
            },
            "annual_fee": 0,
        },
        {
            "amount": 100.0,
            "category": "groceries",
            "merchant": "Store",
            "mcc_code": 5411,
            "_test_type": "cap_check",
            "_pre_spent": 999999.0,
        },
        4.0,
    ),  # no category_caps → bonus always applies
]

TOLERANCE = 1e-4


class ScoringValidator:
    """
    Validates the scoring engine against golden test cases and benchmarks.
    Logs all results to MLflow.
    """

    def __init__(self):
        self.calculator = RewardCalculator()
        self.scorer = TransactionScorer()
        self.ranker = CardRanker()

    def run_golden_tests(self) -> Dict[str, Any]:
        """
        Run all golden test cases and compute accuracy metrics.

        Handles three special test types via _test_type field:
        - "statement_credit": uses calculate_reward_with_credits
        - "cap_check": uses TransactionScorer with SpendingCapTracker
        - welcome bonus: detected by presence of "user_status" key in txn
        - default: uses calculate_reward

        Returns:
            Dict with total, passed, failed, accuracy, and per-case details.
        """
        results = []
        passed = 0
        failed = 0

        # Statement credit calculator
        credit_calculator = RewardCalculator(include_statement_credits=True)

        for test_id, card, txn, expected in GOLDEN_CASES:
            test_type = txn.get("_test_type", "")

            if "user_status" in txn:
                # Welcome bonus eligibility test
                eligible = self.calculator.is_welcome_bonus_eligible(
                    card, txn["user_status"]
                )
                actual = 1.0 if eligible else 0.0

            elif test_type == "statement_credit":
                actual = credit_calculator.calculate_reward_with_credits(card, txn)

            elif test_type == "cap_check":
                tracker = SpendingCapTracker(user_id="validator_user")
                pre_spent = txn.get("_pre_spent", 0.0)
                pre_spent_category = txn.get(
                    "_pre_spent_category", txn.get("category", "general")
                )
                if pre_spent > 0:
                    tracker.record_transaction(
                        card.get("card_id", ""), pre_spent_category, pre_spent
                    )
                scorer = TransactionScorer(cap_tracker=tracker)
                result = scorer.score_card(card, txn)
                actual = result["reward_amount"]

            else:
                actual = self.calculator.calculate_reward(card, txn)

            is_pass = abs(actual - expected) < TOLERANCE

            if is_pass:
                passed += 1
            else:
                failed += 1
                logger.warning(
                    f"FAIL [{test_id}]: expected={expected:.4f}, actual={actual:.4f}"
                )

            results.append(
                {
                    "test_id": test_id,
                    "expected": expected,
                    "actual": round(actual, 6),
                    "passed": is_pass,
                }
            )

        total = len(GOLDEN_CASES)
        accuracy = passed / total if total > 0 else 0.0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "accuracy": accuracy,
            "details": results,
        }

    def run_throughput_benchmark(
        self, n_transactions: int = 5000, n_cards: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark scoring throughput.

        Args:
            n_transactions: Number of transactions to score
            n_cards: Number of cards in test portfolio

        Returns:
            Dict with single_card_throughput, batch_throughput, latency_per_txn.
        """
        categories = ["dining", "travel", "gas", "groceries", "utilities"]

        card = {
            "card_id": "bench_card",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 5.0},
            },
            "annual_fee": 0,
        }

        transactions = [
            {
                "amount": 50.0 + i,
                "category": categories[i % len(categories)],
                "merchant": f"M_{i}",
                "mcc_code": 5812,
            }
            for i in range(n_transactions)
        ]

        # Single card throughput
        start = time.time()
        for txn in transactions:
            self.calculator.calculate_reward(card, txn)
        single_elapsed = time.time() - start
        single_throughput = n_transactions / single_elapsed

        # Batch throughput
        portfolio = [
            {
                "card_id": f"card_{i}",
                "card_name": f"Card {i}",
                "reward_rates": {
                    "universal_base_rate": 1.0 + i * 0.5,
                    "category_bonuses": {"dining": 2.0 + i},
                },
                "annual_fee": i * 100,
            }
            for i in range(n_cards)
        ]

        start = time.time()
        self.scorer.score_batch(portfolio, transactions)
        batch_elapsed = time.time() - start
        batch_throughput = n_transactions / batch_elapsed

        return {
            "n_transactions": n_transactions,
            "n_cards": n_cards,
            "single_card_throughput": round(single_throughput, 1),
            "batch_throughput": round(batch_throughput, 1),
            "single_latency_ms": round((single_elapsed / n_transactions) * 1000, 4),
            "batch_latency_ms": round((batch_elapsed / n_transactions) * 1000, 4),
        }

    def validate_and_log(self, log_to_mlflow: bool = True) -> Dict[str, Any]:
        """
        Run full validation and optionally log to MLflow.

        Args:
            log_to_mlflow: If True, log results to MLflow reward-scoring experiment

        Returns:
            Combined validation report dict
        """
        logger.info("Starting scoring engine validation...")

        golden_results = self.run_golden_tests()
        benchmark_results = self.run_throughput_benchmark()

        report = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "golden_tests": golden_results,
            "benchmarks": benchmark_results,
        }

        logger.info(
            f"Golden tests: {golden_results['passed']}/{golden_results['total']} passed "
            f"({golden_results['accuracy']:.1%} accuracy)"
        )
        logger.info(
            f"Throughput: {benchmark_results['single_card_throughput']:.0f} txn/s (single), "
            f"{benchmark_results['batch_throughput']:.0f} txn/s (batch)"
        )

        if log_to_mlflow:
            self._log_to_mlflow(report)

        return report

    def _log_to_mlflow(self, report: Dict[str, Any]) -> None:
        """Log validation report to MLflow reward-scoring experiment."""
        try:
            import mlflow

            mlflow.set_experiment("reward-scoring")

            with mlflow.start_run(run_name="scoring-validation"):
                # Golden test metrics
                mlflow.log_metric("golden_accuracy", report["golden_tests"]["accuracy"])
                mlflow.log_metric("golden_passed", report["golden_tests"]["passed"])
                mlflow.log_metric("golden_failed", report["golden_tests"]["failed"])
                mlflow.log_metric("golden_total", report["golden_tests"]["total"])

                # Benchmark metrics
                mlflow.log_metric(
                    "throughput_single", report["benchmarks"]["single_card_throughput"]
                )
                mlflow.log_metric(
                    "throughput_batch", report["benchmarks"]["batch_throughput"]
                )
                mlflow.log_metric(
                    "latency_single_ms", report["benchmarks"]["single_latency_ms"]
                )
                mlflow.log_metric(
                    "latency_batch_ms", report["benchmarks"]["batch_latency_ms"]
                )

                # Params
                mlflow.log_param("n_golden_cases", report["golden_tests"]["total"])
                mlflow.log_param(
                    "n_bench_transactions", report["benchmarks"]["n_transactions"]
                )
                mlflow.log_param("n_bench_cards", report["benchmarks"]["n_cards"])

                # Full report as artifact
                report_json = json.dumps(report, indent=2, default=str)
                with open("/tmp/scoring_validation_report.json", "w") as f:
                    f.write(report_json)
                mlflow.log_artifact("/tmp/scoring_validation_report.json")

                # Failed cases as separate artifact if any
                failed_cases = [
                    d for d in report["golden_tests"]["details"] if not d["passed"]
                ]
                if failed_cases:
                    with open("/tmp/scoring_failed_cases.json", "w") as f:
                        json.dump(failed_cases, f, indent=2)
                    mlflow.log_artifact("/tmp/scoring_failed_cases.json")

            logger.info(
                "Validation results logged to MLflow 'reward-scoring' experiment"
            )

        except ImportError:
            logger.warning("MLflow not installed, skipping logging")
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")


# ── CLI entrypoint ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validator = ScoringValidator()
    report = validator.validate_and_log(log_to_mlflow=True)
    print(f"\nValidation complete: {report['golden_tests']['accuracy']:.1%} accuracy")
    print(f"Throughput: {report['benchmarks']['batch_throughput']:.0f} txn/sec (batch)")
