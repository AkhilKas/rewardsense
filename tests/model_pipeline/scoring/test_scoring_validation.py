"""
TDD Tests for Scoring Engine Validation & Regression - Story 2.3

Golden test suite with 50+ manually verified scenarios,
regression tests, and performance benchmarks.
"""

import pytest
import time
from datetime import datetime

# ── Golden Test Dataset ──────────────────────────────────────────────
# Each case: (test_id, card, transaction, expected_reward)
# Manually verified expected values serve as the ground truth.

GOLDEN_TEST_CASES = [
    # ── Base Rate Cards ──────────────────────────────────────────
    (
        "base_1pct_dining_100",
        {
            "card_id": "g01",
            "card_name": "Simple 1%",
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
        "base_1.5pct_gas_80",
        {
            "card_id": "g02",
            "card_name": "Cash Back 1.5%",
            "reward_rates": {"universal_base_rate": 1.5},
            "annual_fee": 0,
        },
        {"amount": 80.0, "category": "gas", "merchant": "Shell", "mcc_code": 5541},
        1.2,
    ),
    (
        "base_2pct_groceries_250",
        {
            "card_id": "g03",
            "card_name": "Double Cash",
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
            "card_name": "Basic Travel",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
        },
        {"amount": 500.0, "category": "travel", "merchant": "Delta", "mcc_code": 3000},
        5.0,
    ),
    (
        "base_2pct_online_15",
        {
            "card_id": "g05",
            "card_name": "Flat 2%",
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
        "base_1pct_utilities_158",
        {
            "card_id": "g06",
            "card_name": "Simple 1%",
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
        "base_1pct_entertainment_51",
        {
            "card_id": "g07",
            "card_name": "Simple 1%",
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
    # ── Category Bonus Cards ─────────────────────────────────────
    (
        "cat_3x_dining_100",
        {
            "card_id": "g08",
            "card_name": "Dining 3x",
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
            "card_name": "Amex Gold Style",
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
        "cat_4x_groceries_200",
        {
            "card_id": "g10",
            "card_name": "Amex Gold Style",
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
            "card_name": "Sapphire Reserve Style",
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
        "cat_bonus_fallback_to_base",
        {
            "card_id": "g12",
            "card_name": "Dining Only Bonus",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "category": "gas", "merchant": "BP", "mcc_code": 5541},
        1.0,  # falls back to base
    ),
    (
        "cat_5x_travel_1000",
        {
            "card_id": "g13",
            "card_name": "Travel 5x",
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
            "card_name": "Gas Card",
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
        "cat_3x_streaming_15",
        {
            "card_id": "g15",
            "card_name": "Streaming Card",
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
            "card_name": "Blue Cash Preferred Style",
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
        "cat_multiple_bonuses_uses_correct",
        {
            "card_id": "g17",
            "card_name": "Multi Bonus",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 5.0, "gas": 2.0},
            },
            "annual_fee": 0,
        },
        {"amount": 60.0, "category": "travel", "merchant": "Hilton", "mcc_code": 7011},
        3.0,  # picks travel 5%
    ),
    # ── Rotating Quarterly Bonuses ───────────────────────────────
    (
        "rot_q1_gas_active",
        {
            "card_id": "g18",
            "card_name": "Freedom Flex Style",
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
            "card_name": "Freedom Flex Style",
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
        "rot_q1_gas_inactive_q2",
        {
            "card_id": "g20",
            "card_name": "Freedom Flex Style",
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
        0.4,  # Q2, falls back to 1% base
    ),
    (
        "rot_q3_dining_active",
        {
            "card_id": "g21",
            "card_name": "Discover Style",
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
        "rot_q4_online_shopping_active",
        {
            "card_id": "g22",
            "card_name": "Discover Style",
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
            "card_name": "Rotating Q2",
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
        "rot_non_matching_category",
        {
            "card_id": "g24",
            "card_name": "Rotating Q1 Gas",
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
        1.0,  # Q1 active but category doesn't match
    ),
    # ── Foreign Transaction Fees ─────────────────────────────────
    (
        "ftf_3pct_on_2pct_card",
        {
            "card_id": "g25",
            "card_name": "FTF Card",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 3.0,
        },
        {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Foreign Restaurant",
            "mcc_code": 5812,
            "is_foreign": True,
        },
        -1.0,  # 2% reward - 3% fee = -1%
    ),
    (
        "ftf_0pct_no_fee_card",
        {
            "card_id": "g26",
            "card_name": "No FTF Card",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 0.0,
        },
        {
            "amount": 200.0,
            "category": "travel",
            "merchant": "Foreign Hotel",
            "mcc_code": 7011,
            "is_foreign": True,
        },
        4.0,  # 2% reward, no fee
    ),
    (
        "ftf_domestic_not_applied",
        {
            "card_id": "g27",
            "card_name": "FTF Card",
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
        2.0,  # domestic, fee not applied
    ),
    (
        "ftf_3pct_on_3x_dining",
        {
            "card_id": "g28",
            "card_name": "Dining + FTF",
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
        0.0,  # 3% reward - 3% fee = 0
    ),
    # ── Zero & Small Amounts ─────────────────────────────────────
    (
        "zero_amount",
        {
            "card_id": "g29",
            "card_name": "Any Card",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"amount": 0.0, "category": "dining", "merchant": "Test", "mcc_code": 5812},
        0.0,
    ),
    (
        "tiny_amount_0.01",
        {
            "card_id": "g30",
            "card_name": "2% Card",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {
            "amount": 0.01,
            "category": "dining",
            "merchant": "Penny Store",
            "mcc_code": 5812,
        },
        0.0002,
    ),
    (
        "small_amount_1.50",
        {
            "card_id": "g31",
            "card_name": "1.5% Card",
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
    # ── Large Amounts ────────────────────────────────────────────
    (
        "large_amount_5000",
        {
            "card_id": "g32",
            "card_name": "2% Card",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {
            "amount": 5000.0,
            "category": "travel",
            "merchant": "Cruise Line",
            "mcc_code": 4411,
        },
        100.0,
    ),
    (
        "large_amount_10000_with_bonus",
        {
            "card_id": "g33",
            "card_name": "Travel 5x",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"travel": 5.0},
            },
            "annual_fee": 0,
        },
        {
            "amount": 10000.0,
            "category": "travel",
            "merchant": "Private Jet",
            "mcc_code": 4511,
        },
        500.0,
    ),
    # ── Missing / Malformed Data (Graceful Fallback) ─────────────
    (
        "missing_reward_rates",
        {"card_id": "g34", "card_name": "Broken Card", "annual_fee": 0},
        {"amount": 100.0, "category": "dining", "merchant": "Test", "mcc_code": 5812},
        1.0,  # falls back to default_reward_rate=1.0
    ),
    (
        "empty_reward_rates",
        {
            "card_id": "g35",
            "card_name": "Empty Rates",
            "reward_rates": {},
            "annual_fee": 0,
        },
        {"amount": 100.0, "category": "dining", "merchant": "Test", "mcc_code": 5812},
        1.0,  # no universal_base_rate → uses default
    ),
    (
        "missing_category_in_transaction",
        {
            "card_id": "g36",
            "card_name": "Bonus Card",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "merchant": "Mystery Shop", "mcc_code": 9999},
        1.0,  # no category key → no bonus match → base rate
    ),
    (
        "missing_amount_defaults_zero",
        {
            "card_id": "g37",
            "card_name": "Card",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"category": "dining", "merchant": "Test", "mcc_code": 5812},
        0.0,  # missing amount → 0
    ),
    # ── Real-World Card Approximations ───────────────────────────
    # Chase Sapphire Reserve: 3x dining, 3x travel, 1x base
    (
        "csr_dining_50",
        {
            "card_id": "g38",
            "card_name": "Chase Sapphire Reserve",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 3.0},
            },
            "annual_fee": 550,
        },
        {
            "amount": 50.0,
            "category": "dining",
            "merchant": "Sushi Nakazawa",
            "mcc_code": 5812,
        },
        1.5,
    ),
    (
        "csr_travel_400",
        {
            "card_id": "g39",
            "card_name": "Chase Sapphire Reserve",
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
            "card_id": "g40",
            "card_name": "Chase Sapphire Reserve",
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
    # Amex Gold: 4x dining, 4x groceries, 1x base
    (
        "amex_gold_dining_120",
        {
            "card_id": "g41",
            "card_name": "Amex Gold",
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
            "card_id": "g42",
            "card_name": "Amex Gold",
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
    # Capital One Venture X: 2x on everything, 5x travel, 10x hotels via portal
    (
        "venture_x_travel_300",
        {
            "card_id": "g43",
            "card_name": "Capital One Venture X",
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
        "venture_x_general_100",
        {
            "card_id": "g44",
            "card_name": "Capital One Venture X",
            "reward_rates": {
                "universal_base_rate": 2.0,
                "category_bonuses": {"travel": 5.0},
            },
            "annual_fee": 395,
        },
        {"amount": 100.0, "category": "gas", "merchant": "Shell", "mcc_code": 5541},
        2.0,
    ),
    # Chase Freedom Flex: 1% base, 5% rotating quarterly
    (
        "cff_rotating_q3_gas",
        {
            "card_id": "g45",
            "card_name": "Chase Freedom Flex",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {
                    "Q3": {"categories": ["gas", "ev_charging"], "rate": 5.0}
                },
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
    (
        "cff_non_rotating_q3_dining",
        {
            "card_id": "g46",
            "card_name": "Chase Freedom Flex",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q3": {"categories": ["gas"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {
            "amount": 50.0,
            "category": "dining",
            "merchant": "Pizza Hut",
            "mcc_code": 5812,
            "date": datetime(2025, 7, 4),
        },
        0.5,  # Q3 active but dining not in rotating list → base
    ),
    # Citi Double Cash: 2% on everything
    (
        "double_cash_dining_88",
        {
            "card_id": "g47",
            "card_name": "Citi Double Cash",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"amount": 88.0, "category": "dining", "merchant": "Panera", "mcc_code": 5812},
        1.76,
    ),
    (
        "double_cash_gas_55",
        {
            "card_id": "g48",
            "card_name": "Citi Double Cash",
            "reward_rates": {"universal_base_rate": 2.0},
            "annual_fee": 0,
        },
        {"amount": 55.0, "category": "gas", "merchant": "Shell", "mcc_code": 5541},
        1.1,
    ),
    # ── Combination: Rotating + Category + Foreign ───────────────
    (
        "rotating_overrides_category_bonus",
        {
            "card_id": "g49",
            "card_name": "Complex Card",
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
            "merchant": "Fancy Place",
            "mcc_code": 5812,
            "date": datetime(2025, 2, 14),
        },
        5.0,  # rotating (5%) takes priority over category (3%)
    ),
    (
        "category_used_when_rotating_inactive",
        {
            "card_id": "g50",
            "card_name": "Complex Card",
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
            "merchant": "Fancy Place",
            "mcc_code": 5812,
            "date": datetime(2025, 6, 14),
        },
        3.0,  # Q2, rotating inactive → falls to category 3%
    ),
    (
        "foreign_txn_with_category_bonus",
        {
            "card_id": "g51",
            "card_name": "Travel + FTF",
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
        4.0,  # 5% ($10) - 3% ($6) = $4
    ),
    (
        "no_date_no_rotating",
        {
            "card_id": "g52",
            "card_name": "Rotating Card",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "rotating_bonuses": {"Q1": {"categories": ["gas"], "rate": 5.0}},
            },
            "annual_fee": 0,
        },
        {"amount": 100.0, "category": "gas", "merchant": "Shell", "mcc_code": 5541},
        1.0,  # no date → can't resolve quarter → base rate
    ),
]


# ── Parameterized Golden Tests ───────────────────────────────────────


class TestGoldenTestSuite:
    """Parameterized golden tests: 50+ manually verified reward calculations."""

    @pytest.mark.parametrize(
        "test_id, card, transaction, expected_reward",
        GOLDEN_TEST_CASES,
        ids=[case[0] for case in GOLDEN_TEST_CASES],
    )
    def test_golden_reward_calculation(
        self, test_id, card, transaction, expected_reward
    ):
        """Verify reward calculation matches golden expected value."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        actual = calculator.calculate_reward(card, transaction)

        assert actual == pytest.approx(
            expected_reward, abs=1e-4
        ), f"[{test_id}] Expected ${expected_reward:.4f}, got ${actual:.4f}"


# ── Regression Tests ─────────────────────────────────────────────────


class TestScoringRegression:
    """Regression tests to catch unintended scoring changes."""

    def test_category_bonus_does_not_stack_with_base(self):
        """Category bonus REPLACES base rate, not added on top."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator

        calculator = RewardCalculator()

        card = {
            "card_id": "reg_01",
            "reward_rates": {
                "universal_base_rate": 2.0,
                "category_bonuses": {"dining": 3.0},
            },
            "annual_fee": 0,
        }
        txn = {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Test",
            "mcc_code": 5812,
        }

        reward = calculator.calculate_reward(card, txn)
        # Should be 3.0 (category rate), NOT 5.0 (base + category)
        assert reward == 3.0

    def test_rotating_bonus_does_not_stack_with_category(self):
        """Rotating bonus takes priority, does not stack with category bonus."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator

        calculator = RewardCalculator()

        card = {
            "card_id": "reg_02",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0},
                "rotating_bonuses": {"Q1": {"categories": ["dining"], "rate": 5.0}},
            },
            "annual_fee": 0,
        }
        txn = {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Test",
            "mcc_code": 5812,
            "date": datetime(2025, 1, 15),
        }

        reward = calculator.calculate_reward(card, txn)
        # Should be 5.0 (rotating), NOT 8.0 (rotating + category)
        assert reward == 5.0

    def test_foreign_fee_applied_after_reward(self):
        """Foreign fee is subtracted from reward, not from transaction amount."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator

        calculator = RewardCalculator()

        card = {
            "card_id": "reg_03",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 0,
            "foreign_transaction_fee_pct": 3.0,
        }
        txn = {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Foreign",
            "mcc_code": 5812,
            "is_foreign": True,
        }

        reward = calculator.calculate_reward(card, txn)
        # 1% reward ($1) - 3% fee ($3) = -$2
        assert reward == -2.0

    def test_annual_fee_amortization_reduces_reward(self):
        """Amortized annual fee makes reward less than base calculation."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator

        calc_no_fee = RewardCalculator(amortize_annual_fee=False)
        calc_with_fee = RewardCalculator(amortize_annual_fee=True)

        card = {
            "card_id": "reg_04",
            "reward_rates": {"universal_base_rate": 1.0},
            "annual_fee": 550,
        }
        txn = {
            "amount": 100.0,
            "category": "dining",
            "merchant": "Test",
            "mcc_code": 5812,
        }

        reward_no_fee = calc_no_fee.calculate_reward(card, txn)
        reward_with_fee = calc_with_fee.calculate_reward(card, txn)

        assert reward_with_fee < reward_no_fee

    def test_scorer_ranker_ordering_is_stable(self):
        """Same input always produces same ranking (determinism check)."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer
        from src.model_pipeline.scoring.card_ranker import CardRanker

        scorer = TransactionScorer()
        ranker = CardRanker()

        portfolio = [
            {
                "card_id": "c1",
                "card_name": "C1",
                "reward_rates": {"universal_base_rate": 2.0},
                "annual_fee": 0,
            },
            {
                "card_id": "c2",
                "card_name": "C2",
                "reward_rates": {"universal_base_rate": 2.0},
                "annual_fee": 0,
            },
            {
                "card_id": "c3",
                "card_name": "C3",
                "reward_rates": {"universal_base_rate": 2.0},
                "annual_fee": 0,
            },
        ]
        txn = {
            "amount": 100.0,
            "category": "general",
            "merchant": "Store",
            "mcc_code": 9999,
        }

        # Run 10 times — should always be the same order
        first_run = [
            c["card_id"] for c in ranker.rank(scorer.score_portfolio(portfolio, txn))
        ]
        for _ in range(10):
            run = [
                c["card_id"]
                for c in ranker.rank(scorer.score_portfolio(portfolio, txn))
            ]
            assert run == first_run


# ── Performance Benchmark ────────────────────────────────────────────


class TestScoringPerformance:
    """Performance benchmarks: >1000 transactions/second."""

    def test_throughput_single_card(self):
        """Single card scoring throughput > 1000 txn/sec."""
        from src.model_pipeline.scoring.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        card = {
            "card_id": "perf_01",
            "reward_rates": {
                "universal_base_rate": 1.0,
                "category_bonuses": {"dining": 3.0, "travel": 5.0},
            },
            "annual_fee": 0,
        }

        categories = ["dining", "travel", "gas", "groceries", "utilities"]
        transactions = [
            {
                "amount": 50.0 + i,
                "category": categories[i % len(categories)],
                "merchant": f"Merchant_{i}",
                "mcc_code": 5812,
            }
            for i in range(2000)
        ]

        start = time.time()
        for txn in transactions:
            calculator.calculate_reward(card, txn)
        elapsed = time.time() - start

        throughput = len(transactions) / elapsed
        assert throughput > 1000, f"Throughput {throughput:.0f} txn/sec < 1000 required"

    def test_throughput_batch_scoring(self):
        """Batch scoring 5 cards x 1000 transactions > 1000 txn/sec."""
        from src.model_pipeline.scoring.transaction_scorer import TransactionScorer

        scorer = TransactionScorer()

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
            for i in range(5)
        ]

        categories = ["dining", "travel", "gas", "groceries", "utilities"]
        transactions = [
            {
                "amount": 25.0 + i,
                "category": categories[i % len(categories)],
                "merchant": f"Merchant_{i}",
                "mcc_code": 5812,
            }
            for i in range(1000)
        ]

        start = time.time()
        scorer.score_batch(portfolio, transactions)
        elapsed = time.time() - start

        throughput = len(transactions) / elapsed
        assert (
            throughput > 1000
        ), f"Batch throughput {throughput:.0f} txn/sec < 1000 required"
