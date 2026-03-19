import pytest


@pytest.fixture
def scoring_output_fixture():
    return {
        "transaction": {"amount": 100.0, "category": "dining", "merchant": "Chipotle"},
        "best_card": {
            "card_id": "amex_gold",
            "card_name": "Amex Gold",
            "reward_rate": 4.0,
            "reward_amount": 4.0,
        },
        "alternatives": [
            {
                "card_id": "citi_double",
                "card_name": "Citi Double Cash",
                "reward_rate": 2.0,
                "reward_amount": 2.0,
            }
        ],
    }
