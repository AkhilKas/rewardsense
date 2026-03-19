from src.model_pipeline.llm.validators import FactualAccuracyChecker, ReadabilityScorer


def test_factual_accuracy_passes_when_claims_match_context():
    checker = FactualAccuracyChecker(min_score=0.95)
    result = checker.evaluate(
        summary="Use Amex Gold for this dining purchase.",
        rationale=["It earns 4x rewards", "Expected reward is $4.8"],
        context={
            "scoring": {
                "transaction": {"amount": 120.0},
                "best_card": {
                    "card_name": "Amex Gold",
                    "reward_rate": 4.0,
                    "reward_amount": 4.8,
                },
                "alternatives": [{"reward_rate": 2.0, "reward_amount": 2.4}],
            }
        },
    )

    assert result.passed is True
    assert result.score == 1.0


def test_factual_accuracy_fails_on_unsupported_claims():
    checker = FactualAccuracyChecker(min_score=0.95)
    result = checker.evaluate(
        summary="Use Amex Gold.",
        rationale=["It earns 10x rewards", "Expected reward is $99"],
        context={
            "scoring": {
                "transaction": {"amount": 120.0},
                "best_card": {
                    "card_name": "Amex Gold",
                    "reward_rate": 4.0,
                    "reward_amount": 4.8,
                },
                "alternatives": [{"reward_rate": 2.0, "reward_amount": 2.4}],
            }
        },
    )

    assert result.passed is False
    assert result.score == 0.0
    assert "10x" in result.unsupported_claims


def test_readability_scorer_reports_metrics():
    scorer = ReadabilityScorer(min_flesch_score=40.0, max_grade_level=12.0)
    result = scorer.evaluate(
        summary="Use Amex Gold for dining purchases.",
        rationale=["It has higher dining rewards", "You get better expected value"],
    )

    assert isinstance(result.flesch_reading_ease, float)
    assert isinstance(result.grade_level, float)
