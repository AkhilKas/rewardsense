import json
import random
import time
import uuid

from src.model_pipeline.llm import ExplanationGenerator, ExplanationLatencyBenchmark
from src.model_pipeline.llm.prompt_builder import ExplanationType
from src.model_pipeline.tracking import RewardSenseTracker

class MockVertexGeminiClient:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "gemini-2.5-flash")

    def generate(self, *args, **kwargs) -> str:
        # Simulate network latency
        time.sleep(random.uniform(0.1, 0.4))
        
        # Return a mock valid explanation response depending on the explanation type
        return json.dumps({
            "explanation": "Because you spend heavily on Dining, the Amex Gold is highly recommended as it offers 4x points on restaurants, giving you the best return.",
            "key_factors": [
                "Dining category matches user spend",
                "High 4x reward rate",
                "Sign-up bonus offsets annual fee"
            ],
            "confidence_score": 0.95
        })

def sample_scoring_output() -> dict:
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

def main():
    print("Initializing mock LLM client and Generator...")
    mock_client = MockVertexGeminiClient(model="mock-gemini-2.5-flash")
    generator = ExplanationGenerator(llm_client=mock_client, model_name="mock-gemini-2.5-flash")
    
    # Initialize Tracker pointing to local MLflow Docker
    tracker = RewardSenseTracker(
        experiment="llm-explainability",
        tracking_uri="http://localhost:5001"
    )
    # Patch log_dict to avoid local Mac /mlflow/ read-only crash
    tracker.log_dict = lambda d, path: print(f"Mocked artifact save: {path}")
    
    print("Running ExplanationLatencyBenchmark...")
    bench = ExplanationLatencyBenchmark(
        generator=generator,
        latency_budget_ms=2000.0,
        tracker=tracker,
    )
    
    result = bench.run(
        scoring_output=sample_scoring_output(),
        personalization_signals={"user_segment": "foodie"},
        n_requests=10,
        explanation_type=ExplanationType.SINGLE_TRANSACTION,
    )
    
    print("\nBenchmark completed. Summary:")
    print(json.dumps(result.__dict__, indent=2))
    print("\nCheck http://localhost:5001 in your browser to see the experiment!")

if __name__ == "__main__":
    main()
