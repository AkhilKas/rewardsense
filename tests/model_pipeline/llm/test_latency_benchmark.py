from contextlib import contextmanager

from src.model_pipeline.llm.explanation_generator import ExplanationGenerator
from src.model_pipeline.llm.latency_benchmark import ExplanationLatencyBenchmark


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response

    def generate(self, system_message: str, user_message: str, **kwargs) -> str:
        return self.response


class _FakeTracker:
    def __init__(self):
        self.metrics = None
        self.params = None
        self.payload = None

    @contextmanager
    def start_run(self, run_name=None, tags=None, nested=False):
        yield object()

    def log_metrics(self, metrics, step=None):
        self.metrics = metrics

    def log_params(self, params):
        self.params = params

    def log_dict(self, payload, filename):
        self.payload = (payload, filename)


def _context():
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


def test_latency_benchmark_passes_and_logs():
    raw = '{"summary":"Use Amex Gold.","rationale":["4x dining","best expected value"],"confidence":0.95}'
    generator = ExplanationGenerator(llm_client=_FakeLLMClient(raw))
    tracker = _FakeTracker()

    bench = ExplanationLatencyBenchmark(
        generator=generator,
        latency_budget_ms=2000.0,
        tracker=tracker,
    )
    result = bench.run(
        scoring_output=_context(),
        personalization_signals={"user_segment": "foodie"},
        n_requests=5,
    )

    assert result.n_requests == 5
    assert result.passed is True
    assert tracker.metrics is not None
    assert "latency_p95_ms" in tracker.metrics
    assert tracker.payload[1] == "llm_latency_single_transaction_recommendation.json"
