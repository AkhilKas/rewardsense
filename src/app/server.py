"""
Serving layer for RewardSense recommendations + LLM explanations.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
import logging
import os
from typing import Any, Dict, Generator, List, Optional

from dotenv import load_dotenv

load_dotenv()  # must run before any env-dependent imports

import pandas as pd  # noqa: E402

from src.model_pipeline.llm import ExplanationGenerator, ExplanationType  # noqa: E402
from src.model_pipeline.llm.validators import (  # noqa: E402
    FactualAccuracyChecker,
    ReadabilityScorer,
)
from src.model_pipeline.personalization.personalized_scorer import (  # noqa: E402
    PersonalizedScorer,
)
from src.model_pipeline.tracking import RewardSenseTracker  # noqa: E402
from src.app.transactions.router import router as transactions_router  # noqa: E402

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = dict  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class RewardSenseService:
    """Runtime service for scoring and optional explanation generation."""

    def __init__(
        self,
        scorer: Optional[PersonalizedScorer] = None,
        explanation_generator: Optional[ExplanationGenerator] = None,
        enable_llm_explanations: Optional[bool] = None,
        tracker: Optional[RewardSenseTracker] = None,
        factual_checker: Optional[FactualAccuracyChecker] = None,
        readability_scorer: Optional[ReadabilityScorer] = None,
    ) -> None:
        self.scorer = scorer or PersonalizedScorer()
        self.explanation_generator = explanation_generator
        self.enable_llm_explanations = (
            _env_flag("ENABLE_LLM_EXPLANATIONS", default=False)
            if enable_llm_explanations is None
            else enable_llm_explanations
        )
        self.tracker = tracker
        self.factual_checker = factual_checker or FactualAccuracyChecker(min_score=0.95)
        self.readability_scorer = readability_scorer or ReadabilityScorer()

    def recommend(
        self,
        portfolio: List[Dict[str, Any]],
        transaction: Dict[str, Any],
        user_features: Optional[pd.DataFrame] = None,
        personalization_signals: Optional[Dict[str, Any]] = None,
        explanation_type: ExplanationType = ExplanationType.SINGLE_TRANSACTION,
    ) -> Dict[str, Any]:
        """Return card recommendation with optional LLM explanation payload."""
        scoring_result = self.scorer.score(
            portfolio=portfolio,
            transaction=transaction,
            user_features=user_features,
        )

        response: Dict[str, Any] = {
            "recommendation": scoring_result,
            "explanation": None,
            "llm_explanations_enabled": self.enable_llm_explanations,
        }

        if not self.enable_llm_explanations or self.explanation_generator is None:
            return response

        context = self._build_scoring_context(scoring_result, transaction)
        personalization = personalization_signals or {}

        generated = self.explanation_generator.generate(
            explanation_type=explanation_type,
            scoring_output=context,
            personalization_signals=personalization,
        )

        factual = self.factual_checker.evaluate(
            summary=generated.summary,
            rationale=generated.rationale,
            context={"scoring": context},
        )
        readability = self.readability_scorer.evaluate(
            summary=generated.summary,
            rationale=generated.rationale,
        )

        explanation = {
            "summary": generated.summary,
            "pros": generated.pros,
            "cons": generated.cons,
            "best_for": generated.best_for,
            "rationale": generated.rationale,
            "confidence": generated.confidence,
            "disclaimers": generated.disclaimers,
            "used_fallback": generated.used_fallback,
            "fallback_reason": generated.fallback_reason,
            "latency_ms": generated.latency_ms,
            "quality_checks": generated.quality_checks,
            "prompt_hash": generated.prompt_hash,
            "model_name": generated.model_name,
            "factual_accuracy": {
                "score": factual.score,
                "passed": factual.passed,
                "total_claims": factual.total_claims,
                "supported_claims": factual.supported_claims,
                "unsupported_claims": factual.unsupported_claims,
            },
            "readability": {
                "flesch_reading_ease": readability.flesch_reading_ease,
                "grade_level": readability.grade_level,
                "passed": readability.passed,
            },
        }
        response["explanation"] = explanation

        self._log_llm_metrics(
            explanation=explanation,
            explanation_type=explanation_type.value,
            scoring_context=context,
            recommendation=scoring_result,
        )

        return response

    def _build_scoring_context(
        self,
        scoring_result: Dict[str, Any],
        transaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        ranked = scoring_result.get("ranked", [])
        best = ranked[0] if ranked else {}
        alternatives = ranked[1:3] if len(ranked) > 1 else []
        return {
            "transaction": transaction,
            "best_card": best,
            "alternatives": alternatives,
        }

    def _log_llm_metrics(
        self,
        explanation: Dict[str, Any],
        explanation_type: str,
        scoring_context: Dict[str, Any],
        recommendation: Dict[str, Any],
    ) -> None:
        if self.tracker is None:
            return

        quality = explanation.get("quality_checks", {})
        factual = explanation.get("factual_accuracy", {})
        readability = explanation.get("readability", {})

        metrics = {
            "factual_accuracy_score": float(factual.get("score", 0.0)),
            "factual_accuracy_passed": 1.0 if factual.get("passed", False) else 0.0,
            "readability_flesch": float(readability.get("flesch_reading_ease", 0.0)),
            "readability_grade_level": float(readability.get("grade_level", 99.0)),
            "readability_passed": 1.0 if readability.get("passed", False) else 0.0,
            "quality_length_ok": 1.0 if quality.get("length_ok", False) else 0.0,
            "quality_relevance_ok": 1.0 if quality.get("relevance_ok", False) else 0.0,
            "quality_hallucination_guard_ok": (
                1.0 if quality.get("hallucination_guard_ok", False) else 0.0
            ),
            "explanation_used_fallback": (
                1.0 if explanation.get("used_fallback", False) else 0.0
            ),
            "explanation_latency_ms": float(explanation.get("latency_ms", 0.0)),
        }

        with self._tracking_run(run_name=f"llm-explain-{explanation_type}"):
            self.tracker.log_metrics(metrics)
            self.tracker.log_params({"explanation_type": explanation_type})
            self.tracker.log_dict(
                {
                    "explanation_type": explanation_type,
                    "recommendation": recommendation,
                    "scoring_context": scoring_context,
                    "explanation": explanation,
                },
                f"llm_explanation_{explanation_type}.json",
            )

    @contextmanager
    def _tracking_run(self, run_name: str) -> Generator[None, None, None]:
        if self.tracker is None:
            yield
            return
        with self.tracker.start_run(run_name=run_name):
            yield


def build_default_service() -> RewardSenseService:
    """Build default runtime service with environment-driven options."""
    enable_llm = _env_flag("ENABLE_LLM_EXPLANATIONS", default=False)

    tracker: Optional[RewardSenseTracker] = None
    if _env_flag("ENABLE_MLFLOW_TRACKING", default=True):
        try:
            tracker = RewardSenseTracker(experiment="llm-explainability")
        except Exception as exc:
            logger.warning(
                "MLflow tracking disabled due to initialization error: %s", exc
            )

    explanation_generator: Optional[ExplanationGenerator] = None
    if enable_llm:
        from src.model_pipeline.llm import VertexGeminiClient

        client = VertexGeminiClient(
            project_id=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("VERTEX_LOCATION", "us-central1"),
            model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            timeout_sec=float(os.getenv("LLM_TIMEOUT_SEC", "10")),
        )
        explanation_generator = ExplanationGenerator(llm_client=client)

    return RewardSenseService(
        scorer=PersonalizedScorer(),
        explanation_generator=explanation_generator,
        enable_llm_explanations=enable_llm,
        tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Default card catalog — loaded from shared catalog module.
# Used by /predict when the caller does not supply an explicit portfolio.
# ---------------------------------------------------------------------------
from src.app.cards.catalog import CARD_CATALOG as _DEFAULT_CARDS  # noqa: E402

if FASTAPI_AVAILABLE:

    class StrictBaseModel(BaseModel):
        """Base model that rejects unknown fields."""

        model_config = ConfigDict(extra="forbid")

    class RecommendRequest(BaseModel):
        """HTTP contract for recommendation requests."""

        portfolio: List[Dict[str, Any]]
        transaction: Dict[str, Any]
        user_features: Optional[List[Dict[str, Any]]] = None
        personalization_signals: Optional[Dict[str, Any]] = None
        explanation_type: ExplanationType = Field(
            default=ExplanationType.SINGLE_TRANSACTION
        )
        model_config = ConfigDict(extra="forbid")

    class ScoredCardResponse(StrictBaseModel):
        card_id: Optional[str] = None
        card_name: str = ""
        reward_amount: float
        reward_rate: float
        annual_fee: float = 0.0
        rank: Optional[int] = None
        raw_reward_amount: Optional[float] = None

    class RecommendationResponse(StrictBaseModel):
        ranked: List[ScoredCardResponse]
        best_card_id: Optional[str] = None
        point_value: float
        is_personalized: bool

    class FactualAccuracyResponse(StrictBaseModel):
        score: float
        passed: bool
        total_claims: int
        supported_claims: int
        unsupported_claims: List[str]

    class ReadabilityResponse(StrictBaseModel):
        flesch_reading_ease: float
        grade_level: float
        passed: bool

    class ExplanationResponse(StrictBaseModel):
        summary: str
        pros: List[str] = []
        cons: List[str] = []
        best_for: str = ""
        rationale: List[str] = []
        confidence: float
        disclaimers: List[str] = []
        used_fallback: bool
        fallback_reason: Optional[str] = None
        latency_ms: float
        quality_checks: Dict[str, bool]
        prompt_hash: str = ""
        model_name: str = ""
        factual_accuracy: FactualAccuracyResponse
        readability: ReadabilityResponse

    class RecommendResponse(StrictBaseModel):
        recommendation: RecommendationResponse
        explanation: Optional[ExplanationResponse] = None
        llm_explanations_enabled: bool

    class PredictRequest(BaseModel):
        user_id: str
        spending_categories: Dict[str, float]
        monthly_spend: float
        preferred_rewards: Optional[List[str]] = None
        transaction_history: Optional[List[Dict[str, Any]]] = None

    class PredictedCard(StrictBaseModel):
        card_name: str
        score: float
        rank: int

    class PredictResponse(StrictBaseModel):
        recommended_cards: List[PredictedCard]
        scores: Dict[str, float]
        explanation: Optional[ExplanationResponse] = None


def create_app(service: Optional[RewardSenseService] = None) -> Any:
    """Create a FastAPI app exposing recommendation endpoints."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed. Install `fastapi` and `uvicorn` to serve HTTP endpoints."
        )

    from src.app.auth.router import router as auth_router
    from src.app.db.init_db import init_db
    from src.app.feedback.router import router as feedback_router
    from src.app.users.router import router as users_router

    @asynccontextmanager
    async def lifespan(app):  # type: ignore[type-arg]
        init_db()
        yield

    import json as _json

    from fastapi.middleware.cors import CORSMiddleware

    _raw_origins = os.getenv("CORS_ORIGINS")
    if _raw_origins:
        try:
            _origins = _json.loads(_raw_origins)
            if not isinstance(_origins, list):
                _origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
        except _json.JSONDecodeError:
            _origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    else:
        _origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ]

    app = FastAPI(title="RewardSense API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(transactions_router)
    app.include_router(feedback_router)
    runtime_service = service or build_default_service()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model_version": os.getenv("MODEL_VERSION", "unknown"),
        }

    @app.get("/monitoring")
    def monitoring() -> Dict[str, Any]:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        return {
            "model_version": os.getenv("MODEL_VERSION", "unknown"),
            "last_deployment_time": os.getenv("MODEL_DEPLOYED_AT", now),
            "drift_check": {
                "detected": False,
                "timestamp": now,
                "feature_drift": {},
            },
            "serving_metrics": {
                "request_count": 0,
                "avg_latency_ms": 0,
                "error_rate": 0.0,
                "p95_latency_ms": 0,
            },
            "retrain_history": [],
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        # Build a single transaction from history or dominant spending category
        if payload.transaction_history:
            txn = payload.transaction_history[0]
            transaction: Dict[str, Any] = {
                "amount": float(txn.get("amount", 100.0)),
                "category": txn.get("category", "other"),
                "merchant": txn.get("merchant", "unknown"),
            }
        elif payload.spending_categories:
            dominant = max(
                payload.spending_categories, key=payload.spending_categories.get
            )
            transaction = {
                "amount": float(payload.spending_categories[dominant]),
                "category": dominant,
                "merchant": f"{dominant}-merchant",
            }
        else:
            transaction = {"amount": 100.0, "category": "other", "merchant": "unknown"}

        try:
            result = runtime_service.scorer.score(
                portfolio=_DEFAULT_CARDS,
                transaction=transaction,
            )
            ranked = result.get("ranked", [])
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Scoring failed: {exc}"
            ) from exc

        recommended_cards = [
            PredictedCard(
                card_name=card.get("card_name", "Unknown"),
                score=round(float(card.get("reward_amount", 0.0)), 4),
                rank=int(card.get("rank", i + 1)),
            )
            for i, card in enumerate(ranked)
        ]
        scores = {c.card_name: c.score for c in recommended_cards}
        return PredictResponse(
            recommended_cards=recommended_cards, scores=scores, explanation=None
        )

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(payload: RecommendRequest) -> RecommendResponse:
        try:
            features_df = (
                pd.DataFrame(payload.user_features) if payload.user_features else None
            )
            raw_response = runtime_service.recommend(
                portfolio=payload.portfolio,
                transaction=payload.transaction,
                user_features=features_df,
                personalization_signals=payload.personalization_signals,
                explanation_type=payload.explanation_type,
            )
            return RecommendResponse.model_validate(raw_response)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Recommendation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Static frontend serving (SPA fallback)
    # Serves the Vite-built frontend from /app/static if it exists.
    # All non-API paths fall through to index.html for client-side routing.
    # ------------------------------------------------------------------
    from pathlib import Path as _Path

    _static_dir = _Path(os.getenv("STATIC_DIR", "/app/static"))
    if _static_dir.is_dir():
        from fastapi.responses import FileResponse

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str) -> FileResponse:
            file_path = _static_dir / full_path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(_static_dir / "index.html")

    return app


def main() -> None:
    """CLI entrypoint for quick local smoke check."""
    service = build_default_service()
    print(
        "RewardSenseService initialized "
        f"(llm_explanations_enabled={service.enable_llm_explanations})"
    )
    raise SystemExit(0)
