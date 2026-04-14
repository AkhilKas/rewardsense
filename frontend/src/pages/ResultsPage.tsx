import { useState, useEffect, useCallback } from "react";
import { useLocation, Navigate, Link } from "react-router-dom";
import Card from "../components/Card";
import Badge from "../components/Badge";
import Button from "../components/Button";
import ScoreGauge from "../components/ScoreGauge";
import Collapsible from "../components/Collapsible";
import FeedbackButtons from "../components/FeedbackButtons";
import Confetti from "../components/Confetti";
import { submitFeedback } from "../api/client";
import type { PredictionResponse, FeedbackReasonTag } from "../types";
import type {
  RecommendationCardViewModel,
  RecommendationResultViewModel,
} from "../types/viewmodels";
import { mapPredictionToRecommendationVM } from "../viewmodels/viewMappers";

const EXPLANATION_FALLBACK =
  "Explanation unavailable — the AI explainer didn't generate text for this card. The score is based on your spending profile and the card's reward structure.";

const FALLBACK_PROS = [
  "Strong reward rate for your spending profile.",
  "Competitive benefits compared to alternatives.",
];
const FALLBACK_CONS = [
  "Annual fee may offset rewards for low spenders.",
  "Reward rates may vary by merchant within categories.",
];

function prefersReducedMotion(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/* ------------------------------------------------------------------ */
/*  Card View — staggered fade-in list                                */
/* ------------------------------------------------------------------ */

function useFeedbackHandler() {
  return useCallback(
    (cardId: string, target: "card" | "explanation") =>
      (reaction: "like" | "dislike", reasonTag?: FeedbackReasonTag) => {
        submitFeedback({
          card_id: cardId,
          reaction,
          reason_tag: reasonTag,
          target,
        }).catch(() => {
          /* fire-and-forget */
        });
      },
    [],
  );
}

function CardView({ cards }: { cards: RecommendationCardViewModel[] }) {
  const makeFeedbackHandler = useFeedbackHandler();
  return (
    <div className="space-y-4">
      {cards.map((card, i) => (
        <div
          key={card.id}
          className="animate-card-in"
          style={{ animationDelay: `${i * 150}ms` }}
        >
          <Card>
            {/* Header row: gauge + name */}
            <div className="flex items-center gap-5 mb-4">
              <ScoreGauge score={card.score} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <h2 className="text-lg font-bold text-secondary">
                    {card.name}
                  </h2>
                  {card.rank === 1 && (
                    <Badge variant="success">Top Pick</Badge>
                  )}
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  {card.issuer} &middot; ${card.annualFee}/yr &middot;{" "}
                  {card.rewardRate}% avg rewards
                </p>
              </div>
            </div>

            {/* Score breakdown */}
            <ScoreBreakdownBar card={card} />

            {/* Benefits */}
            <div className="flex flex-wrap gap-1.5 mb-3">
              {card.keyBenefits.map((b) => (
                <Badge key={b} variant="info">
                  {b}
                </Badge>
              ))}
            </div>

            {/* Structured explanation */}
            <Collapsible title="Why this card?" defaultOpen={card.rank === 1}>
              <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed mb-3">
                {card.explanation || EXPLANATION_FALLBACK}
              </p>

              {/* Pros */}
              <div className="mb-2">
                <h4 className="text-xs font-semibold text-green-700 dark:text-green-400 uppercase tracking-wide mb-1">Pros</h4>
                <ul className="space-y-1">
                  {(card.pros.length > 0 ? card.pros : FALLBACK_PROS).map((pro, j) => (
                    <li key={j} className="flex items-start gap-1.5 text-sm text-slate-600 dark:text-slate-400">
                      <span className="text-green-500 mt-0.5 shrink-0">&#10003;</span>
                      <span>{pro}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Cons */}
              <div className="mb-2">
                <h4 className="text-xs font-semibold text-amber-700 dark:text-amber-400 uppercase tracking-wide mb-1">Cons</h4>
                <ul className="space-y-1">
                  {(card.cons.length > 0 ? card.cons : FALLBACK_CONS).map((con, j) => (
                    <li key={j} className="flex items-start gap-1.5 text-sm text-slate-600 dark:text-slate-400">
                      <span className="text-amber-500 mt-0.5 shrink-0">&#9888;</span>
                      <span>{con}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Best for */}
              {card.bestFor && (
                <p className="text-sm text-slate-500 dark:text-slate-400 italic">
                  Best for: {card.bestFor}
                </p>
              )}
            </Collapsible>

            {/* Feedback */}
            <FeedbackButtons
              cardId={card.id}
              target="card"
              onSubmit={makeFeedbackHandler(card.id, "card")}
            />
          </Card>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Compare View — aligned grid rows                                  */
/* ------------------------------------------------------------------ */

function CompareView({ cards }: { cards: RecommendationCardViewModel[] }) {
  const [activeIdx, setActiveIdx] = useState(0);

  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    const idx = Math.round(el.scrollLeft / (el.scrollWidth / cards.length));
    setActiveIdx(Math.min(idx, cards.length - 1));
  }

  const cardContent = cards.map((card) => (
    <div
      key={card.id}
      className="min-w-[85vw] snap-center md:min-w-0"
    >
      <Card padding="sm" className="flex flex-col h-full">
        {/* Row 1 — Rank badge */}
        <div className="mb-3 min-h-[28px]">
          {card.rank === 1 ? (
            <Badge variant="success">Top Pick</Badge>
          ) : (
            <Badge variant="info">#{card.rank}</Badge>
          )}
        </div>

        {/* Row 2 — Score gauge */}
        <div className="flex justify-center mb-3 min-h-[88px]">
          <ScoreGauge score={card.score} size={80} />
        </div>

        {/* Row 3 — Name & issuer */}
        <div className="text-center mb-3 min-h-[56px]">
          <h3 className="font-bold text-secondary text-sm leading-tight">
            {card.name}
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            {card.issuer} &middot; ${card.annualFee}/yr
          </p>
        </div>

        {/* Row 4 — Score breakdown */}
        <div className="mb-3 min-h-[48px]">
          <ScoreBreakdownBar card={card} compact />
        </div>

        {/* Row 5 — Benefits */}
        <div className="mb-3 min-h-[72px]">
          <div className="flex flex-wrap gap-1">
            {card.keyBenefits.slice(0, 4).map((b) => (
              <Badge key={b} variant="info" className="text-[10px]">
                {b}
              </Badge>
            ))}
          </div>
        </div>

        {/* Row 6 — Explanation with pros/cons */}
        <div className="flex-1 min-h-[80px]">
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed line-clamp-2 mb-1">
            {card.explanation || EXPLANATION_FALLBACK}
          </p>
          <div className="space-y-0.5">
            {(card.pros.length > 0 ? card.pros : FALLBACK_PROS).slice(0, 1).map((pro, j) => (
              <p key={j} className="text-[10px] text-green-600 dark:text-green-400 truncate">
                &#10003; {pro}
              </p>
            ))}
            {(card.cons.length > 0 ? card.cons : FALLBACK_CONS).slice(0, 1).map((con, j) => (
              <p key={j} className="text-[10px] text-amber-600 dark:text-amber-400 truncate">
                &#9888; {con}
              </p>
            ))}
          </div>
        </div>
      </Card>
    </div>
  ));

  return (
    <div className="animate-card-in">
      {/* Mobile: horizontal swipe | Desktop: grid */}
      <div
        className="flex gap-4 overflow-x-auto snap-x snap-mandatory pb-2 md:grid md:grid-cols-3 md:overflow-x-visible scrollbar-none"
        onScroll={handleScroll}
      >
        {cardContent}
      </div>

      {/* Mobile scroll indicator dots */}
      <div className="flex md:hidden items-center justify-center gap-1.5 mt-3">
        {cards.map((_, i) => (
          <span
            key={i}
            className={`inline-block rounded-full transition-all duration-200 ${
              i === activeIdx
                ? "w-5 h-2 bg-primary"
                : "w-2 h-2 bg-border"
            }`}
          />
        ))}
        <span className="ml-2 text-xs text-slate-400 dark:text-slate-500">
          Swipe to compare
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Score Breakdown Bar — shared by both views                        */
/* ------------------------------------------------------------------ */

function ScoreBreakdownBar({
  card,
  compact = false,
}: {
  card: RecommendationCardViewModel;
  compact?: boolean;
}) {
  const total = card.scoreBreakdown.base + card.scoreBreakdown.boosted;
  const detPct = total > 0 ? (card.scoreBreakdown.base / total) * 100 : 50;
  const persPct = 100 - detPct;
  const textClass = compact
    ? "text-[10px] text-slate-500 dark:text-slate-400"
    : "text-xs text-slate-500 dark:text-slate-400";

  return (
    <div className="mb-3">
      <div className={`flex items-center justify-between mb-1 ${textClass}`}>
        <span>Base: {card.scoreBreakdown.base.toFixed(1)}</span>
        <span>Boosted: {card.scoreBreakdown.boosted.toFixed(1)}</span>
      </div>
      <div className="h-2 rounded-full bg-border overflow-hidden flex">
        <div className="bg-primary/70 h-full" style={{ width: `${detPct}%` }} />
        <div className="bg-accent/70 h-full" style={{ width: `${persPct}%` }} />
      </div>
      {!compact && (
        <div className="flex items-center gap-4 mt-1 text-xs text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-primary/70" />
            Base Match
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-accent/70" />
            Personalized Boost
          </span>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Results Page                                                 */
/* ------------------------------------------------------------------ */

export default function ResultsPage() {
  const location = useLocation();

  const [results] = useState<RecommendationResultViewModel | null>(() => {
    const state = location.state as PredictionResponse | null;
    return state ? mapPredictionToRecommendationVM(state) : null;
  });

  const [view, setView] = useState<"cards" | "compare">("cards");
  const [showConfetti, setShowConfetti] = useState(() => {
    const state = location.state as PredictionResponse | null;
    return Boolean(state && !prefersReducedMotion());
  });

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    if (results) {
      window.history.replaceState({}, document.title);
    }
  }, [results]);

  useEffect(() => {
    if (!results || prefersReducedMotion()) return;
    const t = window.setTimeout(() => setShowConfetti(false), 2500);
    return () => window.clearTimeout(t);
  }, [results]);

  if (!results) {
    return <Navigate to="/recommend" replace />;
  }

  return (
    <>
      {showConfetti ? <Confetti /> : null}
      <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-secondary">Your Results</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            {results.cards.length} cards ranked &middot; analyzed
            in {results.latencyMs}ms
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* View toggle */}
          <div className="inline-flex rounded-md border border-border overflow-hidden">
            <button
              onClick={() => setView("cards")}
              className={`px-3 py-1.5 text-sm font-medium transition-colors cursor-pointer ${
                view === "cards"
                  ? "bg-primary text-white"
                  : "bg-card text-secondary hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              Cards
            </button>
            <button
              onClick={() => setView("compare")}
              className={`px-3 py-1.5 text-sm font-medium transition-colors cursor-pointer ${
                view === "compare"
                  ? "bg-primary text-white"
                  : "bg-card text-secondary hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              Compare
            </button>
          </div>
          <Link to="/recommend">
            <Button variant="secondary" size="sm">
              Try Different Profile
            </Button>
          </Link>
        </div>
      </div>

      {/* Content */}
      {view === "cards" ? (
        <CardView cards={results.cards} />
      ) : (
        <CompareView cards={results.cards} />
      )}
    </div>
    </>
  );
}
