import { useState, useEffect } from "react";
import { useLocation, Navigate, Link } from "react-router-dom";
import Card from "../components/Card";
import Badge from "../components/Badge";
import Button from "../components/Button";
import ScoreGauge from "../components/ScoreGauge";
import Collapsible from "../components/Collapsible";
import type { PredictionResponse, RecommendedCard } from "../types";

const EXPLANATION_FALLBACK =
  "Explanation unavailable — the AI explainer didn't generate text for this card. The score is based on your spending profile and the card's reward structure.";

/* ------------------------------------------------------------------ */
/*  Card View — staggered fade-in list                                */
/* ------------------------------------------------------------------ */

function CardView({ cards }: { cards: RecommendedCard[] }) {
  return (
    <div className="space-y-4">
      {cards.map((card, i) => (
        <div
          key={card.rank}
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
                    {card.card_name}
                  </h2>
                  {card.rank === 1 && (
                    <Badge variant="success">Top Pick</Badge>
                  )}
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  {card.issuer} &middot; ${card.annual_fee}/yr &middot;{" "}
                  {card.reward_rate}% avg rewards
                </p>
              </div>
            </div>

            {/* Score breakdown */}
            <ScoreBreakdownBar card={card} />

            {/* Benefits */}
            <div className="flex flex-wrap gap-1.5 mb-3">
              {card.key_benefits.map((b) => (
                <Badge key={b} variant="info">
                  {b}
                </Badge>
              ))}
            </div>

            {/* Collapsible explanation */}
            <Collapsible title="Why this card?" defaultOpen={card.rank === 1}>
              <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
                {card.explanation || EXPLANATION_FALLBACK}
              </p>
            </Collapsible>
          </Card>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Compare View — aligned grid rows                                  */
/* ------------------------------------------------------------------ */

function CompareView({ cards }: { cards: RecommendedCard[] }) {
  const [activeIdx, setActiveIdx] = useState(0);

  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    const idx = Math.round(el.scrollLeft / (el.scrollWidth / cards.length));
    setActiveIdx(Math.min(idx, cards.length - 1));
  }

  const cardContent = cards.map((card) => (
    <div
      key={card.rank}
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
            {card.card_name}
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            {card.issuer} &middot; ${card.annual_fee}/yr
          </p>
        </div>

        {/* Row 4 — Score breakdown */}
        <div className="mb-3 min-h-[48px]">
          <ScoreBreakdownBar card={card} compact />
        </div>

        {/* Row 5 — Benefits */}
        <div className="mb-3 min-h-[72px]">
          <div className="flex flex-wrap gap-1">
            {card.key_benefits.slice(0, 4).map((b) => (
              <Badge key={b} variant="info" className="text-[10px]">
                {b}
              </Badge>
            ))}
          </div>
        </div>

        {/* Row 6 — Explanation snippet */}
        <div className="flex-1 min-h-[80px]">
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed line-clamp-4">
            {card.explanation || EXPLANATION_FALLBACK}
          </p>
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
  card: RecommendedCard;
  compact?: boolean;
}) {
  const total =
    card.score_breakdown.deterministic + card.score_breakdown.personalization;
  const detPct = total > 0 ? (card.score_breakdown.deterministic / total) * 100 : 50;
  const persPct = 100 - detPct;
  const textClass = compact
    ? "text-[10px] text-slate-500 dark:text-slate-400"
    : "text-xs text-slate-500 dark:text-slate-400";

  return (
    <div className="mb-3">
      <div className={`flex items-center justify-between mb-1 ${textClass}`}>
        <span>Det: {card.score_breakdown.deterministic.toFixed(1)}</span>
        <span>ML: {card.score_breakdown.personalization.toFixed(1)}</span>
      </div>
      <div className="h-2 rounded-full bg-border overflow-hidden flex">
        <div className="bg-primary/70 h-full" style={{ width: `${detPct}%` }} />
        <div className="bg-accent/70 h-full" style={{ width: `${persPct}%` }} />
      </div>
      {!compact && (
        <div className="flex items-center gap-4 mt-1 text-xs text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-primary/70" />
            Deterministic
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-accent/70" />
            Personalization
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

  const [results] = useState<PredictionResponse | null>(
    () => location.state as PredictionResponse | null,
  );

  const [view, setView] = useState<"cards" | "compare">("cards");

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    if (results) {
      window.history.replaceState({}, document.title);
    }
  }, [results]);

  if (!results) {
    return <Navigate to="/recommend" replace />;
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-secondary">Your Results</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Model {results.model_version} &middot;{" "}
            {results.inference_latency_ms}ms inference &middot;{" "}
            {results.recommended_cards.length} cards ranked
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
        <CardView cards={results.recommended_cards} />
      ) : (
        <CompareView cards={results.recommended_cards} />
      )}
    </div>
  );
}
