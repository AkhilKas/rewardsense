import type {
  CardCatalogItem,
  PersonaRecommendResponse,
  PredictionResponse,
} from "../types";
import type {
  QuickRecommendationViewModel,
  RecommendationResultViewModel,
} from "../types/viewmodels";

export function mapPortfolioToPredictionResponse(params: {
  portfolio: PersonaRecommendResponse;
  catalog: CardCatalogItem[];
  totalSpend: number;
  latencyMs: number;
}): PredictionResponse {
  const { portfolio, catalog, totalSpend, latencyMs } = params;
  const catalogById = new Map<string, CardCatalogItem>(
    catalog.map((c) => [c.card_id, c]),
  );
  // Prefer cards that produce positive net rewards for cleaner UX.
  // If all cards are non-positive, fall back to full ranked list.
  const rankedForDisplay = portfolio.ranked.filter((c) => c.reward_amount > 0);
  const cardsToDisplay =
    rankedForDisplay.length > 0 ? rankedForDisplay : portfolio.ranked;

  const rewardValues = cardsToDisplay.map((c) => c.reward_amount);
  const maxReward = Math.max(...rewardValues, 0);
  const minReward = Math.min(...rewardValues, 0);
  const rewardRange = Math.max(maxReward - minReward, 1);

  return {
    recommended_cards: cardsToDisplay.map((card, index) => {
      const meta = card.card_id ? catalogById.get(card.card_id) : undefined;
      const displayRank = index + 1;
      // Use backend-generated explanation/pros/cons when available
      const issuer =
        card.card_display?.issuer ?? meta?.issuer ?? "Unknown issuer";
      const highlights =
        card.card_display?.reward_highlights ??
        meta?.reward_highlights ??
        [];
      const savingsStr = card.projected_savings
        ? `$${card.projected_savings.toLocaleString()}`
        : `$${card.reward_amount.toFixed(2)}`;
      const fallbackExplanation = `${card.card_name} — projected annual reward: ${savingsStr}.`;

      return {
        card_name: card.card_name,
        issuer,
        score: Math.round(((card.reward_amount - minReward) / rewardRange) * 100),
        rank: displayRank,
        explanation: card.explanation || fallbackExplanation,
        pros: card.pros ?? [],
        cons: card.cons ?? [],
        best_for: card.best_for ?? "",
        annual_fee: card.annual_fee,
        reward_rate:
          totalSpend > 0
            ? Number(((card.reward_amount / totalSpend) * 100).toFixed(2))
            : 0,
        key_benefits:
          highlights.length > 0
            ? highlights
            : ["Strong fit for your selected spending profile"],
        score_breakdown: {
          deterministic: Number(Math.max(0, card.reward_amount).toFixed(2)),
          personalization: 0,
        },
      };
    }),
    model_version: portfolio.is_personalized ? "app-persona-v1" : "app-generic-v1",
    inference_latency_ms: latencyMs,
  };
}

export function mapPredictionToRecommendationVM(
  response: PredictionResponse,
): RecommendationResultViewModel {
  return {
    modelVersion: response.model_version,
    latencyMs: response.inference_latency_ms,
    cards: response.recommended_cards.map((card, index) => ({
      id: `${card.card_name}-${card.rank}-${index}`,
      name: card.card_name,
      issuer: card.issuer,
      score: card.score,
      rank: card.rank,
      explanation: card.explanation,
      pros: card.pros ?? [],
      cons: card.cons ?? [],
      bestFor: card.best_for ?? "",
      annualFee: card.annual_fee,
      rewardRate: card.reward_rate,
      keyBenefits: card.key_benefits,
      scoreBreakdown: {
        base: card.score_breakdown.deterministic,
        boosted: card.score_breakdown.personalization,
      },
    })),
  };
}

export function mapQuickRecommendToVM(
  response: PersonaRecommendResponse,
): QuickRecommendationViewModel {
  return {
    context: response.persona_context,
    cards: response.ranked.map((card, index) => ({
      id: card.card_id ?? `${card.card_name}-${card.rank}-${index}`,
      name: card.card_name,
      rank: card.rank,
      rewardAmount: card.reward_amount,
      annualFee: card.annual_fee,
    })),
  };
}
