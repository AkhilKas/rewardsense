export interface RecommendationCardViewModel {
  id: string;
  name: string;
  issuer: string;
  score: number;
  rank: number;
  explanation: string;
  pros: string[];
  cons: string[];
  bestFor: string;
  annualFee: number;
  rewardRate: number;
  keyBenefits: string[];
  scoreBreakdown: {
    base: number;
    boosted: number;
  };
}

export interface RecommendationResultViewModel {
  cards: RecommendationCardViewModel[];
  modelVersion: string;
  latencyMs: number;
}

export interface QuickCardViewModel {
  id: string;
  name: string;
  rank: number;
  rewardAmount: number;
  annualFee: number;
}

export interface QuickRecommendationViewModel {
  context: string;
  categoryUsed: string;
  estimatedReward: number;
  moneySaved: number;
  hasSavedCards: boolean;
  cards: QuickCardViewModel[];
}
