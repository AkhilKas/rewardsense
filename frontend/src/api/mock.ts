import type {
  PredictionRequest,
  PredictionResponse,
  HealthResponse,
  MonitoringData,
  CardCatalogItem,
  PersonaRecommendResponse,
  PortfolioRecommendRequest,
  TransactionRecommendRequest,
  QuickTransactionRequest,
  QuickTransactionResponse,
  TransactionsResponse,
  TransactionsExportResponse,
  SummaryResponse,
  FeedbackRequest,
  FeedbackResponse,
  BusinessMetricsResponse,
} from "../types";

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function mockPredict(
  _request: PredictionRequest,
): Promise<PredictionResponse> {
  await delay(1200);

  return {
    recommended_cards: [
      {
        card_name: "Chase Sapphire Preferred",
        issuer: "Chase",
        score: 92.5,
        rank: 1,
        explanation:
          "This card maximizes your dining and travel spending with 3x points on dining and 2x on travel. Based on your spending pattern, you'd earn approximately 45,000 points annually, worth about $562 in travel redemptions.",
        annual_fee: 95,
        reward_rate: 2.8,
        key_benefits: [
          "3x on dining",
          "2x on travel",
          "$50 hotel credit",
          "Trip cancellation insurance",
        ],
        score_breakdown: { deterministic: 88.0, personalization: 97.0 },
      },
      {
        card_name: "Amex Gold Card",
        issuer: "American Express",
        score: 87.3,
        rank: 2,
        explanation:
          "Your high grocery and dining spend makes this card an excellent choice. 4x points on groceries and dining gives you strong category coverage. The $120 dining credit effectively reduces the annual fee.",
        annual_fee: 250,
        reward_rate: 3.2,
        key_benefits: [
          "4x on groceries",
          "4x on dining",
          "$120 dining credit",
          "$120 Uber credit",
        ],
        score_breakdown: { deterministic: 85.0, personalization: 89.6 },
      },
      {
        card_name: "Citi Double Cash",
        issuer: "Citi",
        score: 78.1,
        rank: 3,
        explanation:
          "A straightforward 2% cashback on everything suits your diversified spending pattern well. No annual fee means all rewards are pure profit. Ideal as a catch-all secondary card.",
        annual_fee: 0,
        reward_rate: 2.0,
        key_benefits: [
          "2% on everything",
          "No annual fee",
          "0% intro APR",
          "Citi Entertainment access",
        ],
        score_breakdown: { deterministic: 80.0, personalization: 76.2 },
      },
    ],
    model_version: "v2.1.0",
    inference_latency_ms: 1150,
  };
}

export async function mockHealth(): Promise<HealthResponse> {
  return {
    status: "healthy",
    model_version: "v2.1.0",
    uptime_seconds: 86400,
  };
}

export async function mockMonitoringData(): Promise<MonitoringData> {
  await delay(500);

  return {
    model_version: "v2.1.0",
    last_deployment_time: "2026-03-12T14:30:00Z",
    drift_check: {
      detected: false,
      timestamp: "2026-03-14T06:00:00Z",
      feature_drift: {
        monthly_spend: 0.03,
        dining_ratio: 0.05,
        travel_ratio: 0.02,
        grocery_ratio: 0.08,
        gas_ratio: 0.01,
        online_ratio: 0.12,
        avg_transaction: 0.04,
        spending_velocity: 0.06,
        category_diversity: 0.09,
        reward_preference: 0.07,
      },
    },
    serving_metrics: {
      request_count: 12847,
      avg_latency_ms: 1250,
      error_rate: 0.002,
      p95_latency_ms: 3200,
    },
    retrain_history: [
      {
        timestamp: "2026-03-12T14:00:00Z",
        trigger_reason: "Scheduled weekly retrain",
        model_version: "v2.1.0",
        status: "success",
      },
      {
        timestamp: "2026-03-05T14:00:00Z",
        trigger_reason: "Scheduled weekly retrain",
        model_version: "v2.0.3",
        status: "success",
      },
      {
        timestamp: "2026-02-28T09:15:00Z",
        trigger_reason: "Feature drift detected (online_ratio)",
        model_version: "v2.0.2",
        status: "success",
      },
      {
        timestamp: "2026-02-26T14:00:00Z",
        trigger_reason: "Scheduled weekly retrain",
        model_version: "v2.0.1",
        status: "failed",
      },
      {
        timestamp: "2026-02-19T14:00:00Z",
        trigger_reason: "Scheduled weekly retrain",
        model_version: "v2.0.0",
        status: "success",
      },
    ],
  };
}

const CATALOG: CardCatalogItem[] = [
  {
    card_id: "chase_sapphire_preferred",
    card_name: "Chase Sapphire Preferred",
    issuer: "Chase",
    annual_fee: 95,
    reward_highlights: ["3x on dining", "2x on travel"],
    image_url: "/cards/chase-gradient.svg",
  },
  {
    card_id: "amex_gold",
    card_name: "Amex Gold Card",
    issuer: "American Express",
    annual_fee: 250,
    reward_highlights: ["4x dining", "4x groceries"],
    image_url: "/cards/amex-gradient.svg",
  },
  {
    card_id: "citi_double_cash",
    card_name: "Citi Double Cash",
    issuer: "Citi",
    annual_fee: 0,
    reward_highlights: ["2% on all spend"],
    image_url: "/cards/citi-gradient.svg",
  },
];

export async function mockCardsCatalog(): Promise<CardCatalogItem[]> {
  await delay(250);
  return CATALOG;
}

export async function mockRecommendPortfolio(
  _request: PortfolioRecommendRequest,
): Promise<PersonaRecommendResponse> {
  await delay(900);
  return {
    ranked: [
      {
        card_id: "chase_sapphire_preferred",
        card_name: "Chase Sapphire Preferred",
        reward_amount: 112.3,
        annual_fee: 95,
        rank: 1,
        persona_adjustments: { traveler: { category_boost: 1.5 } },
      },
      {
        card_id: "amex_gold",
        card_name: "Amex Gold Card",
        reward_amount: 98.7,
        annual_fee: 250,
        rank: 2,
      },
      {
        card_id: "citi_double_cash",
        card_name: "Citi Double Cash",
        reward_amount: 72.1,
        annual_fee: 0,
        rank: 3,
      },
    ],
    best_card_id: "chase_sapphire_preferred",
    is_personalized: true,
    is_generic: false,
    active_personas: ["traveler"],
    persona_context: "Traveler profile boosted travel and dining rewards.",
  };
}

export async function mockRecommendTransaction(
  request: TransactionRecommendRequest,
): Promise<PersonaRecommendResponse> {
  await delay(650);
  return {
    ranked: [
      {
        card_id: "amex_gold",
        card_name: "Amex Gold Card",
        reward_amount: request.amount * 0.04,
        annual_fee: 250,
        rank: 1,
      },
      {
        card_id: "chase_sapphire_preferred",
        card_name: "Chase Sapphire Preferred",
        reward_amount: request.amount * 0.03,
        annual_fee: 95,
        rank: 2,
      },
      {
        card_id: "citi_double_cash",
        card_name: "Citi Double Cash",
        reward_amount: request.amount * 0.02,
        annual_fee: 0,
        rank: 3,
      },
    ],
    best_card_id: "amex_gold",
    is_personalized: true,
    is_generic: false,
    active_personas: ["cashback-focused"],
    persona_context: "Cashback-focused profile favors high flat-value returns.",
  };
}

export async function mockRecommendQuickTransaction(
  request: QuickTransactionRequest,
): Promise<QuickTransactionResponse> {
  await delay(400);
  return {
    top_card: {
      card_id: "amex_gold",
      card_name: "Amex Gold Card",
      reward_amount: request.amount * 0.04,
      annual_fee: 250,
      rank: 1,
    },
    alternatives: [
      {
        card_id: "chase_sapphire_preferred",
        card_name: "Chase Sapphire Preferred",
        reward_amount: request.amount * 0.03,
        annual_fee: 95,
        rank: 2,
      },
      {
        card_id: "citi_double_cash",
        card_name: "Citi Double Cash",
        reward_amount: request.amount * 0.02,
        annual_fee: 0,
        rank: 3,
      },
    ],
    estimated_reward: request.amount * 0.04,
    money_saved: request.amount * 0.04,
    category_used: request.category || "dining",
    is_personalized: true,
    has_saved_cards: true,
    active_personas: ["cashback-focused"],
    persona_context: "Cashback-focused profile favors high flat-value returns.",
  };
}

export async function mockTransactions(
  page = 1,
  pageSize = 10,
): Promise<TransactionsResponse> {
  await delay(300);
  const items: TransactionsResponse["items"] = Array.from({ length: pageSize }).map(
    (_, i) => {
    const idx = (page - 1) * pageSize + i + 1;
    return {
      id: `tx_${idx}`,
      merchant: ["Whole Foods", "Uber", "Starbucks"][idx % 3],
      category: ["groceries", "travel", "dining"][idx % 3],
      amount: 25 + idx * 3,
      card_id: ["amex_gold", "chase_sapphire_preferred", "citi_double_cash"][idx % 3],
      reward_earned: Number((1.2 + idx * 0.15).toFixed(2)),
      estimated_savings: Number((0.7 + idx * 0.1).toFixed(2)),
      baseline_savings: Number((0.5 + idx * 0.05).toFixed(2)),
      timestamp: new Date(Date.now() - idx * 3600000).toISOString(),
      source_flow: idx % 2 ? "quick_transaction" : "portfolio_recommendation",
    };
  });
  return {
    items,
    page,
    page_size: pageSize,
    total_items: 84,
    total_pages: Math.ceil(84 / pageSize),
  };
}

export async function mockTransactionsExport(
  format: "csv" | "xlsx",
): Promise<TransactionsExportResponse> {
  await delay(250);
  return {
    format,
    download_url: `/downloads/transactions-${Date.now()}.${format === "csv" ? "csv" : "xlsx"}`,
  };
}

export async function mockSummary(): Promise<SummaryResponse> {
  await delay(350);
  return {
    spend_by_category: [
      { category: "dining", amount: 980 },
      { category: "travel", amount: 760 },
      { category: "groceries", amount: 640 },
    ],
    rewards_by_category: [
      { category: "dining", reward_earned: 39.2 },
      { category: "travel", reward_earned: 22.8 },
      { category: "groceries", reward_earned: 25.6 },
    ],
    savings_by_card: [
      { card_id: "amex_gold", card_name: "Amex Gold Card", savings: 52.1 },
      {
        card_id: "chase_sapphire_preferred",
        card_name: "Chase Sapphire Preferred",
        savings: 41.4,
      },
      { card_id: "citi_double_cash", card_name: "Citi Double Cash", savings: 28.2 },
    ],
    fee_adjusted_savings_total: 104.3,
  };
}

export async function mockFeedback(
  _request: FeedbackRequest,
): Promise<FeedbackResponse> {
  await delay(220);
  return {
    ok: true,
    feedback_id: `fb_${Date.now()}`,
  };
}

export async function mockBusinessMetrics(): Promise<BusinessMetricsResponse> {
  await delay(450);
  return {
    generated_at: new Date().toISOString(),
    report_url_html: "/reports/business-metrics-latest.html",
    report_url_pdf: "/reports/business-metrics-latest.pdf",
    total_requests: 12847,
    avg_latency_ms: 1248,
    p95_latency_ms: 3180,
    estimated_llm_cost_usd: 42.17,
    fallback_rate: 0.018,
    error_rate: 0.003,
  };
}
