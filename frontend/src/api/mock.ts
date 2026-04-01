import type {
  PredictionRequest,
  PredictionResponse,
  HealthResponse,
  MonitoringData,
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
