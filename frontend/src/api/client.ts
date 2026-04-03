// API CLIENT — auto-switches between mock and real API based on VITE_API_URL.
// To switch back to mocks: set USE_MOCK to true below.
//
// Setup:
//   1. Create frontend/.env with:  VITE_API_URL=https://rewardsense-serving-xxxxx.us-central1.run.app
//   2. For local dev:              VITE_API_URL=http://localhost:8000
//
// Endpoints (all implemented on the serving API):
//   POST ${VITE_API_URL}/predict     → PredictionResponse
//   GET  ${VITE_API_URL}/health      → HealthResponse
//   GET  ${VITE_API_URL}/monitoring   → MonitoringData
//
// Smoke test:
//   curl -X POST ${VITE_API_URL}/predict \
//     -H "Content-Type: application/json" \
//     -d '{"user_id":"test","spending_categories":{"dining":500,"travel":300},"monthly_spend":2000,"preferred_rewards":["travel"]}'
//   curl ${VITE_API_URL}/monitoring

import type {
  PredictionRequest,
  PredictionResponse,
  HealthResponse,
  MonitoringData,
} from "../types";
import { mockPredict, mockHealth, mockMonitoringData } from "./mock";

const API_BASE_URL = import.meta.env.VITE_API_URL || "";
const USE_MOCK = !API_BASE_URL; // Set to `true` to force mock data for development

export async function predict(
  request: PredictionRequest,
): Promise<PredictionResponse> {
  if (USE_MOCK) return mockPredict(request);

  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data: PredictionResponse = await res.json();

  const maxScore = Math.max(...data.recommended_cards.map((c) => c.score), 1);

  data.recommended_cards = data.recommended_cards.map((card) => ({
    ...card,
    score: Math.round((card.score / maxScore) * 100),
    score_breakdown: {
      deterministic: card.deterministic_score ?? 0,
      personalization: card.personalization_score ?? 0,
    },
  }));

  return data;
}

export async function health(): Promise<HealthResponse> {
  if (USE_MOCK) return mockHealth();

  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getMonitoringData(): Promise<MonitoringData> {
  if (USE_MOCK) return mockMonitoringData();

  const res = await fetch(`${API_BASE_URL}/monitoring`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
