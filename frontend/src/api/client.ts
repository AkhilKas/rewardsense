// API SWAP GUIDE (remove this block when real API is live)
//
// 1. Set VITE_API_URL in your .env file:
//    VITE_API_URL=https://rewardsense-serving-xxxxx.us-central1.run.app
//
// 2. The real API endpoints are:
//    - POST ${VITE_API_URL}/predict  (body: PredictionRequest → response: PredictionResponse)
//    - GET  ${VITE_API_URL}/health   (response: HealthResponse)
//    - GET  ${VITE_API_URL}/monitoring (response: MonitoringData)
//
// 3. Once VITE_API_URL is set, this file auto-switches from mock to real fetch calls.
//    No other changes needed — the USE_MOCK flag handles it.
//
// 4. To fully remove mocks: delete mock.ts, remove the mock imports below,
//    and remove the USE_MOCK branches.
//
// 5. Quick smoke test:
//    curl -X POST ${VITE_API_URL}/predict \
//      -H "Content-Type: application/json" \
//      -d '{"user_id":"test","spending_categories":{"dining":500,"travel":300},"monthly_spend":2000,"preferred_rewards":["travel"]}'

import type {
  PredictionRequest,
  PredictionResponse,
  HealthResponse,
  MonitoringData,
} from "../types";
import { mockPredict, mockHealth, mockMonitoringData } from "./mock";

const API_BASE_URL = import.meta.env.VITE_API_URL || "";
const USE_MOCK = !API_BASE_URL;

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
  return res.json();
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
