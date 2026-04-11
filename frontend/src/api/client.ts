// API CLIENT — Phase 4 uses src/app/* endpoints.
// To switch back to mocks: set USE_MOCK to true below.

import type {
  PredictionRequest,
  PredictionResponse,
  HealthResponse,
  MonitoringData,
  UserProfile,
  ProfilePatch,
  CardCatalogItem,
  PortfolioRecommendRequest,
  TransactionRecommendRequest,
  PersonaRecommendResponse,
  TransactionsResponse,
  TransactionCreateRequest,
  TransactionLogEntry,
  SummaryResponse,
  FeedbackRequest,
  FeedbackResponse,
  BusinessMetricsResponse,
  LoginRequest,
  SignupRequest,
  TokenResponse,
} from "../types";
import {
  mockPredict,
  mockHealth,
  mockMonitoringData,
  mockCardsCatalog,
  mockRecommendPortfolio,
  mockRecommendTransaction,
  mockTransactions,
  mockSummary,
  mockFeedback,
  mockBusinessMetrics,
} from "./mock";

const API_BASE_URL = import.meta.env.VITE_API_URL || "";
const USE_MOCK = !API_BASE_URL; // Set to `true` to force mock data for development

const TOKEN_KEY = "rs_token";

async function buildApiError(res: Response): Promise<Error> {
  let message = `API error: ${res.status}`;
  try {
    const data = (await res.json()) as { detail?: string };
    if (data?.detail) {
      message = data.detail;
    }
  } catch {
    // Ignore JSON parse failures and keep generic fallback message.
  }
  return new Error(message);
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function handleUnauthorized(status: number): void {
  if (status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem("rs_user");
    window.location.href = "/login";
  }
}

export async function predict(
  request: PredictionRequest,
): Promise<PredictionResponse> {
  if (USE_MOCK) return mockPredict(request);

  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(request),
  });
  handleUnauthorized(res.status);
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

  const res = await fetch(`${API_BASE_URL}/health`, {
    headers: authHeaders(),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getMe(): Promise<UserProfile> {
  const res = await fetch(`${API_BASE_URL}/me`, {
    headers: authHeaders(),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json() as Promise<UserProfile>;
}

export async function updateProfile(patch: ProfilePatch): Promise<UserProfile> {
  const res = await fetch(`${API_BASE_URL}/me/profile`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(patch),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json() as Promise<UserProfile>;
}

export async function updateSavedCards(
  cardIds: string[],
): Promise<UserProfile> {
  const res = await fetch(`${API_BASE_URL}/me/cards`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ card_ids: cardIds }),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json() as Promise<UserProfile>;
}

export async function getCardCatalog(): Promise<CardCatalogItem[]> {
  if (USE_MOCK) return mockCardsCatalog();

  const res = await fetch(`${API_BASE_URL}/cards/catalog`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json() as Promise<CardCatalogItem[]>;
}

export async function getMonitoringData(): Promise<MonitoringData> {
  if (USE_MOCK) return mockMonitoringData();

  const res = await fetch(`${API_BASE_URL}/monitoring`, {
    headers: authHeaders(),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function signup(payload: SignupRequest): Promise<TokenResponse> {
  if (USE_MOCK) {
    return {
      access_token: "mock-token",
      token_type: "bearer",
      user_id: 1,
      display_name: payload.display_name,
    };
  }
  const res = await fetch(`${API_BASE_URL}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await buildApiError(res);
  return res.json() as Promise<TokenResponse>;
}

export async function login(payload: LoginRequest): Promise<TokenResponse> {
  if (USE_MOCK) {
    return {
      access_token: "mock-token",
      token_type: "bearer",
      user_id: 1,
      display_name: "Demo User",
    };
  }
  const res = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await buildApiError(res);
  return res.json() as Promise<TokenResponse>;
}

export async function logout(): Promise<void> {
  if (USE_MOCK) return;
  await fetch(`${API_BASE_URL}/auth/logout`, {
    method: "POST",
    headers: authHeaders(),
  });
}

export async function recommendPortfolio(
  payload: PortfolioRecommendRequest,
): Promise<PersonaRecommendResponse> {
  if (USE_MOCK) return mockRecommendPortfolio(payload);
  const res = await fetch(`${API_BASE_URL}/recommendations/portfolio`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json() as Promise<PersonaRecommendResponse>;
}

export async function recommendTransaction(
  payload: TransactionRecommendRequest,
): Promise<PersonaRecommendResponse> {
  if (USE_MOCK) return mockRecommendTransaction(payload);
  const res = await fetch(`${API_BASE_URL}/recommendations/transaction`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json() as Promise<PersonaRecommendResponse>;
}

export async function getTransactions(
  page = 1,
  pageSize = 20,
): Promise<TransactionsResponse> {
  if (USE_MOCK) return mockTransactions(page, pageSize);

  const res = await fetch(
    `${API_BASE_URL}/transactions?page=${page}&page_size=${pageSize}`,
    { headers: authHeaders() },
  );
  handleUnauthorized(res.status);
  if (!res.ok) throw await buildApiError(res);
  return res.json() as Promise<TransactionsResponse>;
}

export async function createTransaction(
  payload: TransactionCreateRequest,
): Promise<TransactionLogEntry> {
  const res = await fetch(`${API_BASE_URL}/transactions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw await buildApiError(res);
  return res.json() as Promise<TransactionLogEntry>;
}

export async function exportTransactions(
  format: "csv" | "xlsx",
): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/transactions/export?format=${format}`,
    { headers: authHeaders() },
  );
  handleUnauthorized(res.status);
  if (!res.ok) throw await buildApiError(res);

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `rewardsense_transactions.${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function getSummary(): Promise<SummaryResponse> {
  if (USE_MOCK) return mockSummary();

  const res = await fetch(`${API_BASE_URL}/summary`, {
    headers: authHeaders(),
  });
  handleUnauthorized(res.status);
  if (!res.ok) throw await buildApiError(res);
  return res.json() as Promise<SummaryResponse>;
}

export async function submitFeedback(
  payload: FeedbackRequest,
): Promise<FeedbackResponse> {
  return mockFeedback(payload);
}

export async function getBusinessMetrics(): Promise<BusinessMetricsResponse> {
  return mockBusinessMetrics();
}
