export interface SpendingCategories {
  groceries?: number;
  dining?: number;
  travel?: number;
  gas?: number;
  online_shopping?: number;
  entertainment?: number;
  utilities?: number;
  other?: number;
}

export interface Transaction {
  merchant_category: string;
  amount: number;
  date: string;
}

export interface PredictionRequest {
  user_id: string;
  spending_categories: SpendingCategories;
  monthly_spend: number;
  preferred_rewards: string[];
  transaction_history?: Transaction[];
}

export interface ScoreBreakdown {
  deterministic: number;
  personalization: number;
}

export interface RecommendedCard {
  card_name: string;
  issuer: string;
  score: number;
  rank: number;
  explanation: string;
  annual_fee: number;
  reward_rate: number;
  key_benefits: string[];
  deterministic_score?: number;
  personalization_score?: number;
  score_breakdown: ScoreBreakdown;
}

export interface PredictionResponse {
  recommended_cards: RecommendedCard[];
  model_version: string;
  inference_latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model_version: string;
  uptime_seconds: number;
}

export interface FeatureDrift {
  [feature: string]: number;
}

export interface DriftCheck {
  detected: boolean;
  timestamp: string;
  feature_drift: FeatureDrift;
}

export interface ServingMetrics {
  request_count: number;
  avg_latency_ms: number;
  error_rate: number;
  p95_latency_ms: number;
}

export interface RetrainEvent {
  timestamp: string;
  trigger_reason: string;
  model_version: string;
  status: "success" | "failed" | "in_progress";
}

export interface MonitoringData {
  model_version: string;
  last_deployment_time: string;
  drift_check: DriftCheck;
  serving_metrics: ServingMetrics;
  retrain_history: RetrainEvent[];
}

// ---------------------------------------------------------------------------
// User profile & catalog types (Story 1.2)
// ---------------------------------------------------------------------------

export interface UserProfile {
  user_id: number;
  email: string;
  display_name: string;
  personas: string[];
  reward_preference: string;
  transaction_logging_enabled: boolean;
  dark_mode: boolean;
  saved_card_ids: string[];
}

export interface ProfilePatch {
  display_name?: string;
  personas?: string[];
  reward_preference?: string;
  transaction_logging_enabled?: boolean;
  dark_mode?: boolean;
}

export interface CardCatalogItem {
  card_id: string;
  card_name: string;
  issuer: string;
  annual_fee: number;
  reward_highlights: string[];
  image_url: string | null;
}

export interface SignupRequest {
  email: string;
  password: string;
  display_name: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  user_id: number;
  display_name: string;
}

export interface CardListRequest {
  card_ids: string[];
}

export interface ScoredCard {
  card_id?: string;
  card_name: string;
  reward_amount: number;
  annual_fee: number;
  rank: number;
  persona_adjustments?: Record<string, unknown>;
}

export interface PortfolioRecommendRequest {
  spending_categories: Record<string, number>;
  monthly_spend?: number;
  use_full_catalog?: boolean;
}

export interface TransactionRecommendRequest {
  merchant: string;
  amount: number;
  category?: string;
}

export interface PersonaRecommendResponse {
  ranked: ScoredCard[];
  best_card_id?: string;
  is_personalized: boolean;
  is_generic: boolean;
  active_personas: string[];
  persona_context: string;
}

export interface TransactionLogEntry {
  id: string;
  merchant: string;
  category: string;
  amount: number;
  card_id: string;
  reward_earned: number;
  estimated_savings: number;
  baseline_savings: number;
  timestamp: string;
  source_flow: "quick_transaction" | "portfolio_recommendation";
}

export interface TransactionsResponse {
  items: TransactionLogEntry[];
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
}

export interface TransactionsExportResponse {
  format: "csv" | "xlsx";
  download_url: string;
}

export interface SummaryResponse {
  spend_by_category: Array<{ category: string; amount: number }>;
  rewards_by_category: Array<{ category: string; reward_earned: number }>;
  savings_by_card: Array<{ card_id: string; card_name: string; savings: number }>;
  fee_adjusted_savings_total: number;
}

export interface FeedbackRequest {
  recommendation_event_id?: string;
  card_id?: string;
  reaction: "like" | "dislike";
  reason_tag?: string;
}

export interface FeedbackResponse {
  ok: boolean;
  feedback_id: string;
}

export interface BusinessMetricsResponse {
  generated_at: string;
  report_url_html: string;
  report_url_pdf: string;
  total_requests: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  estimated_llm_cost_usd: number;
  fallback_rate: number;
  error_rate: number;
}
