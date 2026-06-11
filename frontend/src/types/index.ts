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
  pros?: string[];
  cons?: string[];
  best_for?: string;
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
  active_personas?: string[];
  persona_context?: string;
  recommended_for_label?: string;
  recommended_for_emoji?: string;
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

export interface CardDisplayInfo {
  card_id: string;
  card_name: string;
  issuer: string;
  annual_fee: number;
  reward_highlights: string[];
  image_url: string | null;
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

export interface ScoredCardScoreBreakdown {
  raw_reward_rate: number;
  raw_reward_amount: number;
  personalization_multiplier: number;
  persona_category_boost: number;
  persona_fee_penalty: number;
}

export interface ScoredCardDisplay {
  card_id: string;
  card_name: string;
  issuer: string;
  annual_fee: number;
  reward_highlights: string[];
  image_url?: string | null;
}

export interface ScoredCard {
  card_id?: string;
  card_name: string;
  reward_amount: number;
  annual_fee: number;
  rank: number;
  explanation?: string;
  pros?: string[];
  cons?: string[];
  best_for?: string;
  persona_adjustments?: Record<string, unknown>;
  score_breakdown?: ScoredCardScoreBreakdown;
  persona_match_reason?: string;
  projected_savings?: number;
  card_display?: ScoredCardDisplay;
}

export interface PortfolioRecommendRequest {
  spending_categories: Record<string, number>;
  monthly_spend?: number;
  use_full_catalog?: boolean;
  personas?: string[];
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
  id: number;
  merchant: string;
  amount: number;
  category: string;
  chosen_card_id: string | null;
  chosen_card_name: string | null;
  reward_earned: number;
  estimated_savings: number;
  source_flow: string;
  card_was_saved: boolean;
  recommendation_event_id: number | null;
  timestamp: string;
}

export interface TransactionsResponse {
  transactions: TransactionLogEntry[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  /** Sum of reward_earned across all transactions (omitted by older API versions). */
  total_rewards?: number;
  /** Sum of estimated_savings across all transactions (omitted by older API versions). */
  total_savings?: number;
}

export interface TransactionsExportResponse {
  format: "csv" | "xlsx";
  download_url: string;
}

export interface CategorySummaryItem {
  category: string;
  total_spend: number;
  total_reward: number;
  total_savings: number;
  transaction_count: number;
}

export interface CardSummaryItem {
  card_id: string | null;
  card_name: string | null;
  total_spend: number;
  total_reward: number;
  total_savings: number;
  transaction_count: number;
}

export interface TopInsight {
  label: string;
  value: string;
}

export interface SummaryResponse {
  spend_by_category: CategorySummaryItem[];
  rewards_by_category: CategorySummaryItem[];
  savings_by_card: CardSummaryItem[];
  total_spend: number;
  total_rewards: number;
  total_savings: number;
  fee_adjusted_savings: number;
  transaction_count: number;
  top_insights: TopInsight[];
}

export type FeedbackReasonTag =
  | "too_expensive"
  | "not_relevant"
  | "already_have"
  | "explanation_unclear";

export interface FeedbackRequest {
  recommendation_event_id?: number;
  card_id: string;
  reaction: "like" | "dislike";
  reason_tag?: FeedbackReasonTag;
  target: "card" | "explanation";
}

export interface FeedbackResponse {
  ok: boolean;
  feedback_id: number;
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

export interface TransactionCreateRequest {
  merchant: string;
  amount: number;
  category: string;
  chosen_card_id?: string;
  chosen_card_name?: string;
  reward_earned?: number;
  estimated_savings?: number;
  source_flow?: string;
  recommendation_event_id?: number;
  timestamp?: string;
}

export interface QuickTransactionRequest {
  merchant: string;
  amount: number;
  category?: string;
  date?: string;
}

export interface QuickTransactionResponse {
  top_card: ScoredCard | null;
  alternatives: ScoredCard[];
  estimated_reward: number;
  money_saved: number;
  category_used: string;
  is_personalized: boolean;
  has_saved_cards: boolean;
  active_personas: string[];
  persona_context: string;
}