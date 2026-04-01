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
