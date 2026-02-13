// ============================================
// Core Types
// ============================================

export type FilterGrade = "A" | "B" | "C" | "D" | "E" | "F";

// ============================================
// Chat Types
// ============================================

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  grade?: FilterGrade;
  metadata?: MessageMetadata;
}

export interface MessageMetadata {
  feature?: string;
  tokens?: number;
  responseTime?: number;
  question_type?: string;
  handled_by?: string;
  classification_method?: string;
  response_time_ms?: number;
  trust?: { score: number; hallucination_risk?: number };
  quality?: { score: number; level?: string };
  relevance?: { score: number };
  validation?: {
    is_valid?: boolean;
    grade: string;
    score: number;
    status: string;
    retries: number;
  };
}

export interface ChatRequest {
  brand_id: string;
  message: string;
  session_id?: string;
  context?: Record<string, unknown>;
}

export interface ChatResponse {
  brand_id: string;
  message: string;
  question_type?: string;
  retrieval_contexts?: RetrievalContext[];
  metadata: MessageMetadata & Record<string, unknown>;
}

export interface ChatWithSessionRequest {
  brand_id: string;
  message: string;
  session_id?: string;
  user_id?: string;
  use_history?: boolean;
  question_type?: string;
}

export interface ChatWithSessionResponse {
  brand_id: string;
  message: string;
  session_id: string;
  question_type?: string;
  filter_grade?: FilterGrade;
  metadata: MessageMetadata & Record<string, unknown>;
}

export interface RetrievalContext {
  source: string;
  data: unknown[];
  metadata?: Record<string, unknown>;
}

// ============================================
// Brand Types
// ============================================

export interface Brand {
  id: string;
  name: string;
  description?: string;
  industry?: string;
  features: string[];
  created_at?: string;
}

export interface BrandStats {
  brand_id: string;
  total_conversations: number;
  total_messages: number;
  avg_response_time_ms: number;
  satisfaction_rate: number;
  top_questions: string[];
  active_users: number;
  total_documents?: number;
  avg_quality_score?: number;
  // Neo4j graph data
  total_nodes?: number;
  nodes?: Record<string, number>;
  total_relationships?: number;
  relationships?: Record<string, number>;
}

export interface BrandCreate {
  id: string;
  name: string;
  description?: string;
  industry?: string;
  features?: string[];
  neo4j_brand_id?: string;
  neo4j_namespaces?: string[];
}

// ============================================
// Health & System Types
// ============================================

export interface SystemHealth {
  status: "healthy" | "degraded" | "unhealthy";
  version?: string;
  llm_model?: string;
  features?: string[];
  services: {
    neo4j: Neo4jHealth;
    cache: CacheHealth;
    llm: LlmHealth;
  };
}

export interface Neo4jHealth {
  status: "healthy" | "unhealthy";
  state?: string;
  latency_ms?: number;
  database?: string;
  vector_available?: boolean;
  metrics?: {
    total_queries: number;
    read_queries: number;
    write_queries: number;
    vector_queries: number;
    batch_operations: number;
    avg_query_time_ms: number;
    error_count: number;
    uptime_seconds: number;
  };
}

export interface CacheHealth {
  status: "healthy" | "unavailable" | "unhealthy";
}

export interface LlmHealth {
  status: "healthy" | "unhealthy";
  model?: string;
}

// ============================================
// Alert Types
// ============================================

export interface Alert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  title: string;
  message: string;
  brand_id?: string;
  data?: Record<string, unknown>;
  created_at: string;
  acknowledged: boolean;
  acknowledged_at?: string;
  acknowledged_by?: string;
}

export type AlertType =
  | "system"
  | "brand"
  | "conversation"
  | "performance"
  | "security";

export type AlertSeverity = "info" | "warning" | "error" | "critical";

export interface AlertRule {
  id: string;
  name: string;
  description?: string;
  type: AlertType;
  condition: AlertCondition;
  severity: AlertSeverity;
  enabled: boolean;
  notify_channels: string[];
  created_at: string;
}

export interface AlertCondition {
  metric: string;
  operator: "gt" | "lt" | "eq" | "gte" | "lte";
  threshold: number;
  duration_seconds?: number;
}

export interface AlertsResponse {
  alerts: Alert[];
  total: number;
  unacknowledged: number;
}

// ============================================
// Analytics Types
// ============================================

export interface AnalyticsOverview {
  period: string;
  total_conversations: number;
  total_messages: number;
  unique_users: number;
  avg_response_time_ms: number;
  satisfaction_rate: number;
  top_brands: BrandMetric[];
  hourly_activity: HourlyActivity[];
  feature_usage?: Record<string, number>;
}

export interface BrandMetric {
  brand_id: string;
  brand_name: string;
  conversations: number;
  messages: number;
  satisfaction_rate: number;
}

export interface HourlyActivity {
  hour: number;
  conversations: number;
  messages: number;
}

export interface ConversationAnalytics {
  period: string;
  data: ConversationDataPoint[];
  summary: {
    total: number;
    trend: number;
    avg_duration_minutes: number;
    completion_rate: number;
  };
}

export interface ConversationDataPoint {
  date: string;
  conversations: number;
  messages: number;
  unique_users: number;
}

export interface PerformanceAnalytics {
  period: string;
  avg_response_time_ms: number;
  p50_response_time_ms: number;
  p95_response_time_ms: number;
  p99_response_time_ms: number;
  error_rate: number;
  throughput_per_minute: number;
  data: PerformanceDataPoint[];
}

export interface PerformanceDataPoint {
  timestamp: string;
  response_time_ms: number;
  error_rate: number;
  throughput: number;
}

// ============================================
// Pipeline Types
// ============================================

export type PipelinePlatform =
  | "instagram"
  | "youtube"
  | "tiktok"
  | "twitter"
  | "website";
export type PipelineJobStatus = "pending" | "running" | "completed" | "failed";
export type PipelineTargetType = "accounts" | "hashtags" | "search" | "urls";

export interface PipelineRunRequest {
  brand_id: string;
  platform: PipelinePlatform;
  max_items: number;
  target_type: PipelineTargetType;
  targets: string[];
  dry_run?: boolean;
  skip_crawl?: boolean;
  skip_llm?: boolean;
}

// ============================================
// Session Types
// ============================================

export interface ChatSession {
  id: string;
  brand_id: string;
  user_id?: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  metadata?: Record<string, unknown>;
}

export interface SessionMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface ChatSessionResponse {
  id: string;
  brand_id: string;
  user_id?: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  metadata?: Record<string, unknown>;
}

export interface ChatMessageResponse {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

// ============================================
// Social Monitoring Types
// ============================================

export interface SocialMonitoringData {
  brand_id: string;
  mentions: number;
  sentiment: {
    positive: number;
    neutral: number;
    negative: number;
  };
  platforms: Record<string, number>;
  recent_mentions: SocialMention[];
  trending_topics: TrendingTopic[];
  total_engagement: number;
  period_days: number;
}

export interface SocialMention {
  id: string;
  platform: string;
  content: string;
  sentiment: string;
  timestamp: string;
  likes: number;
  comments: number;
  author: string;
}

export interface TrendingTopic {
  topic: string;
  count: number;
  change: number;
}

// ============================================
// Web Crawler Types
// ============================================

export type CrawlerType = "PLAYWRIGHT_ADAPTIVE" | "SIMPLE" | "DEEP";

export interface StartCrawlRequest {
  urls: string[];
  brand_id?: string;
  max_pages?: number;
  max_depth?: number;
  crawler_type?: CrawlerType;
  include_patterns?: string[];
  exclude_patterns?: string[];
  save_markdown?: boolean;
  wait_for_finish?: boolean;
}

export interface CrawlJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  run_id?: string;
  urls: string[];
  max_pages: number;
  message?: string;
  results?: CrawlResult[];
}

export interface CrawlResult {
  url: string;
  title?: string;
  text_preview?: string;
  markdown_length?: number;
  crawled_at: string;
}

// ============================================
// Analytics Extended Types
// ============================================

export interface BrandAnalytics {
  brand_id: string;
  period_days: number;
  total_sessions: number;
  total_messages: number;
  avg_messages_per_session: number;
  grade_distribution: Record<FilterGrade, number>;
  daily_stats: DailyStat[];
  top_questions: TopQuestion[];
}

export interface DailyStat {
  date: string;
  sessions: number;
  messages: number;
  engagement?: number;
}

export interface TopQuestion {
  question: string;
  count: number;
  category?: string;
}

export interface ContentPerformance {
  brand_id: string;
  content: Array<{
    id: string;
    title: string;
    platform: string;
    engagement_rate: number;
    likes: number;
    comments: number;
    shares: number;
  }>;
}

export interface EngagementTrend {
  date: string;
  posts: number;
  likes: number;
  comments: number;
  shares: number;
}

// ============================================
// Products Types
// ============================================

export interface Product {
  id: string;
  name: string;
  category: string;
  price: number;
  stock: number;
  sales: number;
  rating: number;
  image_url?: string;
}

export interface ProductsResponse {
  brand_id: string;
  products: Product[];
  stats: {
    total_products: number;
    active_products: number;
    total_sales: number;
    avg_price: number;
  };
  categories: Array<{ category: string; count: number }>;
  pagination: {
    limit: number;
    offset: number;
    total: number;
  };
}

export interface ProductRecommendation {
  product_id: string;
  name: string;
  score: number;
  reason: string;
  category: string;
}

export interface PipelineRunResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface PipelineJob {
  job_id: string;
  status: PipelineJobStatus;
  brand_id: string;
  platform: PipelinePlatform;
  target_type?: PipelineTargetType;
  target_value?: string;
  targets?: string[];
  max_items: number;
  created_at: string;
  progress?: number;
  statistics?: PipelineStatistics;
  error?: string;
}

export interface PipelineStatistics {
  brand_id: string;
  platform: string;
  started_at: string;
  completed_at?: string;
  duration_seconds: number;
  crawled_count: number;
  transformed_count: number;
  filtered_count: number;
  processed_count: number;
  saved_nodes: number;
  saved_relationships: number;
  errors: string[];
  success: boolean;
}

export interface PipelineLogEntry {
  timestamp: string;
  level: "info" | "warning" | "error";
  message: string;
}

export interface PipelineLogsResponse {
  job_id: string;
  status: string;
  progress?: string;
  logs: PipelineLogEntry[];
}

// ============================================
// Auth Types
// ============================================

export interface User {
  id: string;
  email: string;
  name: string;
  role: "super_admin" | "client_admin" | "viewer";
  brand_ids: string[];
  is_active: boolean;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

// ============================================
// Feature Types
// ============================================

export interface Feature {
  name: string;
  description: string;
  enabled: boolean;
  config?: Record<string, unknown>;
}

// ============================================
// On-Chain Intelligence Types
// ============================================

export * from "./onchain";
