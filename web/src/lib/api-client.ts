import type {
  Brand,
  BrandStats,
  BrandCreate,
  ChatRequest,
  ChatResponse,
  ChatWithSessionRequest,
  ChatWithSessionResponse,
  ChatSessionResponse,
  ChatMessageResponse,
  SystemHealth,
  Alert,
  AlertRule,
  AlertsResponse,
  AnalyticsOverview,
  ConversationAnalytics,
  PerformanceAnalytics,
  PipelineRunRequest,
  PipelineRunResponse,
  PipelineJob,
  LoginRequest,
  LoginResponse,
  User,
} from "@/types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const PIPELINE_API_URL =
  process.env.NEXT_PUBLIC_PIPELINE_API_URL || API_BASE_URL;

const ONCHAIN_API_URL =
  process.env.NEXT_PUBLIC_ONCHAIN_API_URL || PIPELINE_API_URL;

// ============================================
// Token Management
// ============================================

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("ontix_token");
}

export function setToken(token: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem("ontix_token", token);
}

export function removeToken(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem("ontix_token");
}

// ============================================
// API Client Class
// ============================================

class ApiClient {
  private baseUrl: string;
  private pipelineUrl: string;
  private onchainUrl: string;

  constructor(baseUrl: string, pipelineUrl: string, onchainUrl: string) {
    this.baseUrl = baseUrl;
    this.pipelineUrl = pipelineUrl;
    this.onchainUrl = onchainUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    requiresAuth: boolean = false
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (requiresAuth) {
      const token = getToken();
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (response.status === 401 && requiresAuth) {
      removeToken();
      throw new Error("Unauthorized");
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      let errorMessage: string;

      if (typeof error.detail === 'string') {
        errorMessage = error.detail;
      } else if (Array.isArray(error.detail)) {
        // Handle Pydantic validation errors (array of error objects)
        errorMessage = error.detail
          .map((e: { msg?: string; loc?: string[] }) =>
            e.msg || JSON.stringify(e)
          )
          .join(', ');
      } else if (error.detail?.message) {
        errorMessage = error.detail.message;
      } else if (error.message) {
        errorMessage = error.message;
      } else {
        errorMessage = `API Error: ${response.status}`;
      }

      console.error(`[API Error] ${endpoint}: ${response.status} - ${errorMessage}`);
      throw new Error(errorMessage);
    }

    return response.json();
  }

  private async pipelineRequest<T>(
    endpoint: string,
    options: RequestInit = {},
    requiresAuth: boolean = false
  ): Promise<T> {
    const url = `${this.pipelineUrl}${endpoint}`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (requiresAuth) {
      const token = getToken();
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (response.status === 401 && requiresAuth) {
      removeToken();
      throw new Error("Unauthorized");
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      let errorMessage: string;

      if (typeof error.detail === 'string') {
        errorMessage = error.detail;
      } else if (Array.isArray(error.detail)) {
        errorMessage = error.detail
          .map((e: { msg?: string; loc?: string[] }) =>
            e.msg || JSON.stringify(e)
          )
          .join(', ');
      } else if (error.detail?.message) {
        errorMessage = error.detail.message;
      } else if (error.message) {
        errorMessage = error.message;
      } else {
        errorMessage = `API Error: ${response.status}`;
      }

      console.error(`[Pipeline API Error] ${endpoint}: ${response.status} - ${errorMessage}`);
      throw new Error(errorMessage);
    }

    return response.json();
  }

  private async onchainRequest<T>(
    endpoint: string,
    options: RequestInit = {},
    requiresAuth: boolean = false
  ): Promise<T> {
    const url = `${this.onchainUrl}${endpoint}`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (requiresAuth) {
      const token = getToken();
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (response.status === 401 && requiresAuth) {
      removeToken();
      throw new Error("Unauthorized");
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      let errorMessage: string;

      if (typeof error.detail === 'string') {
        errorMessage = error.detail;
      } else if (Array.isArray(error.detail)) {
        errorMessage = error.detail
          .map((e: { msg?: string; loc?: string[] }) =>
            e.msg || JSON.stringify(e)
          )
          .join(', ');
      } else if (error.detail?.message) {
        errorMessage = error.detail.message;
      } else if (error.message) {
        errorMessage = error.message;
      } else {
        errorMessage = `API Error: ${response.status}`;
      }

      console.error(`[OnChain API Error] ${endpoint}: ${response.status} - ${errorMessage}`);
      throw new Error(errorMessage);
    }

    return response.json();
  }

  // ============================================
  // Auth API
  // ============================================

  async login(data: LoginRequest): Promise<LoginResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Login failed");
    }

    return response.json();
  }

  async getCurrentUser(): Promise<User> {
    return this.request<User>("/api/v1/auth/me", {}, true);
  }

  async updateProfile(data: { email?: string; name?: string }): Promise<User> {
    return this.request<User>(
      "/api/v1/auth/me",
      {
        method: "PATCH",
        body: JSON.stringify(data),
      },
      true
    );
  }

  async changePassword(data: { current_password: string; new_password: string }): Promise<{ message: string }> {
    return this.request<{ message: string }>(
      "/api/v1/auth/me/password",
      {
        method: "POST",
        body: JSON.stringify(data),
      },
      true
    );
  }

  // ============================================
  // Chat API
  // ============================================

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async sendMessageWithSession(
    request: ChatWithSessionRequest
  ): Promise<ChatWithSessionResponse> {
    return this.request<ChatWithSessionResponse>(
      "/api/v1/chat/with-session",
      {
        method: "POST",
        body: JSON.stringify(request),
      }
    );
  }

  // ============================================
  // AI Advisor API (Dedicated Endpoint)
  // ============================================

  async sendAdvisorMessage(request: {
    brand_id: string;
    message: string;
    session_id?: string;
  }): Promise<{
    brand_id: string;
    message: string;
    session_id?: string;
    metadata: Record<string, unknown>;
  }> {
    return this.request("/api/v1/advisor", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async sendAdvisorMessageWithSession(request: {
    brand_id: string;
    message: string;
    session_id?: string;
  }): Promise<{
    brand_id: string;
    message: string;
    session_id: string;
    metadata: Record<string, unknown>;
  }> {
    return this.request("/api/v1/advisor/with-session", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  // Streaming chat
  async *streamMessage(
    request: ChatWithSessionRequest
  ): AsyncGenerator<string, void, unknown> {
    const response = await fetch(`${this.baseUrl}/api/v1/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Stream error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") return;
          try {
            const parsed = JSON.parse(data);
            if (parsed.content) {
              yield parsed.content;
            }
          } catch {
            // Skip non-JSON lines
          }
        }
      }
    }
  }

  // ============================================
  // Brand API
  // ============================================

  async getBrands(): Promise<Brand[]> {
    return this.request<Brand[]>("/api/v1/brands");
  }

  async getBrand(brandId: string): Promise<Brand> {
    return this.request<Brand>(`/api/v1/brands/${brandId}`);
  }

  async createBrand(data: BrandCreate): Promise<Brand> {
    return this.request<Brand>(
      "/api/v1/brands",
      {
        method: "POST",
        body: JSON.stringify(data),
      },
      true
    );
  }

  async getBrandStats(brandId: string): Promise<BrandStats> {
    return this.request<BrandStats>(`/api/v1/brands/${brandId}/stats`, {}, false);
  }

  async updateBrand(brandId: string, data: Partial<BrandCreate>): Promise<Brand> {
    return this.request<Brand>(
      `/api/v1/brands/${brandId}`,
      {
        method: "PATCH",
        body: JSON.stringify(data),
      },
      true
    );
  }

  async deleteBrand(brandId: string): Promise<void> {
    await this.request(
      `/api/v1/brands/${brandId}`,
      { method: "DELETE" },
      true
    );
  }

  async validateBrand(brandId: string): Promise<{ valid: boolean; errors: string[] }> {
    return this.request(`/api/v1/brands/${brandId}/validate`, {}, true);
  }

  // ============================================
  // Neo4j Data API
  // ============================================

  async getNeo4jStats(brandId: string): Promise<{
    total_nodes: number;
    nodes: Record<string, number>;
    relationships: number;
  }> {
    return this.request(`/api/v1/brands/${brandId}/stats`, {}, true);
  }

  // ============================================
  // Social Monitoring API
  // ============================================

  async getSocialSummary(brandId: string): Promise<{
    total_mentions: number;
    sentiment: { positive: number; neutral: number; negative: number };
    platforms: Record<string, number>;
    engagement: { likes: number; comments: number; shares: number };
  }> {
    return this.request(`/api/v1/social/${brandId}`, {}, true);
  }

  async getSocialMentions(brandId: string, params?: {
    platform?: string;
    sentiment?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    mentions: Array<{
      id: string;
      platform: string;
      content: string;
      sentiment: string;
      engagement: number;
      created_at: string;
      author?: string;
      timestamp?: string;
      likes?: number;
      comments?: number;
    }>;
    total: number;
  }> {
    const searchParams = new URLSearchParams();
    if (params?.platform) searchParams.set("platform", params.platform);
    if (params?.sentiment) searchParams.set("sentiment", params.sentiment);
    if (params?.limit) searchParams.set("limit", String(params.limit));
    if (params?.offset) searchParams.set("offset", String(params.offset));
    const query = searchParams.toString();
    return this.request(`/api/v1/social/${brandId}/mentions${query ? `?${query}` : ""}`, {}, true);
  }

  async getTrendingTopics(brandId: string): Promise<{
    topics: Array<{ topic: string; count: number; sentiment: number }>;
  }> {
    return this.request(`/api/v1/social/${brandId}/trending`, {}, true);
  }

  // ============================================
  // Chat Sessions API
  // ============================================

  async getChatSessions(brandId: string, params?: {
    limit?: number;
    offset?: number;
  }): Promise<{
    sessions: Array<{
      session_id: string;
      user_id?: string;
      message_count: number;
      created_at: string;
      last_message_at: string;
    }>;
    total: number;
  }> {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set("limit", String(params.limit));
    if (params?.offset) searchParams.set("offset", String(params.offset));
    const query = searchParams.toString();
    return this.request(`/api/v1/analytics/${brandId}/sessions${query ? `?${query}` : ""}`, {}, true);
  }

  async getSessionMessages(brandId: string, sessionId: string): Promise<{
    messages: Array<{
      id: string;
      role: "user" | "assistant";
      content: string;
      grade?: string;
      created_at: string;
    }>;
  }> {
    return this.request(`/api/v1/analytics/${brandId}/sessions/${sessionId}/messages`, {}, true);
  }

  // ============================================
  // Health API
  // ============================================

  async getHealth(): Promise<SystemHealth> {
    return this.request<SystemHealth>("/health");
  }

  // ============================================
  // Features API
  // ============================================

  async getFeatures(): Promise<string[]> {
    return this.request<string[]>("/api/v1/features");
  }

  // ============================================
  // Alerts API
  // ============================================

  async getAlerts(params?: {
    acknowledged?: boolean;
    severity?: string;
    limit?: number;
  }): Promise<AlertsResponse> {
    const searchParams = new URLSearchParams();
    if (params?.acknowledged !== undefined) {
      searchParams.set("acknowledged", String(params.acknowledged));
    }
    if (params?.severity) {
      searchParams.set("severity", params.severity);
    }
    if (params?.limit) {
      searchParams.set("limit", String(params.limit));
    }

    const query = searchParams.toString();
    return this.request<AlertsResponse>(
      `/api/v1/alerts${query ? `?${query}` : ""}`,
      {},
      true
    );
  }

  async acknowledgeAlert(alertId: string): Promise<Alert> {
    return this.request<Alert>(
      `/api/v1/alerts/${alertId}/acknowledge`,
      { method: "POST" },
      true
    );
  }

  async getAlertRules(): Promise<AlertRule[]> {
    return this.request<AlertRule[]>("/api/v1/alerts/rules", {}, true);
  }

  async createAlertRule(rule: Omit<AlertRule, "id" | "created_at">): Promise<AlertRule> {
    return this.request<AlertRule>(
      "/api/v1/alerts/rules",
      {
        method: "POST",
        body: JSON.stringify(rule),
      },
      true
    );
  }

  async updateAlertRule(
    ruleId: string,
    rule: Partial<AlertRule>
  ): Promise<AlertRule> {
    return this.request<AlertRule>(
      `/api/v1/alerts/rules/${ruleId}`,
      {
        method: "PATCH",
        body: JSON.stringify(rule),
      },
      true
    );
  }

  async deleteAlertRule(ruleId: string): Promise<void> {
    await this.request(
      `/api/v1/alerts/rules/${ruleId}`,
      { method: "DELETE" },
      true
    );
  }

  // ============================================
  // Analytics API
  // ============================================

  async getAnalyticsOverview(period: string = "7d"): Promise<AnalyticsOverview> {
    // period를 days로 변환 (7d -> 7)
    const days = parseInt(period.replace("d", "")) || 7;

    // 모든 브랜드 가져와서 첫번째 브랜드 사용 (임시)
    // TODO: 모든 브랜드 집계 또는 대시보드용 별도 API 필요
    const brands = await this.getBrands();
    if (brands.length === 0) {
      return {
        period: period,
        total_conversations: 0,
        total_messages: 0,
        unique_users: 0,
        avg_response_time_ms: 0,
        satisfaction_rate: 0,
        top_brands: [],
        hourly_activity: []
      };
    }

    // 첫번째 브랜드 데이터 가져오기
    const analytics = await this.request<any>(
      `/api/v1/analytics/${brands[0].id}?days=${days}`,
      {},
      false
    );

    return {
      period: period,
      total_conversations: analytics.total_sessions || 0,
      total_messages: analytics.total_messages || 0,
      unique_users: analytics.total_messages || 0,
      avg_response_time_ms: 0,
      satisfaction_rate: 0,
      top_brands: [],
      hourly_activity: analytics.daily_stats?.map((d: any) => ({
        hour: new Date(d.date).getHours(),
        conversations: d.sessions || 0
      })) || []
    };
  }

  async getConversationAnalytics(
    period: string = "7d",
    brandId?: string
  ): Promise<ConversationAnalytics> {
    const days = parseInt(period.replace("d", "").replace("h", "")) || 7;

    if (!brandId) {
      const brands = await this.getBrands();
      brandId = brands[0]?.id;
    }

    if (!brandId) {
      return { period: period, data: [], summary: { total: 0, trend: 0, avg_duration_minutes: 0, completion_rate: 0 } };
    }

    const analytics = await this.request<any>(
      `/api/v1/analytics/${brandId}?days=${days}`,
      {},
      false
    );

    return {
      period: period,
      data: analytics.daily_stats?.map((d: any) => ({
        date: d.date,
        conversations: d.sessions || 0,
        messages: d.messages || 0,
        unique_users: 0
      })) || [],
      summary: {
        total: analytics.total_sessions || 0,
        trend: 0,
        avg_duration_minutes: 0,
        completion_rate: 100
      }
    };
  }

  async getPerformanceAnalytics(
    period: string = "24h"
  ): Promise<PerformanceAnalytics> {
    // 성능 데이터는 별도 API가 없으므로 기본값 반환
    return {
      period: period,
      avg_response_time_ms: 0,
      p50_response_time_ms: 0,
      p95_response_time_ms: 0,
      p99_response_time_ms: 0,
      error_rate: 0,
      throughput_per_minute: 0,
      data: []
    };
  }

  // ============================================
  // Pipeline Control API (EC2 Management)
  // ============================================

  async getPipelineServerStatus(): Promise<{
    instance_id: string;
    state: string;
    public_ip: string | null;
    pipeline_url: string | null;
    ready: boolean;
  }> {
    return this.request("/api/v1/pipeline-control/status", {}, true);
  }

  async startPipelineServer(): Promise<{
    message: string;
    instance_id: string;
    state: string;
    public_ip: string | null;
  }> {
    return this.request("/api/v1/pipeline-control/start", { method: "POST" }, true);
  }

  async stopPipelineServer(): Promise<{
    message: string;
    instance_id: string;
    state: string;
  }> {
    return this.request("/api/v1/pipeline-control/stop", { method: "POST" }, true);
  }

  async ensurePipelineRunning(): Promise<{
    message: string;
    pipeline_url: string;
    state: string;
  }> {
    return this.request("/api/v1/pipeline-control/ensure-running", { method: "POST" }, true);
  }

  // ============================================
  // Pipeline API
  // ============================================

  async getPipelinePlatforms(): Promise<{
    platforms: Array<{ id: string; name: string; icon: string }>;
  }> {
    return this.pipelineRequest("/api/v1/pipeline/platforms");
  }

  async runPipeline(request: PipelineRunRequest): Promise<PipelineRunResponse> {
    // 먼저 파이프라인 서버가 실행 중인지 확인하고 시작
    await this.ensurePipelineRunning();

    return this.pipelineRequest<PipelineRunResponse>(
      "/api/v1/pipeline/run",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
      true
    );
  }

  async getPipelineJobs(brandId?: string): Promise<{ jobs: PipelineJob[] }> {
    const params = brandId ? `?brand_id=${brandId}` : "";
    return this.pipelineRequest(`/api/v1/pipeline/jobs${params}`, {}, true);
  }

  async getPipelineJob(jobId: string): Promise<PipelineJob> {
    return this.pipelineRequest<PipelineJob>(
      `/api/v1/pipeline/jobs/${jobId}`,
      {},
      true
    );
  }

  async deletePipelineJob(jobId: string): Promise<void> {
    await this.pipelineRequest(
      `/api/v1/pipeline/jobs/${jobId}`,
      { method: "DELETE" },
      true
    );
  }

  async getPipelineLogs(jobId: string, offset: number = 0): Promise<{
    job_id: string;
    status: string;
    progress?: string;
    logs: Array<{ timestamp: string; level: string; message: string }>;
  }> {
    return this.pipelineRequest(
      `/api/v1/pipeline/status/${jobId}/logs?offset=${offset}`,
      {},
      true
    );
  }

  // ============================================
  // Social Monitoring API (Extended)
  // ============================================

  async getSocialMonitoring(brandId: string, days: number = 7): Promise<{
    brand_id: string;
    mentions: number;
    sentiment: { positive: number; neutral: number; negative: number };
    platforms: Record<string, number>;
    recent_mentions: Array<{
      id: string;
      platform: string;
      content: string;
      sentiment: string;
      timestamp: string;
      likes: number;
      comments: number;
      author: string;
    }>;
    trending_topics: Array<{ topic: string; count: number; change: number }>;
    total_engagement: number;
    period_days: number;
  }> {
    return this.request(`/api/v1/social/${brandId}?days=${days}`, {}, false);
  }

  async getSocialSentiment(brandId: string, days: number = 30): Promise<{
    daily_sentiment: Array<{
      date: string;
      positive: number;
      neutral: number;
      negative: number;
    }>;
    trend: number;
  }> {
    return this.request(`/api/v1/social/${brandId}/sentiment?days=${days}`, {}, true);
  }

  async getSocialPlatforms(brandId: string, days: number = 30): Promise<{
    platforms: Array<{
      platform: string;
      mentions: number;
      engagement: number;
      sentiment: number;
    }>;
  }> {
    return this.request(`/api/v1/social/${brandId}/platforms?days=${days}`, {}, true);
  }

  // ============================================
  // Analytics API (Extended)
  // ============================================

  async getBrandAnalytics(brandId: string, days: number = 30): Promise<{
    brand_id: string;
    period_days: number;
    total_sessions: number;
    total_messages: number;
    avg_messages_per_session: number;
    grade_distribution: Record<string, number>;
    daily_stats: Array<{
      date: string;
      sessions: number;
      messages: number;
      engagement?: number;
    }>;
    top_questions: Array<{
      question: string;
      count: number;
      category?: string;
    }>;
  }> {
    return this.request(`/api/v1/analytics/${brandId}?days=${days}`, {}, false);
  }

  async getContentPerformance(brandId: string, params?: {
    days?: number;
    platform?: string;
    limit?: number;
  }): Promise<{
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
  }> {
    const searchParams = new URLSearchParams();
    if (params?.days) searchParams.set("days", String(params.days));
    if (params?.platform) searchParams.set("platform", params.platform);
    if (params?.limit) searchParams.set("limit", String(params.limit));
    const query = searchParams.toString();
    return this.request(`/api/v1/analytics/${brandId}/content-performance${query ? `?${query}` : ""}`, {}, true);
  }

  async getEngagementTrends(brandId: string, days: number = 30): Promise<{
    trends: Array<{
      date: string;
      posts: number;
      likes: number;
      comments: number;
      shares: number;
    }>;
  }> {
    return this.request(`/api/v1/analytics/${brandId}/engagement-trends?days=${days}`, {}, true);
  }

  async getQualityMetrics(brandId: string, days: number = 30): Promise<{
    metrics: Array<{
      grade: string;
      count: number;
      percentage: number;
    }>;
    avg_score: number;
  }> {
    return this.request(`/api/v1/analytics/${brandId}/quality-metrics?days=${days}`, {}, true);
  }

  async exportAnalytics(brandId: string, params?: {
    days?: number;
    format?: "json" | "csv";
  }): Promise<Blob> {
    const searchParams = new URLSearchParams();
    if (params?.days) searchParams.set("days", String(params.days));
    if (params?.format) searchParams.set("format", params.format);
    const query = searchParams.toString();

    const response = await fetch(
      `${this.baseUrl}/api/v1/analytics/${brandId}/export${query ? `?${query}` : ""}`,
      {
        headers: {
          Authorization: `Bearer ${getToken()}`,
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`);
    }

    return response.blob();
  }

  // ============================================
  // Web Crawler API
  // ============================================

  async startCrawl(request: {
    urls: string[];
    brand_id?: string;
    max_pages?: number;
    max_depth?: number;
    crawler_type?: string;
    include_patterns?: string[];
    exclude_patterns?: string[];
    save_markdown?: boolean;
    wait_for_finish?: boolean;
  }): Promise<{
    job_id: string;
    status: string;
    run_id: string;
    urls: string[];
    max_pages: number;
    message: string;
  }> {
    return this.request(
      "/api/v1/crawler/start",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
      true
    );
  }

  async getCrawlerJobs(brandId?: string): Promise<{
    jobs: Array<{
      job_id: string;
      status: string;
      urls: string[];
      max_pages: number;
      created_at: string;
      results_count?: number;
    }>;
  }> {
    const params = brandId ? `?brand_id=${brandId}` : "";
    return this.request(`/api/v1/crawler/jobs${params}`, {}, true);
  }

  async getCrawlerJob(jobId: string): Promise<{
    job_id: string;
    status: string;
    urls: string[];
    max_pages: number;
    created_at: string;
    results?: Array<{
      url: string;
      title?: string;
      text_preview?: string;
    }>;
  }> {
    return this.request(`/api/v1/crawler/jobs/${jobId}`, {}, true);
  }

  async getCrawlerResults(jobId: string, params?: {
    page?: number;
    page_size?: number;
  }): Promise<{
    results: Array<{
      url: string;
      title?: string;
      text_preview?: string;
      markdown_length?: number;
    }>;
    total: number;
    page: number;
    page_size: number;
  }> {
    const searchParams = new URLSearchParams();
    if (params?.page) searchParams.set("page", String(params.page));
    if (params?.page_size) searchParams.set("page_size", String(params.page_size));
    const query = searchParams.toString();
    return this.request(`/api/v1/crawler/jobs/${jobId}/results${query ? `?${query}` : ""}`, {}, true);
  }

  async cancelCrawl(jobId: string): Promise<{ message: string }> {
    return this.request(
      `/api/v1/crawler/jobs/${jobId}/cancel`,
      { method: "POST" },
      true
    );
  }

  async saveCrawlToNeo4j(jobId: string, brandId: string): Promise<{ message: string; count: number }> {
    return this.request(
      `/api/v1/crawler/jobs/${jobId}/save-to-neo4j?brand_id=${brandId}`,
      { method: "POST" },
      true
    );
  }

  // ============================================
  // Products API
  // ============================================

  async getProducts(brandId: string, params?: {
    category?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    brand_id: string;
    products: Array<{
      id: string;
      name: string;
      category: string;
      price: number;
      stock: number;
      sales: number;
      rating: number;
      image_url?: string;
    }>;
    stats: {
      total_products: number;
      active_products: number;
      total_sales: number;
      avg_price: number;
    };
    categories: Array<{ category: string; count: number }>;
    pagination: { limit: number; offset: number; total: number };
  }> {
    const searchParams = new URLSearchParams();
    if (params?.category) searchParams.set("category", params.category);
    if (params?.limit) searchParams.set("limit", String(params.limit));
    if (params?.offset) searchParams.set("offset", String(params.offset));
    const query = searchParams.toString();
    return this.request(`/api/v1/products/${brandId}${query ? `?${query}` : ""}`, {}, true);
  }

  async getProductRecommendations(brandId: string, params?: {
    segment?: string;
    limit?: number;
  }): Promise<{
    brand_id: string;
    segment: string;
    recommendations: Array<{
      product_id: string;
      name: string;
      score: number;
      reason: string;
      category: string;
    }>;
    total: number;
  }> {
    const searchParams = new URLSearchParams();
    if (params?.segment) searchParams.set("segment", params.segment);
    if (params?.limit) searchParams.set("limit", String(params.limit));
    const query = searchParams.toString();
    return this.request(`/api/v1/products/${brandId}/recommendations${query ? `?${query}` : ""}`, {}, true);
  }

  async getProductCategories(brandId: string): Promise<{
    categories: Array<{
      category: string;
      count: number;
      total_sales: number;
    }>;
  }> {
    return this.request(`/api/v1/products/${brandId}/categories`, {}, true);
  }

  async getTopSellers(brandId: string, params?: {
    days?: number;
    limit?: number;
  }): Promise<{
    top_sellers: Array<{
      rank: number;
      product_id: string;
      name: string;
      sales: number;
      revenue: number;
      category: string;
    }>;
  }> {
    const searchParams = new URLSearchParams();
    if (params?.days) searchParams.set("days", String(params.days));
    if (params?.limit) searchParams.set("limit", String(params.limit));
    const query = searchParams.toString();
    return this.request(`/api/v1/products/${brandId}/top-sellers${query ? `?${query}` : ""}`, {}, true);
  }

  // ============================================
  // Alerts API (Extended)
  // ============================================

  async getAlertStats(brandId: string): Promise<{
    total_alerts: number;
    by_status: Record<string, number>;
    by_severity: Record<string, number>;
    by_type: Record<string, number>;
    active_rules: number;
    alerts_today: number;
    alerts_this_week: number;
  }> {
    return this.request(`/api/v1/alerts/stats?brand_id=${brandId}`, {}, true);
  }

  async resolveAlert(alertId: string, userId: string, comment?: string): Promise<Alert> {
    const params = new URLSearchParams();
    params.set("user_id", userId);
    if (comment) params.set("comment", comment);
    return this.request<Alert>(
      `/api/v1/alerts/${alertId}/resolve?${params.toString()}`,
      { method: "POST" },
      true
    );
  }

  async toggleAlertRule(ruleId: string): Promise<AlertRule> {
    return this.request<AlertRule>(
      `/api/v1/alerts/rules/${ruleId}/toggle`,
      { method: "POST" },
      true
    );
  }

  // ============================================
  // Neo4j Knowledge Graph API
  // ============================================

  async getKnowledgeGraph(brandId: string, params?: {
    limit?: number;
    nodeTypes?: string[];
    balanced?: boolean;
  }): Promise<{
    nodes: Array<{
      id: string;
      label: string;
      type: string;
      properties: Record<string, unknown>;
    }>;
    relationships: Array<{
      source: string;
      target: string;
      type: string;
      properties?: Record<string, unknown>;
    }>;
    stats: {
      total_nodes: number;
      total_relationships: number;
      node_types: Record<string, number>;
      relationship_types: Record<string, number>;
    };
  }> {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set("limit", String(params.limit));
    if (params?.nodeTypes) searchParams.set("node_types", params.nodeTypes.join(","));
    if (params?.balanced) searchParams.set("balanced", "true");
    const query = searchParams.toString();
    return this.request(`/api/v1/brands/${brandId}/graph${query ? `?${query}` : ""}`, {}, false);
  }

  async getConceptCloud(brandId: string, limit: number = 30): Promise<{
    concepts: Array<{
      name: string;
      count: number;
      type: string | null;
      related_concepts: string[];
    }>;
  }> {
    return this.request(`/api/v1/brands/${brandId}/concepts?limit=${limit}`, {}, true);
  }

  async getGraphSummary(brandId: string): Promise<{
    brand_id: string;
    summary: {
      total_nodes: number;
      total_relationships: number;
    };
    node_types: Record<string, number>;
    relationship_types: Record<string, number>;
    platforms: Record<string, number>;
    date_range: {
      first: string | null;
      last: string | null;
    };
    top_concepts: Array<{
      name: string;
      count: number;
      type: string | null;
      related_concepts: string[];
    }>;
  }> {
    return this.request(`/api/v1/brands/${brandId}/graph-summary`, {}, true);
  }

  // ============================================
  // Neo4j Direct API
  // ============================================

  async getNeo4jHealth(): Promise<{
    status: string;
    state: string;
    latency_ms: number;
    database: string;
    vector_available: boolean;
    metrics: {
      total_queries: number;
      read_queries: number;
      write_queries: number;
      avg_query_time_ms: number;
      error_count: number;
    };
  }> {
    return this.request("/api/v1/neo4j/health", {}, true);
  }

  async getNeo4jSchema(): Promise<{
    labels: string[];
    relationship_types: string[];
    indexes: number;
  }> {
    return this.request("/api/v1/neo4j/schema", {}, true);
  }

  // ============================================
  // Content Generation API
  // ============================================

  async generateContent(params: {
    brand_id: string;
    platform: "instagram" | "twitter" | "facebook" | "linkedin" | "blog";
    content_type: "post" | "caption" | "ad" | "article";
    topic: string;
    tone?: "professional" | "friendly" | "playful" | "inspiring" | "urgent";
    reference_content_ids?: string[];
  }): Promise<{
    brand_id: string;
    content: string;
    hashtags: string[];
    platform: string;
    content_type: string;
    metadata: {
      validation?: { grade: string; score: number };
      brand_concepts_used?: string[];
      reference_contents_count?: number;
      tone?: string;
    };
  }> {
    return this.request("/api/v1/content/generate", {
      method: "POST",
      body: JSON.stringify(params),
    }, false);
  }

  async getContentTemplates(brandId: string, platform?: string): Promise<{
    brand_id: string;
    templates: Array<{
      id: string;
      platform: string;
      preview: string;
      structure: {
        line_count: number;
        char_count: number;
        has_emoji: boolean;
        has_hashtags: boolean;
        has_numbered_list: boolean;
        has_links: boolean;
      };
      metrics?: string;
    }>;
    total: number;
  }> {
    const query = platform ? `?platform=${platform}` : "";
    return this.request(`/api/v1/content/${brandId}/templates${query}`, {}, false);
  }

  async getContentSuggestions(brandId: string): Promise<{
    brand_id: string;
    suggestions: Array<{
      topic: string;
      type: "concept" | "trending" | "opportunity";
      reason: string;
    }>;
    concepts: string[];
    trending_hashtags: string[];
  }> {
    return this.request(`/api/v1/content/${brandId}/suggestions`, {}, false);
  }

  // ============================================
  // On-Chain Intelligence API
  // ============================================

  async getOnChainStats(): Promise<{
    total_wallets: number;
    total_transfers: number;
    total_entities: number;
    total_volume_usd: number;
    last_collection?: string;
    collection_status?: "idle" | "running" | "completed" | "error";
  }> {
    return this.onchainRequest("/api/v1/onchain/stats", {}, false);
  }

  async analyzeAddress(address: string, chain: string = "ethereum"): Promise<{
    address: string;
    display_name: string;
    entity?: string;
    counterparties: number;
    transfers: number;
    total_received_usd?: number;
    total_sent_usd?: number;
    first_seen?: string;
    last_active?: string;
    tags?: string[];
  }> {
    return this.onchainRequest(
      `/api/v1/onchain/analyze/${address}?chain=${chain}`,
      {},
      false
    );
  }

  async traceFlow(params: {
    address: string;
    direction?: "in" | "out" | "all";
    depth?: number;
    min_usd?: number;
    chain?: string;
  }): Promise<{
    transfers: Array<{
      id: string;
      transaction_hash: string;
      from_address: { address: string; display_name: string; entity?: string };
      to_address: { address: string; display_name: string; entity?: string };
      token_symbol?: string;
      historical_usd: number;
      chain: string;
      block_timestamp?: number;
    }>;
    addresses: Record<string, { address: string; display_name: string }>;
    entities: Record<string, { id: string; name: string; type?: string }>;
    total_volume_usd: number;
  }> {
    const searchParams = new URLSearchParams();
    searchParams.set("address", params.address);
    if (params.direction) searchParams.set("direction", params.direction);
    if (params.depth) searchParams.set("depth", String(params.depth));
    if (params.min_usd) searchParams.set("min_usd", String(params.min_usd));
    if (params.chain) searchParams.set("chain", params.chain);
    return this.onchainRequest(
      `/api/v1/onchain/trace?${searchParams.toString()}`,
      {},
      false
    );
  }

  async queryOnChain(question: string, useLanggraph: boolean = true): Promise<{
    query: string;
    query_type: string;
    analysis: string;
    confidence: number;
    sources: string[];
    cypher_query?: string;
    addresses_found?: string[];
    graph_results_count: number;
    vector_results_count: number;
    error?: string;
  }> {
    return this.pipelineRequest(
      "/api/v1/onchain/query",
      {
        method: "POST",
        body: JSON.stringify({ question, use_langgraph: useLanggraph }),
      },
      true
    );
  }

  async startCollection(params: {
    strategy: "all" | "seeds" | "events" | "entities";
    categories?: string[];
    depth?: number;
    min_usd?: number;
  }): Promise<{
    job_id: string;
    status: string;
    message: string;
  }> {
    return this.pipelineRequest(
      "/api/v1/onchain/collect",
      {
        method: "POST",
        body: JSON.stringify(params),
      },
      true
    );
  }

  async getCollectionStatus(jobId?: string): Promise<{
    status: "idle" | "running" | "completed" | "error";
    current_job?: {
      job_id: string;
      strategy: string;
      progress: number;
      started_at: string;
    };
    last_result?: {
      strategy: string;
      completed_at: string;
      duration_seconds: number;
      transfers_collected: number;
      addresses_collected: number;
      entities_found: number;
      errors: string[];
    };
  }> {
    const path = jobId
      ? `/api/v1/onchain/collect/status/${jobId}`
      : "/api/v1/onchain/collect/status";
    return this.pipelineRequest(path, {}, true);
  }

  async getSeeds(params?: {
    category?: string;
    priority?: number;
  }): Promise<{
    seeds: Array<{
      address: string;
      name: string;
      category: string;
      chain: string;
      priority: number;
      notes?: string;
    }>;
    entities: Array<{
      entity_id: string;
      name: string;
      category: string;
      priority: number;
    }>;
    total: number;
  }> {
    const searchParams = new URLSearchParams();
    if (params?.category) searchParams.set("category", params.category);
    if (params?.priority) searchParams.set("priority", String(params.priority));
    const query = searchParams.toString();
    return this.pipelineRequest(
      `/api/v1/onchain/seeds${query ? `?${query}` : ""}`,
      {},
      true
    );
  }

  async getWalletGraph(address: string, depth: number = 2): Promise<{
    nodes: Array<{
      id: string;
      label: string;
      type: "wallet" | "entity" | "contract";
      properties: Record<string, unknown>;
    }>;
    relationships: Array<{
      source: string;
      target: string;
      type: string;
      usd?: number;
      count?: number;
    }>;
    center_address: string;
  }> {
    return this.pipelineRequest(
      `/api/v1/onchain/graph/${address}?depth=${depth}`,
      {},
      true
    );
  }
}

export const apiClient = new ApiClient(API_BASE_URL, PIPELINE_API_URL, ONCHAIN_API_URL);
