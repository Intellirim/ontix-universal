"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import type {
  ChatWithSessionRequest,
  BrandCreate,
  AlertRule,
  PipelineRunRequest,
  StartCrawlRequest,
} from "@/types";

// ============================================
// Brand Hooks
// ============================================

export function useBrands() {
  return useQuery({
    queryKey: ["brands"],
    queryFn: () => apiClient.getBrands(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useBrand(brandId: string) {
  return useQuery({
    queryKey: ["brands", brandId],
    queryFn: () => apiClient.getBrand(brandId),
    enabled: !!brandId,
    staleTime: 5 * 60 * 1000,
  });
}

export function useBrandStats(brandId: string) {
  return useQuery({
    queryKey: ["brands", brandId, "stats"],
    queryFn: () => apiClient.getBrandStats(brandId),
    enabled: !!brandId,
    refetchInterval: 60 * 1000, // 1 minute auto-refresh
  });
}

export function useCreateBrand() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: BrandCreate) => apiClient.createBrand(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["brands"] });
    },
  });
}

// ============================================
// Chat Hooks
// ============================================

export function useSendMessage(brandId: string) {
  return useMutation({
    mutationFn: (message: string) =>
      apiClient.sendMessage({ brand_id: brandId, message }),
  });
}

export function useSendMessageWithSession(brandId: string) {
  return useMutation({
    mutationFn: (request: ChatWithSessionRequest) =>
      apiClient.sendMessageWithSession(request),
  });
}

// ============================================
// AI Advisor Hooks (Dedicated Endpoint)
// ============================================

export function useSendAdvisorMessage() {
  return useMutation({
    mutationFn: (request: { brand_id: string; message: string; session_id?: string }) =>
      apiClient.sendAdvisorMessage(request),
  });
}

export function useSendAdvisorMessageWithSession() {
  return useMutation({
    mutationFn: (request: { brand_id: string; message: string; session_id?: string }) =>
      apiClient.sendAdvisorMessageWithSession(request),
  });
}

// ============================================
// Health Hooks
// ============================================

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 30 * 1000, // 30 seconds
    staleTime: 10 * 1000,
  });
}

// ============================================
// Features Hooks
// ============================================

export function useFeatures() {
  return useQuery({
    queryKey: ["features"],
    queryFn: () => apiClient.getFeatures(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

// ============================================
// User Profile Hooks
// ============================================

export function useCurrentUser() {
  return useQuery({
    queryKey: ["currentUser"],
    queryFn: () => apiClient.getCurrentUser(),
    staleTime: 5 * 60 * 1000,
  });
}

export function useUpdateProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { email?: string; name?: string }) =>
      apiClient.updateProfile(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["currentUser"] });
    },
  });
}

export function useChangePassword() {
  return useMutation({
    mutationFn: (data: { current_password: string; new_password: string }) =>
      apiClient.changePassword(data),
  });
}

// ============================================
// Alert Hooks
// ============================================

export function useAlerts(params?: {
  acknowledged?: boolean;
  severity?: string;
  limit?: number;
}) {
  return useQuery({
    queryKey: ["alerts", params],
    queryFn: () => apiClient.getAlerts(params),
    refetchInterval: 60 * 1000, // 1 minute auto-refresh
  });
}

export function useAcknowledgeAlert() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (alertId: string) => apiClient.acknowledgeAlert(alertId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
    },
  });
}

export function useAlertRules() {
  return useQuery({
    queryKey: ["alertRules"],
    queryFn: () => apiClient.getAlertRules(),
    refetchInterval: 60 * 1000,
  });
}

export function useCreateAlertRule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (rule: Omit<AlertRule, "id" | "created_at">) =>
      apiClient.createAlertRule(rule),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alertRules"] });
    },
  });
}

export function useUpdateAlertRule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      ruleId,
      rule,
    }: {
      ruleId: string;
      rule: Partial<AlertRule>;
    }) => apiClient.updateAlertRule(ruleId, rule),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alertRules"] });
    },
  });
}

export function useDeleteAlertRule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (ruleId: string) => apiClient.deleteAlertRule(ruleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alertRules"] });
    },
  });
}

// ============================================
// Analytics Hooks
// ============================================

export function useAnalyticsOverview(period: string = "7d") {
  return useQuery({
    queryKey: ["analytics", "overview", period],
    queryFn: () => apiClient.getAnalyticsOverview(period),
    refetchInterval: 60 * 1000, // 1 minute auto-refresh
  });
}

export function useConversationAnalytics(
  period: string = "7d",
  brandId?: string
) {
  return useQuery({
    queryKey: ["analytics", "conversations", period, brandId],
    queryFn: () => apiClient.getConversationAnalytics(period, brandId),
    refetchInterval: 60 * 1000,
  });
}

export function usePerformanceAnalytics(period: string = "24h") {
  return useQuery({
    queryKey: ["analytics", "performance", period],
    queryFn: () => apiClient.getPerformanceAnalytics(period),
    refetchInterval: 60 * 1000,
  });
}

// ============================================
// Pipeline Hooks
// ============================================

export function usePipelinePlatforms() {
  return useQuery({
    queryKey: ["pipeline", "platforms"],
    queryFn: () => apiClient.getPipelinePlatforms(),
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

export function usePipelineJobs(brandId?: string) {
  return useQuery({
    queryKey: ["pipeline", "jobs", brandId],
    queryFn: () => apiClient.getPipelineJobs(brandId),
    refetchInterval: 5 * 1000, // 5 seconds for active jobs
  });
}

export function usePipelineJob(jobId: string) {
  return useQuery({
    queryKey: ["pipeline", "jobs", jobId],
    queryFn: () => apiClient.getPipelineJob(jobId),
    enabled: !!jobId,
    refetchInterval: 2 * 1000,
  });
}

export function useRunPipeline() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: PipelineRunRequest) => apiClient.runPipeline(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline", "jobs"] });
    },
  });
}

export function useDeletePipelineJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => apiClient.deletePipelineJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline", "jobs"] });
    },
  });
}

export function usePipelineLogs(jobId: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ["pipeline", "logs", jobId],
    queryFn: () => apiClient.getPipelineLogs(jobId),
    enabled: !!jobId && enabled,
    refetchInterval: enabled ? 1000 : false, // Poll every 1 second when enabled
    staleTime: 0,
  });
}

// ============================================
// Pipeline Server Control Hooks
// ============================================

export function usePipelineServerStatus() {
  return useQuery({
    queryKey: ["pipeline", "server", "status"],
    queryFn: () => apiClient.getPipelineServerStatus(),
    refetchInterval: 10000, // 10초마다 상태 확인
    staleTime: 5000,
  });
}

export function useStartPipelineServer() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => apiClient.startPipelineServer(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline", "server", "status"] });
    },
  });
}

export function useStopPipelineServer() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => apiClient.stopPipelineServer(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline", "server", "status"] });
    },
  });
}

// ============================================
// Social Monitoring Hooks
// ============================================

export function useSocialMonitoring(brandId: string, days: number = 7) {
  return useQuery({
    queryKey: ["social", brandId, days],
    queryFn: () => apiClient.getSocialMonitoring(brandId, days),
    enabled: !!brandId,
    refetchInterval: 60 * 1000,
  });
}

export function useSocialMentions(brandId: string, params?: {
  platform?: string;
  sentiment?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: ["social", brandId, "mentions", params],
    queryFn: () => apiClient.getSocialMentions(brandId, params),
    enabled: !!brandId,
  });
}

export function useSocialSentiment(brandId: string, days: number = 30) {
  return useQuery({
    queryKey: ["social", brandId, "sentiment", days],
    queryFn: () => apiClient.getSocialSentiment(brandId, days),
    enabled: !!brandId,
  });
}

// ============================================
// Sessions Hooks
// ============================================

export function useChatSessions(brandId: string, params?: {
  userId?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: ["sessions", brandId, params],
    queryFn: () => apiClient.getChatSessions(brandId, params),
    enabled: !!brandId,
  });
}

export function useSessionMessages(brandId: string, sessionId: string) {
  return useQuery({
    queryKey: ["sessions", brandId, sessionId, "messages"],
    queryFn: () => apiClient.getSessionMessages(brandId, sessionId),
    enabled: !!brandId && !!sessionId,
  });
}

// ============================================
// Brand Analytics Hooks
// ============================================

export function useBrandAnalytics(brandId: string, days: number = 30) {
  return useQuery({
    queryKey: ["analytics", "brand", brandId, days],
    queryFn: () => apiClient.getBrandAnalytics(brandId, days),
    enabled: !!brandId,
    refetchInterval: 60 * 1000,
  });
}

export function useContentPerformance(brandId: string, params?: {
  days?: number;
  platform?: string;
  limit?: number;
}) {
  return useQuery({
    queryKey: ["analytics", brandId, "content", params],
    queryFn: () => apiClient.getContentPerformance(brandId, params),
    enabled: !!brandId,
  });
}

export function useEngagementTrends(brandId: string, days: number = 30) {
  return useQuery({
    queryKey: ["analytics", brandId, "engagement", days],
    queryFn: () => apiClient.getEngagementTrends(brandId, days),
    enabled: !!brandId,
  });
}

export function useQualityMetrics(brandId: string, days: number = 30) {
  return useQuery({
    queryKey: ["analytics", brandId, "quality", days],
    queryFn: () => apiClient.getQualityMetrics(brandId, days),
    enabled: !!brandId,
  });
}

// ============================================
// Web Crawler Hooks
// ============================================

export function useCrawlerJobs(brandId?: string) {
  return useQuery({
    queryKey: ["crawler", "jobs", brandId],
    queryFn: () => apiClient.getCrawlerJobs(brandId),
    refetchInterval: 5 * 1000,
  });
}

export function useStartCrawl() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: StartCrawlRequest) => apiClient.startCrawl(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["crawler", "jobs"] });
    },
  });
}

export function useCancelCrawl() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => apiClient.cancelCrawl(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["crawler", "jobs"] });
    },
  });
}

// ============================================
// Products Hooks
// ============================================

export function useProducts(brandId: string, params?: {
  category?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: ["products", brandId, params],
    queryFn: () => apiClient.getProducts(brandId, params),
    enabled: !!brandId,
  });
}

export function useProductRecommendations(brandId: string, params?: {
  segment?: string;
  limit?: number;
}) {
  return useQuery({
    queryKey: ["products", brandId, "recommendations", params],
    queryFn: () => apiClient.getProductRecommendations(brandId, params),
    enabled: !!brandId,
  });
}

export function useTopSellers(brandId: string, params?: {
  days?: number;
  limit?: number;
}) {
  return useQuery({
    queryKey: ["products", brandId, "top-sellers", params],
    queryFn: () => apiClient.getTopSellers(brandId, params),
    enabled: !!brandId,
  });
}

// ============================================
// Alert Stats Hook
// ============================================

export function useAlertStats(brandId: string) {
  return useQuery({
    queryKey: ["alerts", "stats", brandId],
    queryFn: () => apiClient.getAlertStats(brandId),
    enabled: !!brandId,
    refetchInterval: 60 * 1000,
  });
}

// ============================================
// Neo4j Knowledge Graph Hooks
// ============================================

export function useKnowledgeGraph(brandId: string, params?: {
  limit?: number;
  nodeTypes?: string[];
  balanced?: boolean;
}) {
  return useQuery({
    queryKey: ["graph", brandId, params],
    queryFn: () => apiClient.getKnowledgeGraph(brandId, params),
    enabled: !!brandId,
  });
}

export function useConceptCloud(brandId: string, limit: number = 30) {
  return useQuery({
    queryKey: ["concepts", brandId, limit],
    queryFn: () => apiClient.getConceptCloud(brandId, limit),
    enabled: !!brandId,
  });
}

export function useGraphSummary(brandId: string) {
  return useQuery({
    queryKey: ["graph", "summary", brandId],
    queryFn: () => apiClient.getGraphSummary(brandId),
    enabled: !!brandId,
  });
}

// ============================================
// Neo4j Direct Query Hooks
// ============================================

export function useNeo4jHealth() {
  return useQuery({
    queryKey: ["neo4j", "health"],
    queryFn: () => apiClient.getNeo4jHealth(),
    refetchInterval: 30 * 1000,
  });
}

export function useNeo4jSchema() {
  return useQuery({
    queryKey: ["neo4j", "schema"],
    queryFn: () => apiClient.getNeo4jSchema(),
    staleTime: 5 * 60 * 1000,
  });
}
