"use client";

import { useState, useEffect, useCallback } from "react";
import { Zap, Check, Clock, Loader2, RefreshCw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { apiClient } from "@/lib/api-client";

interface Action {
  id: string;
  title: string;
  priority: "high" | "medium" | "low";
  status: "pending" | "done";
}

interface ActionRecommendationsCardProps {
  brandId?: string;
  actions?: Action[];
}

const MOCK_ACTIONS: Action[] = [
  { id: "1", title: "Launch Instagram Reels Series", priority: "high", status: "pending" },
  { id: "2", title: "Optimize Product Descriptions", priority: "high", status: "done" },
  { id: "3", title: "Customer Testimonial Campaign", priority: "medium", status: "pending" },
];

export default function ActionRecommendationsCard({ brandId, actions: initialActions }: ActionRecommendationsCardProps) {
  const [actions, setActions] = useState<Action[]>(initialActions || MOCK_ACTIONS);
  const [isLoading, setIsLoading] = useState(false);
  const [completedIds, setCompletedIds] = useState<Set<string>>(new Set());

  // Fetch AI-generated recommendations
  const fetchRecommendations = useCallback(async () => {
    if (!brandId) return;

    setIsLoading(true);
    try {
      // Get recommendations based on brand analytics and trends
      const [analytics, social] = await Promise.all([
        apiClient.getBrandAnalytics(brandId, 7).catch(() => null),
        apiClient.getSocialMonitoring(brandId, 7).catch(() => null),
      ]);

      const recommendations: Action[] = [];
      let id = 1;

      // Generate recommendations based on data
      if (analytics) {
        const topQuestions = analytics.top_questions || [];
        if (topQuestions.length > 0) {
          recommendations.push({
            id: String(id++),
            title: `FAQ 콘텐츠 업데이트: "${topQuestions[0]?.question?.slice(0, 30)}..."`,
            priority: "high",
            status: "pending",
          });
        }

        if (analytics.grade_distribution) {
          const lowGrades = (analytics.grade_distribution["D"] || 0) + (analytics.grade_distribution["F"] || 0);
          if (lowGrades > 0) {
            recommendations.push({
              id: String(id++),
              title: "응답 품질 개선을 위한 지식 베이스 보강",
              priority: "high",
              status: "pending",
            });
          }
        }
      }

      if (social) {
        const trending = social.trending_topics || [];
        if (trending.length > 0 && trending[0].change > 0) {
          recommendations.push({
            id: String(id++),
            title: `트렌딩 토픽 활용: "${trending[0].topic}"`,
            priority: "medium",
            status: "pending",
          });
        }

        if (social.sentiment) {
          const { positive, negative } = social.sentiment;
          if (negative > positive * 0.5) {
            recommendations.push({
              id: String(id++),
              title: "부정적 멘션 대응 전략 수립",
              priority: "high",
              status: "pending",
            });
          }
        }

        const platforms = social.platforms || {};
        const platformEntries = Object.entries(platforms);
        const lowPlatforms = platformEntries.filter(([, count]) => count < 5);
        if (lowPlatforms.length > 0) {
          recommendations.push({
            id: String(id++),
            title: `${lowPlatforms[0][0]} 채널 활성화 캠페인`,
            priority: "medium",
            status: "pending",
          });
        }
      }

      // Add default recommendations if we don't have enough
      if (recommendations.length < 3) {
        recommendations.push(
          { id: String(id++), title: "주간 콘텐츠 캘린더 업데이트", priority: "medium", status: "pending" },
          { id: String(id++), title: "고객 리뷰 수집 캠페인", priority: "low", status: "pending" }
        );
      }

      // Keep completed status for existing items
      setActions(recommendations.slice(0, 5).map(action => ({
        ...action,
        status: completedIds.has(action.id) ? "done" : action.status,
      })));
    } catch (error) {
      console.error("Failed to fetch recommendations:", error);
      setActions(MOCK_ACTIONS);
    } finally {
      setIsLoading(false);
    }
  }, [brandId, completedIds]);

  useEffect(() => {
    fetchRecommendations();
  }, [fetchRecommendations]);

  const toggleComplete = (actionId: string) => {
    setCompletedIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(actionId)) {
        newSet.delete(actionId);
      } else {
        newSet.add(actionId);
      }
      return newSet;
    });
    setActions(prev =>
      prev.map(a =>
        a.id === actionId ? { ...a, status: a.status === "done" ? "pending" : "done" } : a
      )
    );
  };
  const doneCount = actions.filter((a) => a.status === "done").length;

  if (isLoading && actions.length === 0) {
    return (
      <Card className="h-full flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-sm font-medium">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-muted-foreground" />
              Actions
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-sm font-medium">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-muted-foreground" />
            Actions
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">{doneCount}/{actions.length}</span>
            <button
              onClick={fetchRecommendations}
              disabled={isLoading}
              className="p-1 rounded hover:bg-muted transition-colors disabled:opacity-50"
              title="새로고침"
            >
              <RefreshCw className={`h-3 w-3 text-muted-foreground ${isLoading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <div className="h-1 bg-muted rounded-full overflow-hidden mb-3">
          <div
            className="h-full bg-foreground/60 rounded-full transition-all duration-300"
            style={{ width: `${(doneCount / actions.length) * 100}%` }}
          />
        </div>
        <div className="flex-1 space-y-2 overflow-y-auto">
          {actions.map((action) => (
            <button
              key={action.id}
              onClick={() => toggleComplete(action.id)}
              className="w-full flex items-start gap-2 p-2 rounded border bg-card hover:bg-muted/50 transition-colors text-left"
            >
              <div className={`mt-0.5 ${action.status === "done" ? "text-foreground" : "text-muted-foreground"}`}>
                {action.status === "done" ? <Check className="h-4 w-4" /> : <Clock className="h-4 w-4" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className={`text-sm ${action.status === "done" ? "line-through text-muted-foreground" : ""}`}>
                  {action.title}
                </div>
                <div className="text-xs text-muted-foreground capitalize">
                  <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1 ${
                    action.priority === "high" ? "bg-red-500" :
                    action.priority === "medium" ? "bg-yellow-500" : "bg-green-500"
                  }`} />
                  {action.priority}
                </div>
              </div>
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
