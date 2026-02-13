"use client";

import { useMemo } from "react";
import { TrendingUp, ArrowUp, ArrowDown, Loader2, BarChart3 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface DataPoint {
  date: string;
  engagement: number;
  sessions: number;
}

interface EngagementWaveCardProps {
  data?: DataPoint[];
  isLoading?: boolean;
}

export default function EngagementWaveCard({ data, isLoading }: EngagementWaveCardProps) {
  const chartData = useMemo(() => data && data.length > 0 ? data : [], [data]);

  const stats = useMemo(() => {
    if (!chartData || chartData.length === 0) {
      return { total: 0, avg: 0, trend: 0 };
    }
    const total = chartData.reduce((acc, d) => acc + d.engagement, 0);
    const avg = total / chartData.length;
    const firstHalf = chartData.slice(0, Math.floor(chartData.length / 2));
    const secondHalf = chartData.slice(Math.floor(chartData.length / 2));
    const firstAvg = firstHalf.reduce((acc, d) => acc + d.engagement, 0) / firstHalf.length || 0;
    const secondAvg = secondHalf.reduce((acc, d) => acc + d.engagement, 0) / secondHalf.length || 0;
    const trend = firstAvg > 0 ? ((secondAvg - firstAvg) / firstAvg) * 100 : 0;
    return { total, avg, trend };
  }, [chartData]);

  const maxEngagement = Math.max(...chartData.map(d => d.engagement), 1);

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-sm font-medium">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              Engagement
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-32">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  // No data state
  if (chartData.length === 0) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
            Engagement
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center h-[calc(100%-60px)]">
          <BarChart3 className="h-10 w-10 text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">데이터가 없습니다</p>
          <p className="text-xs text-muted-foreground/60 mt-1">대화 세션이 기록되면 표시됩니다</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-sm font-medium">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
            Engagement
          </div>
          {stats.total > 0 && (
            <div className="flex items-center gap-1 text-xs">
              {stats.trend >= 0 ? (
                <ArrowUp className="h-3 w-3 text-green-500" />
              ) : (
                <ArrowDown className="h-3 w-3 text-red-500" />
              )}
              <span className={stats.trend >= 0 ? "text-green-500" : "text-red-500"}>
                {Math.abs(stats.trend).toFixed(1)}%
              </span>
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-xl font-bold tabular-nums">{stats.total.toLocaleString()}</div>
            <div className="text-xs text-muted-foreground">Total</div>
          </div>
          <div>
            <div className="text-xl font-bold tabular-nums">{stats.avg.toFixed(0)}</div>
            <div className="text-xs text-muted-foreground">Average</div>
          </div>
          <div>
            <div className="text-xl font-bold tabular-nums">{chartData.length}</div>
            <div className="text-xs text-muted-foreground">Days</div>
          </div>
        </div>
        <div className="flex items-end gap-1 h-24">
          {chartData.map((point, idx) => (
            <div key={idx} className="flex-1 flex flex-col items-center">
              <div
                className="w-full bg-foreground/30 rounded-sm hover:bg-foreground/50 transition-colors"
                style={{ height: `${(point.engagement / maxEngagement) * 100}%`, minHeight: '4px' }}
              />
            </div>
          ))}
        </div>
        <div className="flex gap-1 text-[10px] text-muted-foreground">
          {chartData.map((point, idx) => (
            <div key={idx} className="flex-1 text-center truncate">
              {point.date.split(' ')[1] || point.date.slice(-2)}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
