"use client";

import { useMemo } from "react";
import { Activity, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ChannelActivityNoiseCardProps {
  data?: {
    instagram: number;
    youtube: number;
    tiktok: number;
    twitter: number;
    website: number;
  };
  platforms?: Record<string, number>;
  isLoading?: boolean;
}

export default function ChannelActivityNoiseCard({ data, platforms, isLoading }: ChannelActivityNoiseCardProps) {
  // Convert platforms data from API to normalized channel data
  const channelData = useMemo(() => {
    if (data) return data;

    if (platforms && Object.keys(platforms).length > 0) {
      // API returns percentages (0-100), normalize to 0-1 range
      const getValue = (key: string) => {
        const val = platforms[key] || platforms[key.charAt(0).toUpperCase() + key.slice(1)] || 0;
        // If values are percentages (>1), divide by 100
        return val > 1 ? val / 100 : val;
      };
      return {
        instagram: getValue("instagram"),
        youtube: getValue("youtube"),
        tiktok: getValue("tiktok"),
        twitter: getValue("twitter") || getValue("x"),
        website: getValue("website") || getValue("web"),
      };
    }

    // Default mock data
    return {
      instagram: 0.78,
      youtube: 0.45,
      tiktok: 0.92,
      twitter: 0.34,
      website: 0.67,
    };
  }, [data, platforms]);

  const totalActivity = useMemo(() => {
    return Object.values(channelData).reduce((a, b) => a + b, 0) / Object.values(channelData).length;
  }, [channelData]);

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-sm font-medium">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-muted-foreground" />
              Channel Activity
            </div>
            <span className="text-xs text-muted-foreground">LOADING</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-32">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-sm font-medium">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-muted-foreground" />
            Channel Activity
          </div>
          <span className="text-xs text-muted-foreground flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            LIVE
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center py-2">
          <div className="text-3xl font-bold tabular-nums">
            {(totalActivity * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-muted-foreground">Density Index</div>
        </div>
        <div className="space-y-2">
          {Object.entries(channelData).map(([channel, value]) => (
            <div key={channel} className="flex items-center gap-2 text-sm">
              <span className="w-16 text-muted-foreground capitalize">{channel}</span>
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-foreground/60 rounded-full"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
              <span className="w-8 text-right tabular-nums text-muted-foreground">
                {(value * 100).toFixed(0)}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
