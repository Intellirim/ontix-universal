import { type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "./card";

export interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  trend?: {
    value: number;
    label?: string;
  };
  className?: string;
}

export function StatCard({
  title,
  value,
  subtitle,
  icon,
  trend,
  className,
}: StatCardProps) {
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <p className="text-[11px] font-medium text-zinc-600 tracking-wide">
              {title}
            </p>
            <p className="mt-1.5 text-xl font-medium text-zinc-100 tracking-tight">
              {value}
            </p>
            {subtitle && (
              <p className="mt-0.5 text-[10px] text-zinc-600">{subtitle}</p>
            )}
            {trend && (
              <div className="mt-1.5 flex items-center gap-1">
                <span
                  className={cn(
                    "text-[10px] font-medium",
                    trend.value > 0 ? "text-[var(--sapphire-light)]" : "text-zinc-500"
                  )}
                >
                  {trend.value > 0 ? "+" : ""}
                  {trend.value}%
                </span>
                {trend.label && (
                  <span className="text-[10px] text-zinc-700">{trend.label}</span>
                )}
              </div>
            )}
          </div>
          {icon && (
            <div className="flex-shrink-0 p-2 rounded-md bg-white/[0.02]">
              <div className="text-zinc-500">{icon}</div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
