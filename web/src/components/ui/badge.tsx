import { type HTMLAttributes } from "react";
import { cn, getGradeClass } from "@/lib/utils";
import type { FilterGrade, AlertSeverity } from "@/types";

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "outline" | "subtle";
}

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 text-[10px] font-medium rounded",
        variant === "default" && "bg-white/[0.04] text-zinc-400",
        variant === "outline" && "border border-white/[0.06] text-zinc-500",
        variant === "subtle" && "text-zinc-500",
        className
      )}
      {...props}
    />
  );
}

export interface GradeBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  grade: FilterGrade | string;
}

export function GradeBadge({ grade, className, ...props }: GradeBadgeProps) {
  return (
    <span
      className={cn("grade-badge", getGradeClass(grade), className)}
      {...props}
    >
      {grade}
    </span>
  );
}

export interface SeverityBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  severity: AlertSeverity;
}

export function SeverityBadge({
  severity,
  className,
  ...props
}: SeverityBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs text-zinc-400",
        className
      )}
      {...props}
    >
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          severity === "info" && "bg-[var(--sapphire-light)]",
          severity === "warning" && "bg-zinc-400",
          severity === "error" && "bg-zinc-500",
          severity === "critical" && "bg-zinc-300"
        )}
      />
      {severity}
    </span>
  );
}

export interface StatusBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  status: "healthy" | "degraded" | "unhealthy" | "unavailable" | string;
}

export function StatusBadge({ status, className, ...props }: StatusBadgeProps) {
  const normalizedStatus = status.toLowerCase();
  const isHealthy = normalizedStatus === "healthy" || normalizedStatus === "ok";
  const isDegraded = normalizedStatus === "degraded";
  const isUnavailable = normalizedStatus === "unavailable" || normalizedStatus === "offline";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs",
        isHealthy && "text-zinc-300",
        isDegraded && "text-zinc-400",
        !isHealthy && !isDegraded && "text-zinc-500",
        className
      )}
      {...props}
    >
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          isHealthy && "bg-[var(--sapphire-light)]",
          isDegraded && "bg-zinc-400",
          isUnavailable && "bg-zinc-600",
          !isHealthy && !isDegraded && !isUnavailable && "bg-zinc-500"
        )}
      />
      {isHealthy ? "정상" : isDegraded ? "저하" : isUnavailable ? "오프라인" : status}
    </span>
  );
}
