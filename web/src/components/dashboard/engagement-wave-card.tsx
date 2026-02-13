"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import type { EngagementDataPoint } from "@/types/dashboard";

interface EngagementWaveCardProps {
  data: EngagementDataPoint[];
  period?: string;
}

export function EngagementWaveCard({
  data,
  period = "LAST 7 DAYS",
}: EngagementWaveCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    data: EngagementDataPoint;
  } | null>(null);
  const [mounted, setMounted] = useState(false);

  // Hydration 안정성을 위해 클라이언트 마운트 후에만 동적 값 표시
  useEffect(() => {
    setMounted(true);
  }, []);

  // 서버/클라이언트 일관성을 위해 useMemo로 통계 미리 계산
  const stats = useMemo(() => {
    if (data.length === 0) return { avg: 0, peak: 0, total: 0 };
    const scores = data.map((d) => d.score);
    return {
      avg: Math.round(scores.reduce((a, b) => a + b, 0) / scores.length),
      peak: Math.max(...scores),
      total: data.reduce(
        (sum, d) => sum + (d.likes || 0) + (d.comments || 0) + (d.saves || 0),
        0
      ),
    };
  }, [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || data.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 20, right: 20, bottom: 40, left: 20 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Find min/max values
    const scores = data.map((d) => d.score);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    const scoreRange = maxScore - minScore || 1;

    // Calculate points
    const points = data.map((d, i) => ({
      x: padding.left + (i / (data.length - 1)) * chartWidth,
      y:
        padding.top +
        chartHeight -
        ((d.score - minScore) / scoreRange) * chartHeight,
      data: d,
    }));

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, "rgba(156, 163, 175, 0.15)");
    gradient.addColorStop(1, "rgba(156, 163, 175, 0)");

    ctx.beginPath();
    ctx.moveTo(points[0].x, height - padding.bottom);
    points.forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, height - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw wave line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    // Smooth curve using bezier
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i];
      const p1 = points[i + 1];
      const cpx = (p0.x + p1.x) / 2;

      ctx.bezierCurveTo(cpx, p0.y, cpx, p1.y, p1.x, p1.y);
    }

    ctx.strokeStyle =
      hoveredIndex !== null ? "#38BDF8" : "rgba(156, 163, 175, 0.6)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw data points
    points.forEach((p, i) => {
      const isHovered = i === hoveredIndex;
      const radius = isHovered ? 6 : 3;

      // Glow for hovered point
      if (isHovered) {
        const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 20);
        glow.addColorStop(0, "rgba(56, 189, 248, 0.3)");
        glow.addColorStop(1, "rgba(56, 189, 248, 0)");
        ctx.beginPath();
        ctx.arc(p.x, p.y, 20, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }

      // Point
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isHovered ? "#38BDF8" : "rgba(156, 163, 175, 0.8)";
      ctx.fill();

      if (isHovered) {
        ctx.strokeStyle = "#38BDF8";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

    // Draw timeline axis labels
    ctx.font = "10px system-ui";
    ctx.fillStyle = "#64748B";
    ctx.textAlign = "center";

    const labelIndices = [0, Math.floor(data.length / 2), data.length - 1];
    labelIndices.forEach((i) => {
      const p = points[i];
      const date = new Date(data[i].timestamp);
      const label = date.toLocaleDateString("ko-KR", {
        month: "short",
        day: "numeric",
      });
      ctx.fillText(label, p.x, height - 15);
    });
  }, [data, hoveredIndex]);

  // Mouse interaction
  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || data.length === 0) return;

    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const padding = { left: 20, right: 20 };
    const chartWidth = rect.width - padding.left - padding.right;

    // Find closest point
    const relativeX = (x - padding.left) / chartWidth;
    const index = Math.round(relativeX * (data.length - 1));

    if (index >= 0 && index < data.length) {
      setHoveredIndex(index);

      const pointX = padding.left + (index / (data.length - 1)) * chartWidth;
      setTooltip({
        x: pointX,
        y: e.clientY - rect.top - 60,
        data: data[index],
      });
    }
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
    setTooltip(null);
  };

  return (
    <div className="ontix-card h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="label">ENGAGEMENT WAVE</span>
        <span className="label-sm">{period}</span>
      </div>

      {/* Chart */}
      <div
        ref={containerRef}
        className="relative flex-1 min-h-[200px]"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <canvas ref={canvasRef} className="w-full h-full cursor-crosshair" />

        {/* Tooltip */}
        {tooltip && (
          <div
            className="absolute pointer-events-none z-10 px-3 py-2 rounded-lg"
            style={{
              left: tooltip.x,
              top: Math.max(10, tooltip.y),
              transform: "translateX(-50%)",
              background: "rgba(2, 6, 23, 0.95)",
              border: "1px solid rgba(56, 189, 248, 0.3)",
            }}
          >
            <div className="text-sapphire text-lg font-semibold">
              {tooltip.data.score}
            </div>
            <div className="text-xs text-secondary mt-1 space-y-0.5">
              <div>
                Likes: {tooltip.data.likes?.toLocaleString()}
              </div>
              <div>
                Comments: {tooltip.data.comments?.toLocaleString()}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-6 mt-4 pt-4 border-t border-[#0f172a]">
        {mounted && data.length > 0 && (
          <>
            <div>
              <span className="label-sm">AVG SCORE</span>
              <div className="text-[#E5E7EB] text-lg font-semibold mt-1">
                {stats.avg}
              </div>
            </div>
            <div>
              <span className="label-sm">PEAK</span>
              <div className="text-[#38BDF8] text-lg font-semibold mt-1">
                {stats.peak}
              </div>
            </div>
            <div>
              <span className="label-sm">TOTAL</span>
              <div className="text-[#E5E7EB] text-lg font-semibold mt-1">
                {stats.total.toLocaleString()}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
