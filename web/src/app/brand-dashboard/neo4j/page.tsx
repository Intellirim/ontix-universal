"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { useBrandStats } from "@/hooks/use-api";
import { useBrandContext } from "../layout";
import {
  Database,
  FileText,
  Users,
  ShoppingBag,
  MessageSquare,
  Globe,
  Hash,
  AlertCircle,
  RefreshCw,
  Maximize2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface GraphNode {
  id: string;
  label: string;
  type: string;
  x: number;
  y: number;
  size: number;
  count?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

const nodeTypeConfig: Record<
  string,
  { icon: typeof Database; color: string; label: string; graphColor: string }
> = {
  Content: {
    icon: FileText,
    color: "text-blue-600",
    label: "콘텐츠",
    graphColor: "#3B82F6",
  },
  Post: {
    icon: FileText,
    color: "text-blue-600",
    label: "포스트",
    graphColor: "#3B82F6",
  },
  Product: {
    icon: ShoppingBag,
    color: "text-emerald-600",
    label: "상품",
    graphColor: "#10B981",
  },
  Concept: {
    icon: Hash,
    color: "text-purple-600",
    label: "개념",
    graphColor: "#8B5CF6",
  },
  Brand: {
    icon: Database,
    color: "text-cyan-600",
    label: "브랜드",
    graphColor: "#0891B2",
  },
  ChatSession: {
    icon: MessageSquare,
    color: "text-amber-600",
    label: "채팅 세션",
    graphColor: "#D97706",
  },
  ChatMessage: {
    icon: MessageSquare,
    color: "text-amber-500",
    label: "채팅 메시지",
    graphColor: "#F59E0B",
  },
  WebContent: {
    icon: Globe,
    color: "text-cyan-600",
    label: "웹 콘텐츠",
    graphColor: "#06B6D4",
  },
  Actor: {
    icon: Users,
    color: "text-pink-600",
    label: "사용자",
    graphColor: "#EC4899",
  },
  Interaction: {
    icon: MessageSquare,
    color: "text-orange-600",
    label: "인터랙션",
    graphColor: "#EA580C",
  },
  Topic: {
    icon: Hash,
    color: "text-indigo-600",
    label: "토픽",
    graphColor: "#6366F1",
  },
};

// Neo4j Bloom style colors
const bloomColors: Record<string, { bg: string; border: string }> = {
  Brand: { bg: "#4C8EDA", border: "#3A7BC8" },
  Content: { bg: "#F16667", border: "#D94F50" },
  Post: { bg: "#F16667", border: "#D94F50" },
  WebContent: { bg: "#F16667", border: "#D94F50" },
  Product: { bg: "#ECB83D", border: "#D4A12E" },
  Concept: { bg: "#57C7E3", border: "#40B0CC" },
  Topic: { bg: "#57C7E3", border: "#40B0CC" },
  ChatSession: { bg: "#F79767", border: "#E08050" },
  ChatMessage: { bg: "#F79767", border: "#E08050" },
  Actor: { bg: "#D9C8AE", border: "#C2B197" },
  Interaction: { bg: "#8DCC93", border: "#76B57C" },
};

function KnowledgeGraphVisualization({
  nodes,
  edges,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const animationRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || nodes.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.scale(dpr, dpr);
    };
    resize();

    const width = container.getBoundingClientRect().width;
    const height = container.getBoundingClientRect().height;
    const padding = 80;

    // Calculate positions with larger spacing
    const nodePositions = new Map<string, { x: number; y: number; node: GraphNode; radius: number }>();
    nodes.forEach((node) => {
      const radius = node.type === "Brand" ? 35 : 18 + (node.count ? Math.min(node.count / 20, 12) : 0);
      nodePositions.set(node.id, {
        x: padding + node.x * (width - padding * 2),
        y: padding + node.y * (height - padding * 2),
        node,
        radius,
      });
    });

    const getConnectedEdges = (nodeId: string) => {
      return edges.filter((e) => e.source === nodeId || e.target === nodeId);
    };

    const animate = () => {
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Dark gradient background
      const bgGradient = ctx.createRadialGradient(
        width / 2, height / 2, 0,
        width / 2, height / 2, Math.max(width, height) / 2
      );
      bgGradient.addColorStop(0, "#1e2433");
      bgGradient.addColorStop(1, "#0f1319");
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, width, height);

      ctx.save();
      ctx.translate(width / 2, height / 2);
      ctx.scale(zoom, zoom);
      ctx.translate(-width / 2, -height / 2);

      // Draw edges
      edges.forEach((edge) => {
        const source = nodePositions.get(edge.source);
        const target = nodePositions.get(edge.target);
        if (!source || !target) return;

        const isHighlighted =
          (hoveredNode && (edge.source === hoveredNode || edge.target === hoveredNode)) ||
          (selectedNode && (edge.source === selectedNode || edge.target === selectedNode));

        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);

        if (isHighlighted) {
          ctx.strokeStyle = "rgba(87, 199, 227, 0.8)";
          ctx.lineWidth = 2.5;
        } else {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
          ctx.lineWidth = 1;
        }
        ctx.stroke();
      });

      // Draw nodes
      nodePositions.forEach((pos, nodeId) => {
        const node = pos.node;
        const isHovered = hoveredNode === nodeId;
        const isSelected = selectedNode === nodeId;
        const isConnected =
          (hoveredNode || selectedNode) &&
          getConnectedEdges(hoveredNode || selectedNode || "").some(
            (e) => e.source === nodeId || e.target === nodeId
          );
        const isHighlighted = isHovered || isSelected || isConnected;

        const colors = bloomColors[node.type] || { bg: "#A5ABB6", border: "#8E9AA6" };
        const radius = isHovered ? pos.radius * 1.15 : pos.radius;

        // Glow effect
        if (isHighlighted) {
          const glow = ctx.createRadialGradient(
            pos.x, pos.y, radius * 0.5,
            pos.x, pos.y, radius * 2.5
          );
          glow.addColorStop(0, colors.bg + "50");
          glow.addColorStop(1, "transparent");
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, radius * 2.5, 0, Math.PI * 2);
          ctx.fillStyle = glow;
          ctx.fill();
        }

        // Shadow
        ctx.beginPath();
        ctx.arc(pos.x + 2, pos.y + 3, radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
        ctx.fill();

        // Node gradient fill
        const nodeGradient = ctx.createRadialGradient(
          pos.x - radius * 0.3, pos.y - radius * 0.3, 0,
          pos.x, pos.y, radius
        );
        nodeGradient.addColorStop(0, colors.bg);
        nodeGradient.addColorStop(1, colors.border);

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = nodeGradient;
        ctx.fill();

        // Border
        ctx.strokeStyle = isHighlighted ? "#ffffff" : colors.border;
        ctx.lineWidth = isHighlighted ? 3 : 2;
        ctx.stroke();

        // Label for hovered/selected
        if (isHovered || isSelected) {
          const label = node.label.length > 20 ? node.label.slice(0, 20) + "..." : node.label;
          ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, sans-serif";
          const textWidth = ctx.measureText(label).width;

          // Label background
          const labelY = pos.y - radius - 15;
          ctx.fillStyle = "rgba(0, 0, 0, 0.9)";
          ctx.beginPath();
          ctx.roundRect(pos.x - textWidth / 2 - 10, labelY - 16, textWidth + 20, 28, 8);
          ctx.fill();

          // Label text
          ctx.fillStyle = "#ffffff";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, pos.x, labelY);

          // Count badge
          if (node.count) {
            ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
            ctx.fillStyle = colors.bg;
            ctx.fillText(`${node.count}개`, pos.x, pos.y + radius + 18);
          }
        }
      });

      ctx.restore();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = (e.clientX - rect.left - width / 2) / zoom + width / 2;
      const y = (e.clientY - rect.top - height / 2) / zoom + height / 2;

      let found: string | null = null;
      nodePositions.forEach((pos, nodeId) => {
        const dx = x - pos.x;
        const dy = y - pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < pos.radius + 8) {
          found = nodeId;
        }
      });
      setHoveredNode(found);
      canvas.style.cursor = found ? "pointer" : "default";
    };

    const handleClick = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = (e.clientX - rect.left - width / 2) / zoom + width / 2;
      const y = (e.clientY - rect.top - height / 2) / zoom + height / 2;

      let found: string | null = null;
      nodePositions.forEach((pos, nodeId) => {
        const dx = x - pos.x;
        const dy = y - pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < pos.radius + 8) {
          found = nodeId;
        }
      });
      setSelectedNode(found === selectedNode ? null : found);
    };

    const handleMouseLeave = () => {
      setHoveredNode(null);
    };

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom((prev) => Math.min(Math.max(prev * delta, 0.5), 3));
    };

    container.addEventListener("mousemove", handleMouseMove);
    container.addEventListener("click", handleClick);
    container.addEventListener("mouseleave", handleMouseLeave);
    container.addEventListener("wheel", handleWheel, { passive: false });
    window.addEventListener("resize", resize);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      container.removeEventListener("mousemove", handleMouseMove);
      container.removeEventListener("click", handleClick);
      container.removeEventListener("mouseleave", handleMouseLeave);
      container.removeEventListener("wheel", handleWheel);
      window.removeEventListener("resize", resize);
    };
  }, [nodes, edges, hoveredNode, selectedNode, zoom]);

  return (
    <div className="relative h-full">
      {/* Zoom controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
        <button
          onClick={() => setZoom((prev) => Math.min(prev * 1.2, 3))}
          className="w-8 h-8 rounded-lg bg-gray-800/80 backdrop-blur border border-gray-700 flex items-center justify-center text-gray-300 hover:text-white hover:border-cyan-500/50 transition-all"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((prev) => Math.max(prev * 0.8, 0.5))}
          className="w-8 h-8 rounded-lg bg-gray-800/80 backdrop-blur border border-gray-700 flex items-center justify-center text-gray-300 hover:text-white hover:border-cyan-500/50 transition-all"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom(1)}
          className="w-8 h-8 rounded-lg bg-gray-800/80 backdrop-blur border border-gray-700 flex items-center justify-center text-gray-300 hover:text-white hover:border-cyan-500/50 transition-all"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      <div ref={containerRef} className="w-full h-full">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
}

export default function Neo4jPage() {
  const { selectedBrandId } = useBrandContext();
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useBrandStats(selectedBrandId);

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  // Generate graph nodes and edges from stats.nodes data
  const { graphNodes, graphEdges } = useMemo(() => {
    // Use stats.nodes from brand stats API
    if (stats?.nodes && Object.keys(stats.nodes).length > 0) {
      const nodes: GraphNode[] = [];
      const edges: GraphEdge[] = [];
      const types = Object.entries(stats.nodes);
      const centerX = 0.5;
      const centerY = 0.5;

      // Add brand center node
      nodes.push({
        id: "brand_center",
        label: selectedBrandId,
        type: "Brand",
        x: centerX,
        y: centerY,
        size: 2.5,
      });

      // Add node type nodes around center
      types.forEach(([type, count], idx) => {
        const angle = (idx / types.length) * Math.PI * 2 - Math.PI / 2;
        const radius = 0.35;
        nodes.push({
          id: `type_${type}`,
          label: nodeTypeConfig[type]?.label || type,
          type: type,
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          size: Math.min(2, Math.max(0.8, Math.log10(count + 1) * 0.8)),
          count: count,
        });

        edges.push({
          source: "brand_center",
          target: `type_${type}`,
          weight: Math.min(1, count / 100),
        });
      });

      // Add connections between related node types
      const typeList = types.map(([type]) => type);
      if (typeList.includes("Content") && typeList.includes("Concept")) {
        edges.push({ source: "type_Content", target: "type_Concept", weight: 0.6 });
      }
      if (typeList.includes("Content") && typeList.includes("Interaction")) {
        edges.push({ source: "type_Content", target: "type_Interaction", weight: 0.5 });
      }

      return { graphNodes: nodes, graphEdges: edges };
    }
    return { graphNodes: [], graphEdges: [] };
  }, [stats, selectedBrandId]);

  // Node breakdown from stats
  const nodeBreakdown = useMemo(() => {
    if (stats?.nodes) {
      return stats.nodes;
    }
    return {};
  }, [stats]);

  const totalNodes = stats?.total_nodes || Object.values(nodeBreakdown).reduce((a, b) => a + b, 0);
  const totalRelationships = stats?.total_relationships || 0;

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-gray-900">
            Neo4j 지식 그래프
          </h1>
          <p className="text-xs sm:text-sm text-gray-500 mt-1">
            브랜드의 지식 그래프 데이터를 시각화합니다
          </p>
        </div>
        {selectedBrandId && (
          <button
            onClick={() => refetchStats()}
            className="flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-[#2563EB] text-white text-sm hover:bg-[#1D4ED8] transition-all w-full sm:w-auto"
          >
            <RefreshCw className="w-4 h-4" />
            새로고침
          </button>
        )}
      </div>

      {!selectedBrandId ? (
        <div className="p-8 sm:p-12 rounded-2xl bg-white border border-gray-200 shadow-sm text-center">
          <AlertCircle className="w-10 h-10 sm:w-12 sm:h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-base sm:text-lg font-medium text-gray-900 mb-2">
            브랜드를 선택하세요
          </h3>
          <p className="text-xs sm:text-sm text-gray-500">
            상단에서 브랜드를 선택한 후 데이터를 확인할 수 있습니다
          </p>
        </div>
      ) : statsLoading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-10 h-10 border-2 border-[#2563EB] border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          {/* Overview Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
            <div className="p-4 sm:p-6 rounded-2xl bg-white border border-gray-200 shadow-sm">
              <div className="flex items-center gap-2 sm:gap-3 mb-2 sm:mb-3">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-[#2563EB]/10 flex items-center justify-center">
                  <Database className="w-4 h-4 sm:w-5 sm:h-5 text-[#2563EB]" />
                </div>
                <span className="text-gray-500 text-xs sm:text-sm">총 노드</span>
              </div>
              <div className="text-xl sm:text-3xl font-bold text-gray-900">
                {formatNumber(totalNodes)}
              </div>
            </div>

            <div className="p-4 sm:p-6 rounded-2xl bg-white border border-gray-200 shadow-sm">
              <div className="flex items-center gap-2 sm:gap-3 mb-2 sm:mb-3">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-purple-100 flex items-center justify-center">
                  <Hash className="w-4 h-4 sm:w-5 sm:h-5 text-purple-600" />
                </div>
                <span className="text-gray-500 text-xs sm:text-sm">개념</span>
              </div>
              <div className="text-xl sm:text-3xl font-bold text-gray-900">
                {formatNumber(nodeBreakdown.Concept || 0)}
              </div>
            </div>

            <div className="p-4 sm:p-6 rounded-2xl bg-white border border-gray-200 shadow-sm">
              <div className="flex items-center gap-2 sm:gap-3 mb-2 sm:mb-3">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-blue-100 flex items-center justify-center">
                  <FileText className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600" />
                </div>
                <span className="text-gray-500 text-xs sm:text-sm">콘텐츠</span>
              </div>
              <div className="text-xl sm:text-3xl font-bold text-gray-900">
                {formatNumber(nodeBreakdown.Content || 0)}
              </div>
            </div>

            <div className="p-4 sm:p-6 rounded-2xl bg-white border border-gray-200 shadow-sm">
              <div className="flex items-center gap-2 sm:gap-3 mb-2 sm:mb-3">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-orange-100 flex items-center justify-center">
                  <MessageSquare className="w-4 h-4 sm:w-5 sm:h-5 text-orange-600" />
                </div>
                <span className="text-gray-500 text-xs sm:text-sm">인터랙션</span>
              </div>
              <div className="text-xl sm:text-3xl font-bold text-gray-900">
                {formatNumber(nodeBreakdown.Interaction || 0)}
              </div>
            </div>
          </div>

          {/* Knowledge Graph Visualization */}
          <div className="p-4 sm:p-6 rounded-2xl bg-white border border-gray-200 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-base sm:text-lg font-semibold text-gray-900">
                  지식 그래프
                </h3>
                <p className="text-[10px] sm:text-xs text-gray-500 mt-1">
                  스크롤로 확대/축소, 마우스를 올려 상세 정보 확인
                </p>
              </div>
              <div className="flex items-center gap-2 text-[10px] sm:text-xs text-gray-500">
                <span>{graphNodes.length} NODES</span>
                <span>/</span>
                <span>{graphEdges.length} EDGES</span>
              </div>
            </div>

            <div className="h-[300px] sm:h-[450px] rounded-xl bg-[#0f1319] border border-gray-800 overflow-hidden">
              {graphNodes.length > 0 ? (
                <KnowledgeGraphVisualization
                  nodes={graphNodes}
                  edges={graphEdges}
                />
              ) : (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center">
                    <Database className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                    <p className="text-sm text-gray-500">
                      그래프 데이터가 없습니다
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      파이프라인을 실행하여 데이터를 수집하세요
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Legend */}
            {Object.keys(nodeBreakdown).length > 0 && (
              <div className="flex flex-wrap items-center gap-4 mt-4 pt-4 border-t border-gray-100">
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-[#0891B2]" />
                  <span className="text-[10px] uppercase tracking-wider text-gray-500">
                    브랜드
                  </span>
                </div>
                {Object.entries(nodeBreakdown)
                  .filter(([, count]) => count > 0)
                  .map(([type, config]) => (
                    <div key={type} className="flex items-center gap-2">
                      <div
                        className="w-2.5 h-2.5"
                        style={{
                          backgroundColor: nodeTypeConfig[type]?.graphColor || "#6B7280",
                          borderRadius:
                            type === "Content" || type === "Post"
                              ? "0"
                              : type === "Concept" || type === "Topic"
                              ? "0"
                              : "50%",
                          transform:
                            type === "Concept" || type === "Topic"
                              ? "rotate(45deg)"
                              : "none",
                        }}
                      />
                      <span className="text-[10px] uppercase tracking-wider text-gray-500">
                        {nodeTypeConfig[type]?.label || type}
                      </span>
                    </div>
                  ))}
              </div>
            )}
          </div>

          {/* Node Type Breakdown */}
          {Object.keys(nodeBreakdown).length > 0 && (
            <div className="p-4 sm:p-6 rounded-2xl bg-white border border-gray-200 shadow-sm">
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-4 sm:mb-6">
                노드 유형별 분포
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 sm:gap-4">
                {Object.entries(nodeBreakdown)
                  .filter(([, count]) => count > 0)
                  .sort(([, a], [, b]) => b - a)
                  .map(([type, count]) => {
                    const config = nodeTypeConfig[type] || {
                      icon: Database,
                      color: "text-gray-500",
                      label: type,
                      graphColor: "#6B7280",
                    };
                    const Icon = config.icon;

                    return (
                      <div
                        key={type}
                        className="p-3 sm:p-4 rounded-xl bg-gray-50 border border-gray-100 hover:border-[#2563EB]/20 transition-colors"
                      >
                        <div className="flex items-center gap-2 mb-2 sm:mb-3">
                          <Icon className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${config.color}`} />
                          <span className="text-[10px] sm:text-xs text-gray-500">
                            {config.label}
                          </span>
                        </div>
                        <div className="text-lg sm:text-xl font-semibold text-gray-900">
                          {formatNumber(count)}
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Top Questions from Stats */}
          {stats?.top_questions && stats.top_questions.length > 0 && (
            <div className="p-4 sm:p-6 rounded-2xl bg-white border border-gray-200 shadow-sm">
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-3 sm:mb-4">
                자주 묻는 질문
              </h3>
              <div className="space-y-2">
                {stats.top_questions.map((question, idx) => (
                  <div
                    key={idx}
                    className="flex items-start sm:items-center gap-2 sm:gap-3 p-2 sm:p-3 rounded-lg bg-gray-50"
                  >
                    <span className="text-[10px] sm:text-xs text-[#2563EB] font-mono shrink-0">
                      #{idx + 1}
                    </span>
                    <span className="text-xs sm:text-sm text-gray-700">{question}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
