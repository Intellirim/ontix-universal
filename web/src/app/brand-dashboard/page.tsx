"use client";

import { useBrandContext } from "./layout";
import { useBrand, useBrandStats, useBrandAnalytics, useKnowledgeGraph, useSocialMonitoring } from "@/hooks/use-api";
import {
  ChannelActivityNoiseCard,
  EngagementWaveCard,
  KnowledgeGraphCard,
  AnswerPanelCard,
} from "@/components/control-room";

export default function BrandDashboard() {
  const { selectedBrandId } = useBrandContext();
  const { data: brand } = useBrand(selectedBrandId);
  const { data: stats } = useBrandStats(selectedBrandId);
  const { data: analytics } = useBrandAnalytics(selectedBrandId, 30);
  const { data: socialData, isLoading: socialLoading } = useSocialMonitoring(selectedBrandId, 7);
  // Card shows balanced nodes for diverse colors (Brand + Content + Concept + Interaction)
  const { data: graphData, isLoading: graphLoading } = useKnowledgeGraph(selectedBrandId, { limit: 50, balanced: true });
  // Modal shows full graph (500 nodes, all types)
  const { data: fullGraphData, isLoading: fullGraphLoading } = useKnowledgeGraph(selectedBrandId, { limit: 500 });

  const engagementData = analytics?.daily_stats?.map((day: { date: string; sessions: number; messages: number }) => ({
    date: day.date,
    engagement: day.sessions * 10 + day.messages,
    sessions: day.sessions,
  }));

  // Use real Neo4j graph data if available, otherwise fall back to stats-based graph
  const graphNodes = graphData?.nodes?.length
    ? graphData.nodes.map((node) => ({
        id: node.id,
        label: node.label || node.properties?.name as string || node.type,
        type: node.type,
        x: 0,
        y: 0,
        connections: [] as string[],
        properties: node.properties,
      }))
    : stats?.nodes
    ? (() => {
        const entries = Object.entries(stats.nodes).filter(([type]) => type !== "Brand");
        const nodes: Array<{ id: string; label: string; type: string; x: number; y: number; connections: string[] }> = entries.map(([type], idx) => {
          const angle = (idx / entries.length) * Math.PI * 2;
          const radius = 0.35;
          return {
            id: `node-${type}`,
            label: type,
            type: type === "Content" || type === "Post" ? "content"
              : type === "Product" ? "product"
              : type === "Concept" || type === "Topic" ? "concept"
              : "topic",
            x: 0.5 + Math.cos(angle) * radius,
            y: 0.5 + Math.sin(angle) * radius,
            connections: ["center-brand"],
          };
        });
        nodes.unshift({
          id: "center-brand",
          label: brand?.name || "Brand",
          type: "Brand",
          x: 0.5,
          y: 0.5,
          connections: nodes.map((n) => n.id),
        });
        return nodes;
      })()
    : undefined;

  const graphRelationships = graphData?.relationships?.map((rel) => ({
    source: rel.source,
    target: rel.target,
    type: rel.type,
  })) || [];

  // Full graph data for the modal - directly from Neo4j
  const fullGraphForModal = {
    nodes: fullGraphData?.nodes?.map((node) => ({
      id: node.id,
      label: node.label || (node.properties?.name as string) || node.type,
      type: node.type,
      x: 0,
      y: 0,
      connections: [] as string[],
      properties: node.properties,
    })) || [],
    relationships: fullGraphData?.relationships?.map((rel) => ({
      source: rel.source,
      target: rel.target,
      type: rel.type,
    })) || [],
    isLoading: fullGraphLoading,
  };

  return (
    <div className="min-h-screen bg-background p-3 sm:p-6 pt-16 sm:pt-20">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <h1 className="text-lg sm:text-xl font-semibold text-foreground">{brand?.name || "Brand"} Control Room</h1>
        <p className="text-xs sm:text-sm text-muted-foreground">Real-time brand intelligence dashboard</p>
      </div>

      {/* Mobile Layout - Stack vertically */}
      <div className="flex flex-col gap-4 lg:hidden">
        <div className="h-[200px] sm:h-[260px]">
          <ChannelActivityNoiseCard platforms={socialData?.platforms} isLoading={socialLoading} />
        </div>
        <div className="h-[200px] sm:h-[260px]">
          <EngagementWaveCard data={engagementData} />
        </div>
        <div className="h-[400px] sm:h-[500px]">
          <AnswerPanelCard brandId={selectedBrandId} brandName={brand?.name} />
        </div>
        <div className="h-[350px] sm:h-[450px]">
          <KnowledgeGraphCard nodes={graphNodes} relationships={graphRelationships} brandName={brand?.name} isLoading={graphLoading} fullGraphData={fullGraphForModal} />
        </div>
      </div>

      {/* Desktop Grid Layout */}
      <div
        className="hidden lg:grid gap-4"
        style={{
          gridTemplateColumns: "1fr 1.5fr 1fr",
          gridTemplateRows: "260px 560px",
        }}
      >
        <div><ChannelActivityNoiseCard
          platforms={socialData?.platforms}
          isLoading={socialLoading}
        /></div>
        <div><EngagementWaveCard data={engagementData} /></div>
        <div className="row-span-2"><AnswerPanelCard brandId={selectedBrandId} brandName={brand?.name} /></div>
        <div className="col-span-2"><KnowledgeGraphCard nodes={graphNodes} relationships={graphRelationships} brandName={brand?.name} isLoading={graphLoading} fullGraphData={fullGraphForModal} /></div>
      </div>

      {/* Footer */}
      <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-xs text-muted-foreground">
        <div className="flex gap-4">
          <span>Nodes: {stats?.total_nodes?.toLocaleString() || 0}</span>
          <span>Sessions: {stats?.total_conversations?.toLocaleString() || analytics?.total_sessions?.toLocaleString() || 0}</span>
        </div>
        <span>System Operational</span>
      </div>
    </div>
  );
}
