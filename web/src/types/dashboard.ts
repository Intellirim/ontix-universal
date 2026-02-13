// ============================================
// ONTIX Dashboard Types
// ============================================

export type Channel = "instagram" | "tiktok" | "twitter" | "youtube" | "web";

export type OntixMode =
  | "advisor"
  | "analytics"
  | "product"
  | "content"
  | "onboarding"
  | "monitoring";

export type Grade = "A" | "B" | "C" | "D" | "E" | "F";

export interface OntixAnswer {
  mode: OntixMode;
  channel: Channel;
  grade: Grade;
  text: string;
  filterStatus: "passed" | "warning" | "low_confidence";
}

export interface EngagementDataPoint {
  timestamp: string;
  score: number;
  likes?: number;
  comments?: number;
  saves?: number;
  clicks?: number;
}

export interface GraphNode {
  id: string;
  type: "content" | "hashtag" | "audience" | "channel" | "topic" | "product" |
        "Brand" | "Content" | "Post" | "Product" | "Concept" | "Topic" |
        "ChatSession" | "ChatMessage" | "WebContent" | "Actor" | "Interaction" | string;
  label: string;
  x?: number;
  y?: number;
  size?: number;
  count?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface ActionItem {
  id: string;
  title: string;
  description: string;
  priority: "high" | "medium" | "low";
  type: "content" | "engagement" | "analysis" | "campaign";
}

export interface ChannelOverview {
  channel: Channel;
  engagementTimeline: EngagementDataPoint[];
  activityNoiseSeed: number;
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  topActions: ActionItem[];
}

export interface FilterState {
  period: "7d" | "14d" | "30d" | "90d";
  format: "all" | "image" | "video" | "carousel" | "story";
  segment: "all" | "organic" | "paid" | "ugc";
}

// Channel metadata
export const CHANNELS: { id: Channel; label: string; icon: string }[] = [
  { id: "instagram", label: "Instagram", icon: "instagram" },
  { id: "tiktok", label: "TikTok", icon: "tiktok" },
  { id: "twitter", label: "X", icon: "twitter" },
  { id: "youtube", label: "YouTube", icon: "youtube" },
  { id: "web", label: "Web", icon: "globe" },
];

// Mode metadata
export const MODES: { id: OntixMode; label: string; description: string }[] = [
  { id: "advisor", label: "Advisor", description: "전략 조언" },
  { id: "analytics", label: "Analytics", description: "데이터 분석" },
  { id: "product", label: "Product", description: "상품 추천" },
  { id: "content", label: "Content", description: "콘텐츠 작성" },
  { id: "onboarding", label: "Onboarding", description: "온보딩 안내" },
  { id: "monitoring", label: "Monitoring", description: "실시간 감시" },
];

// Grade metadata
export const GRADE_INFO: Record<Grade, { label: string; status: string }> = {
  A: { label: "High Confidence", status: "Filter Passed" },
  B: { label: "Good Confidence", status: "Filter Passed" },
  C: { label: "Moderate", status: "Review Suggested" },
  D: { label: "Low Confidence", status: "Limited Data" },
  E: { label: "Very Low", status: "Insufficient Data" },
  F: { label: "Unreliable", status: "Not Recommended" },
};
