// ============================================
// On-Chain Intelligence Types
// Based on ontix-onchain domain models
// ============================================

// Seed Categories
export type SeedCategory =
  | "exchange"
  | "defi"
  | "market_maker"
  | "hacker"
  | "bridge"
  | "whale"
  | "protocol";

// Collection Strategies
export type CollectionStrategy = "all" | "seeds" | "events" | "entities";

// Flow Direction
export type FlowDirection = "in" | "out" | "all";

// Chain Types
export type ChainType =
  | "ethereum"
  | "polygon"
  | "bsc"
  | "optimism"
  | "avalanche"
  | "arbitrum_one"
  | "bitcoin"
  | "tron"
  | "base"
  | "solana";

// ============================================
// Address & Entity Types
// ============================================

export interface ArkhamEntity {
  id: string;
  name: string;
  website?: string;
  twitter?: string;
  type?: string;
}

export interface ArkhamLabel {
  name: string;
  address: string;
  chain: string;
}

export interface OnChainAddress {
  address: string;
  chain: string;
  arkham_entity?: ArkhamEntity;
  arkham_label?: ArkhamLabel;
  is_contract: boolean;
  display_name: string;
}

// ============================================
// Transfer Types
// ============================================

export interface Transfer {
  id: string;
  transaction_hash: string;
  from_address: OnChainAddress;
  to_address: OnChainAddress;
  token_symbol?: string;
  historical_usd: number;
  chain: string;
  block_timestamp?: number;
}

export interface Counterparty {
  address: OnChainAddress;
  usd: number;
  transaction_count: number;
  flow: string;
  chains: string[];
}

// ============================================
// Analysis Types
// ============================================

export interface AddressAnalysis {
  address: string;
  display_name: string;
  entity?: string;
  counterparties: number;
  transfers: number;
  total_received_usd?: number;
  total_sent_usd?: number;
  first_seen?: string;
  last_active?: string;
  tags?: string[];
}

export interface FlowTraceResult {
  transfers: Transfer[];
  addresses: Record<string, OnChainAddress>;
  entities: Record<string, ArkhamEntity>;
  total_volume_usd: number;
}

// ============================================
// Query Types
// ============================================

export interface OnChainQueryRequest {
  question: string;
  use_langgraph?: boolean;
}

export interface OnChainQueryResponse {
  query: string;
  query_type: string;
  analysis: string;
  confidence: number;
  sources: string[];
  cypher_query?: string;
  addresses_found?: string[];
  graph_results_count: number;
  vector_results_count: number;
  error?: string;
}

// ============================================
// Collection Types
// ============================================

export interface CollectionRequest {
  strategy: CollectionStrategy;
  categories?: SeedCategory[];
  depth?: number;
  min_usd?: number;
}

export interface CollectionResult {
  strategy: string;
  started_at: string;
  completed_at?: string;
  duration_seconds: number;
  seeds_processed: number;
  addresses_collected: number;
  transfers_collected: number;
  entities_found: number;
  errors: string[];
}

export interface CollectionStats {
  request_count: number;
  collected_addresses: number;
  request_limit: number;
}

// ============================================
// Seed Types (for display)
// ============================================

export interface SeedAddress {
  address: string;
  name: string;
  category: SeedCategory;
  chain: string;
  priority: number;
  notes?: string;
}

export interface SeedEntity {
  entity_id: string;
  name: string;
  category: SeedCategory;
  priority: number;
}

// ============================================
// Dashboard Stats
// ============================================

export interface OnChainStats {
  total_wallets: number;
  total_transfers: number;
  total_entities: number;
  total_volume_usd: number;
  last_collection?: string;
  collection_status?: "idle" | "running" | "completed" | "error";
}

// ============================================
// Request/Response for API
// ============================================

export interface AnalyzeAddressRequest {
  address: string;
  chain?: ChainType;
}

export interface TraceFlowRequest {
  address: string;
  direction?: FlowDirection;
  depth?: number;
  min_usd?: number;
  chain?: ChainType;
}
