"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";

// ============================================
// Stats & Health
// ============================================

export function useOnChainStats() {
  return useQuery({
    queryKey: ["onchain", "stats"],
    queryFn: () => apiClient.getOnChainStats(),
    refetchInterval: 30 * 1000, // 30 seconds
    staleTime: 10 * 1000,
  });
}

// ============================================
// Address Analysis
// ============================================

export function useAnalyzeAddress(address: string | null, chain: string = "ethereum") {
  return useQuery({
    queryKey: ["onchain", "address", address, chain],
    queryFn: () => apiClient.analyzeAddress(address!, chain),
    enabled: !!address && address.length >= 40,
    staleTime: 5 * 60 * 1000, // 5 minutes cache
    retry: 1,
  });
}

export function useAnalyzeAddressMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ address, chain }: { address: string; chain?: string }) =>
      apiClient.analyzeAddress(address, chain),
    onSuccess: (data, variables) => {
      queryClient.setQueryData(
        ["onchain", "address", variables.address, variables.chain || "ethereum"],
        data
      );
    },
  });
}

// ============================================
// Flow Tracing
// ============================================

export function useTraceFlow(params: {
  address: string;
  direction?: "in" | "out" | "all";
  depth?: number;
  min_usd?: number;
  chain?: string;
} | null) {
  return useQuery({
    queryKey: ["onchain", "trace", params],
    queryFn: () => apiClient.traceFlow(params!),
    enabled: !!params?.address,
    staleTime: 5 * 60 * 1000,
  });
}

export function useTraceFlowMutation() {
  return useMutation({
    mutationFn: (params: {
      address: string;
      direction?: "in" | "out" | "all";
      depth?: number;
      min_usd?: number;
      chain?: string;
    }) => apiClient.traceFlow(params),
  });
}

// ============================================
// Natural Language Query
// ============================================

export function useOnChainQuery() {
  return useMutation({
    mutationFn: ({ question, useLanggraph = true }: { question: string; useLanggraph?: boolean }) =>
      apiClient.queryOnChain(question, useLanggraph),
  });
}

// ============================================
// Data Collection
// ============================================

export function useCollectionStatus(jobId?: string) {
  return useQuery({
    queryKey: ["onchain", "collection", "status", jobId],
    queryFn: () => apiClient.getCollectionStatus(jobId),
    refetchInterval: (query) => {
      // Poll faster when collection is running
      const data = query.state.data;
      if (data?.status === "running") return 5000; // 5 seconds
      return 30000; // 30 seconds
    },
    staleTime: 5000,
  });
}

export function useStartCollection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (params: {
      strategy: "all" | "seeds" | "events" | "entities";
      categories?: string[];
      depth?: number;
      min_usd?: number;
    }) => apiClient.startCollection(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["onchain", "collection", "status"] });
    },
  });
}

// ============================================
// Seeds & Entities
// ============================================

export function useSeeds(params?: { category?: string; priority?: number }) {
  return useQuery({
    queryKey: ["onchain", "seeds", params],
    queryFn: () => apiClient.getSeeds(params),
    staleTime: 10 * 60 * 1000, // 10 minutes (seeds don't change often)
  });
}

// ============================================
// Wallet Graph Visualization
// ============================================

export function useWalletGraph(address: string | null, depth: number = 2) {
  return useQuery({
    queryKey: ["onchain", "graph", address, depth],
    queryFn: () => apiClient.getWalletGraph(address!, depth),
    enabled: !!address && address.length >= 40,
    staleTime: 5 * 60 * 1000,
  });
}

export function useWalletGraphMutation() {
  return useMutation({
    mutationFn: ({ address, depth = 2 }: { address: string; depth?: number }) =>
      apiClient.getWalletGraph(address, depth),
  });
}
