"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Share2, ZoomIn, ZoomOut, Maximize2, X, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface GraphNode {
  id: string;
  label: string;
  type: string;
  x: number;
  y: number;
  connections: string[];
  properties?: Record<string, unknown>;
}

interface GraphRelationship {
  source: string;
  target: string;
  type: string;
}

interface KnowledgeGraphCardProps {
  nodes?: GraphNode[];
  relationships?: GraphRelationship[];
  brandName?: string;
  isLoading?: boolean;
  fullGraphData?: {
    nodes: GraphNode[];
    relationships: GraphRelationship[];
    isLoading: boolean;
  };
}

// Neo4j Browser exact color palette
const NODE_COLORS: Record<string, string> = {
  // Brand - Blue (center)
  Brand: "#5C8EE6",
  brand: "#5C8EE6",
  // Content - Pink/Rose
  Content: "#F79CBE",
  content: "#F79CBE",
  Post: "#F79CBE",
  WebContent: "#F79CBE",
  // Concept - Teal/Cyan
  Concept: "#4ECDC4",
  concept: "#4ECDC4",
  Topic: "#4ECDC4",
  topic: "#4ECDC4",
  // Product - Yellow/Lime
  Product: "#C5E86C",
  product: "#C5E86C",
  // Interaction - Green
  Interaction: "#7BE394",
  interaction: "#7BE394",
  // Actor - Orange
  Actor: "#FFA07A",
  actor: "#FFA07A",
  ChatSession: "#FFA07A",
  ChatMessage: "#FFA07A",
};

const DEFAULT_COLOR = "#A0A0A0";

// Simulation node with physics
interface SimNode {
  id: string;
  label: string;
  type: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  fx?: number | null; // fixed x
  fy?: number | null; // fixed y
}

// Force-directed simulation
function useForceSimulation(
  nodes: GraphNode[],
  relationships: GraphRelationship[],
  width: number,
  height: number
) {
  const simNodesRef = useRef<Map<string, SimNode>>(new Map());
  const isInitializedRef = useRef(false);
  const iterationRef = useRef(0);

  const initializeNodes = useCallback(() => {
    if (nodes.length === 0) return;

    const simNodes = simNodesRef.current;
    const centerX = width / 2;
    const centerY = height / 2;

    // Find brand node (center)
    const brandNode = nodes.find(n => n.type === "Brand" || n.type === "brand");
    const brandId = brandNode?.id;

    // Build adjacency for BFS
    const adjacency = new Map<string, Set<string>>();
    nodes.forEach(n => adjacency.set(n.id, new Set()));
    relationships.forEach(rel => {
      adjacency.get(rel.source)?.add(rel.target);
      adjacency.get(rel.target)?.add(rel.source);
    });

    // BFS to find distance from center
    const distances = new Map<string, number>();
    if (brandId) {
      distances.set(brandId, 0);
      const queue = [brandId];
      while (queue.length > 0) {
        const current = queue.shift()!;
        const currentDist = distances.get(current)!;
        adjacency.get(current)?.forEach(neighbor => {
          if (!distances.has(neighbor)) {
            distances.set(neighbor, currentDist + 1);
            queue.push(neighbor);
          }
        });
      }
    }

    // Initialize positions based on distance from center
    nodes.forEach((node, i) => {
      if (simNodes.has(node.id)) {
        // Keep existing position but update label/type
        const existing = simNodes.get(node.id)!;
        existing.label = node.label;
        existing.type = node.type;
        return;
      }

      let x: number, y: number;
      const dist = distances.get(node.id) ?? 3;

      if (node.id === brandId) {
        // Brand at center
        x = centerX;
        y = centerY;
      } else {
        // Radial placement based on distance
        const radius = 60 + dist * 80 + Math.random() * 40;
        const angle = (i / nodes.length) * Math.PI * 2 + Math.random() * 0.5;
        x = centerX + Math.cos(angle) * radius;
        y = centerY + Math.sin(angle) * radius;
      }

      simNodes.set(node.id, {
        id: node.id,
        label: node.label,
        type: node.type,
        x,
        y,
        vx: 0,
        vy: 0,
        fx: node.id === brandId ? centerX : null,
        fy: node.id === brandId ? centerY : null,
      });
    });

    // Remove nodes that no longer exist
    simNodes.forEach((_, id) => {
      if (!nodes.find(n => n.id === id)) {
        simNodes.delete(id);
      }
    });

    isInitializedRef.current = true;
    iterationRef.current = 0;
  }, [nodes, relationships, width, height]);

  const tick = useCallback(() => {
    const simNodes = simNodesRef.current;
    if (simNodes.size === 0) return 0;

    const alpha = Math.max(0.001, 0.3 * Math.pow(0.99, iterationRef.current));
    iterationRef.current++;

    const centerX = width / 2;
    const centerY = height / 2;

    // Parameters tuned for Neo4j-like layout
    const repulsionStrength = 800;
    const attractionStrength = 0.015;
    const centerGravity = 0.008;
    const damping = 0.85;

    // Build edge lookup
    const edges = new Map<string, Set<string>>();
    simNodes.forEach((_, id) => edges.set(id, new Set()));
    relationships.forEach(rel => {
      if (simNodes.has(rel.source) && simNodes.has(rel.target)) {
        edges.get(rel.source)?.add(rel.target);
        edges.get(rel.target)?.add(rel.source);
      }
    });

    // Apply forces
    const nodeArray = Array.from(simNodes.values());

    // Repulsion (all pairs)
    for (let i = 0; i < nodeArray.length; i++) {
      for (let j = i + 1; j < nodeArray.length; j++) {
        const a = nodeArray[i];
        const b = nodeArray[j];

        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        // Coulomb's law style repulsion
        const force = (repulsionStrength * alpha) / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        if (a.fx == null) a.vx -= fx;
        if (a.fy == null) a.vy -= fy;
        if (b.fx == null) b.vx += fx;
        if (b.fy == null) b.vy += fy;
      }
    }

    // Attraction along edges (spring force)
    relationships.forEach(rel => {
      const source = simNodes.get(rel.source);
      const target = simNodes.get(rel.target);
      if (!source || !target) return;

      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;

      // Ideal distance based on node types
      const idealDist = 80;
      const force = (dist - idealDist) * attractionStrength * alpha;

      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;

      if (source.fx == null) source.vx += fx;
      if (source.fy == null) source.vy += fy;
      if (target.fx == null) target.vx -= fx;
      if (target.fy == null) target.vy -= fy;
    });

    // Center gravity
    nodeArray.forEach(node => {
      if (node.fx == null) {
        node.vx += (centerX - node.x) * centerGravity * alpha;
      }
      if (node.fy == null) {
        node.vy += (centerY - node.y) * centerGravity * alpha;
      }
    });

    // Apply velocities with damping
    let totalMovement = 0;
    const padding = 30;

    nodeArray.forEach(node => {
      if (node.fx != null) {
        node.x = node.fx;
        node.vx = 0;
      } else {
        node.vx *= damping;
        node.x += node.vx;
        node.x = Math.max(padding, Math.min(width - padding, node.x));
      }

      if (node.fy != null) {
        node.y = node.fy;
        node.vy = 0;
      } else {
        node.vy *= damping;
        node.y += node.vy;
        node.y = Math.max(padding, Math.min(height - padding, node.y));
      }

      totalMovement += Math.abs(node.vx) + Math.abs(node.vy);
    });

    return totalMovement;
  }, [relationships, width, height]);

  return { simNodesRef, initializeNodes, tick, isInitializedRef };
}

// Canvas-based graph renderer
function ForceGraph({
  nodes,
  relationships,
  width,
  height,
  brandName: _brandName,
  isLoading,
}: {
  nodes: GraphNode[];
  relationships: GraphRelationship[];
  width: number;
  height: number;
  brandName?: string;
  isLoading: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const animationRef = useRef<number | undefined>(undefined);

  const { simNodesRef, initializeNodes, tick, isInitializedRef } = useForceSimulation(
    nodes,
    relationships,
    width,
    height
  );

  // Initialize simulation when nodes change
  useEffect(() => {
    if (width > 0 && height > 0 && nodes.length > 0) {
      initializeNodes();
    }
  }, [nodes, relationships, width, height, initializeNodes]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width === 0 || height === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size with device pixel ratio
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    const render = () => {
      // Run simulation tick
      if (isInitializedRef.current) {
        tick();
      }

      const simNodes = simNodesRef.current;

      // Clear canvas with dark background
      ctx.fillStyle = "#2D2D2D";
      ctx.fillRect(0, 0, width, height);

      ctx.save();
      ctx.translate(width / 2 + pan.x, height / 2 + pan.y);
      ctx.scale(zoom, zoom);
      ctx.translate(-width / 2, -height / 2);

      // Draw edges
      ctx.strokeStyle = "rgba(128, 128, 128, 0.4)";
      ctx.lineWidth = 1;

      relationships.forEach(rel => {
        const source = simNodes.get(rel.source);
        const target = simNodes.get(rel.target);
        if (!source || !target) return;

        const isHighlighted =
          (hoveredNode && (rel.source === hoveredNode || rel.target === hoveredNode)) ||
          (selectedNode && (rel.source === selectedNode || rel.target === selectedNode));

        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);

        if (isHighlighted) {
          ctx.strokeStyle = "rgba(200, 200, 200, 0.8)";
          ctx.lineWidth = 1.5;
        } else {
          ctx.strokeStyle = "rgba(100, 100, 100, 0.5)";
          ctx.lineWidth = 0.8;
        }
        ctx.stroke();
      });

      // Draw nodes
      const nodeRadius = 8;
      const brandRadius = 12;

      simNodes.forEach((node, nodeId) => {
        const isBrand = node.type === "Brand" || node.type === "brand";
        const isHovered = hoveredNode === nodeId;
        const isSelected = selectedNode === nodeId;
        const radius = isBrand ? brandRadius : nodeRadius;

        // Check if connected to hovered/selected
        const isConnected = (hoveredNode || selectedNode) && relationships.some(
          r => (r.source === nodeId || r.target === nodeId) &&
               (r.source === (hoveredNode || selectedNode) || r.target === (hoveredNode || selectedNode))
        );

        const shouldDim = (hoveredNode || selectedNode) && !isHovered && !isSelected && !isConnected && nodeId !== (hoveredNode || selectedNode);

        // Get color
        const color = NODE_COLORS[node.type] || DEFAULT_COLOR;

        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);

        if (shouldDim) {
          ctx.globalAlpha = 0.2;
        } else {
          ctx.globalAlpha = 1;
        }

        // Fill
        ctx.fillStyle = color;
        ctx.fill();

        // Border
        if (isHovered || isSelected) {
          ctx.strokeStyle = "#FFFFFF";
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        ctx.globalAlpha = 1;

        // Label for hovered/selected
        if (isHovered || isSelected) {
          const label = node.label.length > 20 ? node.label.slice(0, 20) + "..." : node.label;
          ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
          const textWidth = ctx.measureText(label).width;

          // Background
          ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
          const labelX = node.x - textWidth / 2 - 6;
          const labelY = node.y - radius - 22;
          ctx.beginPath();
          ctx.roundRect(labelX, labelY, textWidth + 12, 18, 4);
          ctx.fill();

          // Text
          ctx.fillStyle = "#FFFFFF";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, node.x, labelY + 9);

          // Type badge
          ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
          ctx.fillStyle = color;
          ctx.fillText(node.type, node.x, node.y + radius + 12);
        }
      });

      ctx.restore();

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, relationships, width, height, zoom, pan, hoveredNode, selectedNode, tick, simNodesRef, isInitializedRef]);

  // Mouse handlers
  const getMousePos = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - width / 2 - pan.x) / zoom + width / 2;
    const y = (e.clientY - rect.top - height / 2 - pan.y) / zoom + height / 2;
    return { x, y };
  }, [width, height, zoom, pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging) {
      setPan(p => ({
        x: p.x + e.movementX,
        y: p.y + e.movementY,
      }));
      return;
    }

    const { x, y } = getMousePos(e);
    const simNodes = simNodesRef.current;
    let found: string | null = null;

    simNodes.forEach((node, nodeId) => {
      const dx = x - node.x;
      const dy = y - node.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const radius = (node.type === "Brand" || node.type === "brand") ? 12 : 8;
      if (dist < radius + 4) {
        found = nodeId;
      }
    });

    setHoveredNode(found);
    if (canvasRef.current) {
      canvasRef.current.style.cursor = found ? "pointer" : isDragging ? "grabbing" : "grab";
    }
  }, [isDragging, getMousePos, simNodesRef]);

  const handleMouseDown = useCallback(() => {
    if (hoveredNode) {
      setSelectedNode(prev => prev === hoveredNode ? null : hoveredNode);
    } else {
      setIsDragging(true);
    }
  }, [hoveredNode]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => Math.max(0.3, Math.min(3, z * delta)));
  }, []);

  const handleReset = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="relative h-full">
      {/* Controls */}
      <div className="absolute top-2 right-2 flex gap-1 z-10">
        <button
          onClick={() => setZoom(z => Math.min(z * 1.2, 3))}
          className="w-7 h-7 rounded bg-black/60 flex items-center justify-center text-white/70 hover:text-white hover:bg-black/80 transition-colors"
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => setZoom(z => Math.max(z * 0.8, 0.3))}
          className="w-7 h-7 rounded bg-black/60 flex items-center justify-center text-white/70 hover:text-white hover:bg-black/80 transition-colors"
        >
          <ZoomOut className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={handleReset}
          className="w-7 h-7 rounded bg-black/60 flex items-center justify-center text-white/70 hover:text-white hover:bg-black/80 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>

      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        className="w-full h-full rounded-lg"
        style={{ cursor: isDragging ? "grabbing" : "grab" }}
      />
    </div>
  );
}

// Full screen modal
function FullGraphModal({
  isOpen,
  onClose,
  nodes,
  relationships,
  brandName,
  isLoading,
}: {
  isOpen: boolean;
  onClose: () => void;
  nodes: GraphNode[];
  relationships: GraphRelationship[];
  brandName?: string;
  isLoading: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 800 });

  useEffect(() => {
    if (!isOpen || !containerRef.current) return;
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, [isOpen]);

  if (!isOpen) return null;

  // Get unique node types for legend
  const nodeTypes = [...new Set(nodes.map(n => n.type))];

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
      <div className="bg-[#2D2D2D] rounded-xl w-full h-full max-w-[95vw] max-h-[95vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
          <div>
            <h2 className="text-lg font-semibold text-white">Knowledge Graph</h2>
            <p className="text-sm text-white/50">
              {brandName} - {nodes.length} nodes, {relationships.length} relationships
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="h-5 w-5 text-white/70" />
          </button>
        </div>

        {/* Graph */}
        <div ref={containerRef} className="flex-1 overflow-hidden">
          <ForceGraph
            nodes={nodes}
            relationships={relationships}
            width={dimensions.width}
            height={dimensions.height}
            brandName={brandName}
            isLoading={isLoading}
          />
        </div>

        {/* Legend */}
        <div className="px-4 py-3 border-t border-white/10 flex flex-wrap gap-4">
          {nodeTypes.map(type => {
            const color = NODE_COLORS[type] || DEFAULT_COLOR;
            return (
              <span key={type} className="flex items-center gap-2 text-xs text-white/60">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: color }}
                />
                {type}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function KnowledgeGraphCard({
  nodes = [],
  relationships = [],
  brandName = "Brand",
  isLoading = false,
  fullGraphData,
}: KnowledgeGraphCardProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 400, height: 250 });
  const [showFullGraph, setShowFullGraph] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    updateDimensions();
    const observer = new ResizeObserver(updateDimensions);
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Get unique node types for legend
  const nodeTypes = [...new Set(nodes.map(n => n.type))].slice(0, 5);

  return (
    <>
      <Card className="h-full flex flex-col bg-card/50 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-sm font-medium">
            <div className="flex items-center gap-2">
              <Share2 className="h-4 w-4 text-muted-foreground" />
              Knowledge Graph
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowFullGraph(true)}
                className="p-1 rounded hover:bg-muted transition-colors"
                title="View Full Graph"
              >
                <Maximize2 className="h-3.5 w-3.5 text-muted-foreground" />
              </button>
              <span className="text-xs text-muted-foreground">{nodes.length} nodes</span>
              <span className="text-xs text-muted-foreground">{relationships.length} edges</span>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 pt-0 overflow-hidden">
          <div
            ref={containerRef}
            className="relative h-full min-h-[200px] rounded-lg overflow-hidden"
          >
            <ForceGraph
              nodes={nodes}
              relationships={relationships}
              width={dimensions.width}
              height={dimensions.height}
              brandName={brandName}
              isLoading={isLoading}
            />
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-3 mt-2 pt-2 border-t border-border/30">
            {nodeTypes.map(type => {
              const color = NODE_COLORS[type] || DEFAULT_COLOR;
              return (
                <span key={type} className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  {type}
                </span>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <FullGraphModal
        isOpen={showFullGraph}
        onClose={() => setShowFullGraph(false)}
        nodes={fullGraphData?.nodes?.length ? fullGraphData.nodes : nodes}
        relationships={fullGraphData?.relationships?.length ? fullGraphData.relationships : relationships}
        brandName={brandName}
        isLoading={fullGraphData?.isLoading || false}
      />
    </>
  );
}
