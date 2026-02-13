"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { GraphNode, GraphEdge } from "@/types/dashboard";

interface KnowledgeGraphCardProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface NodePosition {
  x: number;
  y: number;
  vx: number;
  vy: number;
  node: GraphNode;
  radius: number;
}

// Neo4j Bloom style colors
const NODE_COLORS: Record<string, { bg: string; border: string }> = {
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

const DEFAULT_COLOR = { bg: "#A5ABB6", border: "#8E9AA6" };

export function KnowledgeGraphCard({ nodes, edges }: KnowledgeGraphCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const positionsRef = useRef<Map<string, NodePosition>>(new Map());
  const animationRef = useRef<number | undefined>(undefined);
  const isSimulatingRef = useRef(true);
  const scaleRef = useRef(1);
  const offsetRef = useRef({ x: 0, y: 0 });

  // Force-directed layout simulation
  const runSimulation = useCallback((
    positions: Map<string, NodePosition>,
    width: number,
    height: number
  ) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const k = Math.sqrt((width * height) / Math.max(nodes.length, 1)) * 0.6;

    // Repulsion between all nodes
    positions.forEach((posA, idA) => {
      positions.forEach((posB, idB) => {
        if (idA === idB) return;

        const dx = posA.x - posB.x;
        const dy = posA.y - posB.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const minDist = posA.radius + posB.radius + 20;

        if (dist < minDist * 3) {
          const force = (k * k) / dist;
          posA.vx += (dx / dist) * force * 0.015;
          posA.vy += (dy / dist) * force * 0.015;
        }
      });
    });

    // Attraction along edges
    edges.forEach((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return;

      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const idealDist = k * 1.2;
      const force = (dist - idealDist) * 0.01;

      source.vx += (dx / dist) * force;
      source.vy += (dy / dist) * force;
      target.vx -= (dx / dist) * force;
      target.vy -= (dy / dist) * force;
    });

    // Center gravity
    positions.forEach((pos) => {
      const dx = centerX - pos.x;
      const dy = centerY - pos.y;
      pos.vx += dx * 0.0008;
      pos.vy += dy * 0.0008;
    });

    // Apply velocities with damping
    let totalMovement = 0;
    const margin = 50;
    positions.forEach((pos) => {
      pos.vx *= 0.88;
      pos.vy *= 0.88;

      pos.x += pos.vx;
      pos.y += pos.vy;

      pos.x = Math.max(margin, Math.min(width - margin, pos.x));
      pos.y = Math.max(margin, Math.min(height - margin, pos.y));

      totalMovement += Math.abs(pos.vx) + Math.abs(pos.vy);
    });

    return totalMovement;
  }, [nodes.length, edges]);

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
    const centerX = width / 2;
    const centerY = height / 2;

    // Initialize positions
    const positions = new Map<string, NodePosition>();
    const nodesByType: Record<string, GraphNode[]> = {};

    nodes.forEach((node) => {
      if (!nodesByType[node.type]) nodesByType[node.type] = [];
      nodesByType[node.type].push(node);
    });

    const brandNode = nodes.find((n) => n.type === "Brand");
    let typeIndex = 0;
    const types = Object.keys(nodesByType).filter((t) => t !== "Brand");

    // Place brand at center with larger radius
    if (brandNode) {
      positions.set(brandNode.id, {
        x: centerX,
        y: centerY,
        vx: 0,
        vy: 0,
        node: brandNode,
        radius: 24,
      });
    }

    // Place other nodes
    types.forEach((type) => {
      const typeNodes = nodesByType[type];
      const baseAngle = (typeIndex / types.length) * Math.PI * 2 - Math.PI / 2;
      const radius = Math.min(width, height) * 0.32;

      typeNodes.forEach((node, i) => {
        const angleSpread = (Math.PI * 0.5) / Math.max(typeNodes.length, 1);
        const angle = baseAngle + (i - (typeNodes.length - 1) / 2) * angleSpread;
        const r = radius + (Math.random() - 0.5) * 30;

        positions.set(node.id, {
          x: centerX + Math.cos(angle) * r,
          y: centerY + Math.sin(angle) * r,
          vx: 0,
          vy: 0,
          node,
          radius: 12,
        });
      });
      typeIndex++;
    });

    positionsRef.current = positions;
    isSimulatingRef.current = true;

    const getConnectedEdges = (nodeId: string) => {
      return edges.filter((e) => e.source === nodeId || e.target === nodeId);
    };

    let simulationSteps = 0;
    const maxSimulationSteps = 250;

    const animate = () => {
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Dark background with subtle gradient
      const gradient = ctx.createRadialGradient(
        width / 2, height / 2, 0,
        width / 2, height / 2, Math.max(width, height) / 2
      );
      gradient.addColorStop(0, "#1a1f2e");
      gradient.addColorStop(1, "#0d1117");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Run simulation
      if (isSimulatingRef.current && simulationSteps < maxSimulationSteps) {
        const movement = runSimulation(positionsRef.current, width, height);
        simulationSteps++;
        if (movement < 0.3 || simulationSteps >= maxSimulationSteps) {
          isSimulatingRef.current = false;
        }
      }

      // Draw edges first (behind nodes)
      edges.forEach((edge) => {
        const source = positionsRef.current.get(edge.source);
        const target = positionsRef.current.get(edge.target);
        if (!source || !target) return;

        const isHighlighted =
          (hoveredNode && (edge.source === hoveredNode || edge.target === hoveredNode)) ||
          (selectedNode && (edge.source === selectedNode || edge.target === selectedNode));

        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);

        if (isHighlighted) {
          ctx.strokeStyle = "rgba(87, 199, 227, 0.8)";
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
          ctx.lineWidth = 1;
        }
        ctx.stroke();
      });

      // Draw nodes
      positionsRef.current.forEach((pos, nodeId) => {
        const node = pos.node;
        const isHovered = hoveredNode === nodeId;
        const isSelected = selectedNode === nodeId;
        const isConnected =
          (hoveredNode || selectedNode) &&
          getConnectedEdges(hoveredNode || selectedNode || "").some(
            (e) => e.source === nodeId || e.target === nodeId
          );
        const isHighlighted = isHovered || isSelected || isConnected;

        const colors = NODE_COLORS[node.type] || DEFAULT_COLOR;
        const radius = isHovered ? pos.radius * 1.2 : pos.radius;

        // Glow effect for highlighted nodes
        if (isHighlighted) {
          const glow = ctx.createRadialGradient(
            pos.x, pos.y, radius * 0.5,
            pos.x, pos.y, radius * 2.5
          );
          glow.addColorStop(0, colors.bg + "60");
          glow.addColorStop(1, "transparent");
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, radius * 2.5, 0, Math.PI * 2);
          ctx.fillStyle = glow;
          ctx.fill();
        }

        // Node shadow
        ctx.beginPath();
        ctx.arc(pos.x + 2, pos.y + 2, radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx.fill();

        // Node fill with gradient
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

        // Node border
        ctx.strokeStyle = isHighlighted ? "#ffffff" : colors.border;
        ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
        ctx.stroke();

        // Label for hovered/selected node
        if (isHovered || isSelected) {
          const label = node.label.length > 25 ? node.label.slice(0, 25) + "..." : node.label;
          ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          const textWidth = ctx.measureText(label).width;

          // Label background
          const labelY = pos.y - radius - 12;
          ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
          ctx.beginPath();
          ctx.roundRect(pos.x - textWidth / 2 - 8, labelY - 14, textWidth + 16, 24, 6);
          ctx.fill();

          // Label text
          ctx.fillStyle = "#ffffff";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, pos.x, labelY);

          // Node type badge
          ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
          ctx.fillStyle = colors.bg;
          ctx.fillText(node.type, pos.x, pos.y + radius + 14);
        }
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Mouse interaction
    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      let found: string | null = null;
      positionsRef.current.forEach((pos, nodeId) => {
        const dx = x - pos.x;
        const dy = y - pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < pos.radius + 5) {
          found = nodeId;
        }
      });
      setHoveredNode(found);
      canvas.style.cursor = found ? "pointer" : "default";
    };

    const handleClick = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      let found: string | null = null;
      positionsRef.current.forEach((pos, nodeId) => {
        const dx = x - pos.x;
        const dy = y - pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < pos.radius + 5) {
          found = nodeId;
        }
      });
      setSelectedNode(found === selectedNode ? null : found);
    };

    const handleMouseLeave = () => {
      setHoveredNode(null);
    };

    container.addEventListener("mousemove", handleMouseMove);
    container.addEventListener("click", handleClick);
    container.addEventListener("mouseleave", handleMouseLeave);
    window.addEventListener("resize", resize);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      container.removeEventListener("mousemove", handleMouseMove);
      container.removeEventListener("click", handleClick);
      container.removeEventListener("mouseleave", handleMouseLeave);
      window.removeEventListener("resize", resize);
    };
  }, [nodes, edges, hoveredNode, selectedNode, runSimulation]);

  // Get unique node types for legend
  const uniqueTypes = [...new Set(nodes.map(n => n.type))];

  return (
    <div className="ontix-card h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-medium text-foreground/80">KNOWLEDGE GRAPH</span>
        </div>
        <span className="text-[10px] text-muted-foreground">
          {nodes.length} nodes · {edges.length} relationships
        </span>
      </div>

      {/* Canvas */}
      <div
        ref={containerRef}
        className="flex-1 min-h-[300px] rounded-lg overflow-hidden"
      >
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-3 mt-3 pt-3 border-t border-border/20">
        {uniqueTypes.slice(0, 5).map((type) => {
          const colors = NODE_COLORS[type] || DEFAULT_COLOR;
          return (
            <div key={type} className="flex items-center gap-1.5">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: colors.bg, boxShadow: `0 0 4px ${colors.bg}40` }}
              />
              <span className="text-[10px] text-muted-foreground">{type}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
