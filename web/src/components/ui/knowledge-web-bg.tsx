"use client";

import { useEffect, useRef, useCallback } from "react";

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  opacity: number;
}

interface KnowledgeWebBgProps {
  nodeCount?: number;
  className?: string;
  brightness?: "subtle" | "normal" | "bright";
}

export function KnowledgeWebBg({
  nodeCount = 25,
  className = "",
  brightness = "subtle",
}: KnowledgeWebBgProps) {
  // Brightness multipliers - subtle mode uses 0.03 opacity
  const brightnessConfig = {
    subtle: { line: 0.03, node: 0.08 },
    normal: { line: 0.06, node: 0.20 },
    bright: { line: 0.12, node: 0.35 },
  };
  const config = brightnessConfig[brightness];
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const animationRef = useRef<number>(0);
  const reducedMotionRef = useRef(false);

  const initNodes = useCallback(
    (width: number, height: number) => {
      const nodes: Node[] = [];
      for (let i = 0; i < nodeCount; i++) {
        nodes.push({
          x: Math.random() * width,
          y: Math.random() * height,
          // Very slow movement - ethereal floating
          vx: (Math.random() - 0.5) * 0.08,
          vy: (Math.random() - 0.5) * 0.08,
          // Node size - slightly larger for visibility
          radius: Math.random() * 2 + 1,
          opacity: Math.random() * 0.4 + 0.2,
        });
      }
      return nodes;
    },
    [nodeCount]
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      ctx.clearRect(0, 0, width, height);

      const nodes = nodesRef.current;
      const connectionDistance = 200;

      // Draw connections - sapphire blue #2563EB
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < connectionDistance) {
            const opacity = (1 - distance / connectionDistance) * config.line;
            ctx.strokeStyle = `rgba(37, 99, 235, ${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw nodes - sapphire blue #2563EB
      for (const node of nodes) {
        ctx.fillStyle = `rgba(37, 99, 235, ${node.opacity * config.node})`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fill();
      }
    },
    [config]
  );

  const animate = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      if (reducedMotionRef.current) {
        draw(ctx, width, height);
        return;
      }

      const nodes = nodesRef.current;

      // Update positions - very slow drift
      for (const node of nodes) {
        node.x += node.vx;
        node.y += node.vy;

        // Soft bounce off edges
        if (node.x < 0 || node.x > width) {
          node.vx *= -1;
          node.x = Math.max(0, Math.min(width, node.x));
        }
        if (node.y < 0 || node.y > height) {
          node.vy *= -1;
          node.y = Math.max(0, Math.min(height, node.y));
        }

        // Subtle opacity fluctuation
        node.opacity += (Math.random() - 0.5) * 0.002;
        node.opacity = Math.max(0.1, Math.min(0.4, node.opacity));
      }

      draw(ctx, width, height);
      animationRef.current = requestAnimationFrame(() =>
        animate(ctx, width, height)
      );
    },
    [draw]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Check for reduced motion preference
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedMotionRef.current = mediaQuery.matches;

    const handleMotionChange = (e: MediaQueryListEvent) => {
      reducedMotionRef.current = e.matches;
    };
    mediaQuery.addEventListener("change", handleMotionChange);

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();

      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      ctx.scale(dpr, dpr);

      nodesRef.current = initNodes(rect.width, rect.height);
    };

    resize();
    window.addEventListener("resize", resize);

    const rect = canvas.getBoundingClientRect();
    animate(ctx, rect.width, rect.height);

    return () => {
      window.removeEventListener("resize", resize);
      mediaQuery.removeEventListener("change", handleMotionChange);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [initNodes, animate]);

  return (
    <>
      {/* Neural Network Canvas - Light Theme */}
      <canvas
        ref={canvasRef}
        className={`fixed inset-0 w-full h-full pointer-events-none ${className}`}
        style={{ zIndex: 1 }}
        aria-hidden="true"
      />
    </>
  );
}
