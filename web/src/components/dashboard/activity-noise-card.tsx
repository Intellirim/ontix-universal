"use client";

import { useEffect, useRef } from "react";
import type { Channel } from "@/types/dashboard";

interface ActivityNoiseCardProps {
  channel: Channel;
  seed: number;
}

export function ActivityNoiseCard({ channel, seed }: ActivityNoiseCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) {
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      }
    };
    resize();

    // Particle system for constellation effect
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      brightness: number;
      pulse: number;
    }> = [];

    const width = canvas.width / window.devicePixelRatio;
    const height = canvas.height / window.devicePixelRatio;

    // Initialize particles based on seed
    const random = (min: number, max: number) => {
      const x = Math.sin(seed++) * 10000;
      return min + (x - Math.floor(x)) * (max - min);
    };

    const particleCount = 60;
    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: random(0, width),
        y: random(0, height),
        vx: random(-0.15, 0.15),
        vy: random(-0.15, 0.15),
        size: random(1, 2.5),
        brightness: random(0.3, 1),
        pulse: random(0, Math.PI * 2),
      });
    }

    // Animation
    let time = 0;
    const animate = () => {
      ctx.clearRect(0, 0, width, height);
      time += 0.01;

      // Update and draw particles
      particles.forEach((p, i) => {
        // Update position
        p.x += p.vx;
        p.y += p.vy;

        // Wrap around edges
        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        // Pulsing brightness
        const pulse = Math.sin(time * 2 + p.pulse) * 0.3 + 0.7;
        const alpha = p.brightness * pulse;

        // Draw particle (star)
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(156, 163, 175, ${alpha * 0.8})`;
        ctx.fill();

        // Draw connections to nearby particles
        particles.slice(i + 1).forEach((p2) => {
          const dx = p2.x - p.x;
          const dy = p2.y - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 80) {
            const connectionAlpha = (1 - dist / 80) * 0.15 * pulse;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(156, 163, 175, ${connectionAlpha})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });

      // Draw some brighter "active" nodes with sapphire color
      const activeNodes = particles.slice(0, 5);
      activeNodes.forEach((p, i) => {
        const pulse = Math.sin(time * 3 + i) * 0.4 + 0.6;

        // Glow effect
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 15);
        gradient.addColorStop(0, `rgba(56, 189, 248, ${0.3 * pulse})`);
        gradient.addColorStop(1, "rgba(56, 189, 248, 0)");
        ctx.beginPath();
        ctx.arc(p.x, p.y, 15, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(56, 189, 248, ${pulse})`;
        ctx.fill();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    window.addEventListener("resize", resize);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      window.removeEventListener("resize", resize);
    };
  }, [seed]);

  const channelLabels: Record<Channel, string> = {
    instagram: "INSTAGRAM",
    tiktok: "TIKTOK",
    twitter: "X (TWITTER)",
    youtube: "YOUTUBE",
    web: "WEB",
  };

  return (
    <div className="ontix-card h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="label">ACTIVITY DENSITY</span>
        <span className="label-sm text-sapphire">{channelLabels[channel]}</span>
      </div>

      {/* Canvas */}
      <div className="canvas-container flex-1 min-h-[180px]">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
}
