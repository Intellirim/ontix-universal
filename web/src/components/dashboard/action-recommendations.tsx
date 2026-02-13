"use client";

import type { ReactElement } from "react";
import type { ActionItem } from "@/types/dashboard";

interface ActionRecommendationsProps {
  actions: ActionItem[];
}

const ActionIcon = ({ type }: { type: ActionItem["type"] }) => {
  const icons: Record<ActionItem["type"], ReactElement> = {
    content: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
        <path d="M14 2v6h6M12 18v-6M9 15l3 3 3-3" />
      </svg>
    ),
    engagement: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
      </svg>
    ),
    analysis: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <circle cx="11" cy="11" r="8" />
        <path d="M21 21l-4.35-4.35" />
      </svg>
    ),
    campaign: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
      </svg>
    ),
  };
  return icons[type];
};

const PriorityIndicator = ({ priority }: { priority: ActionItem["priority"] }) => {
  const colors: Record<ActionItem["priority"], string> = {
    high: "bg-sapphire",
    medium: "bg-[#94A3B8]",
    low: "bg-[#475569]",
  };
  return <div className={`w-1.5 h-1.5 rounded-full ${colors[priority]}`} />;
};

export function ActionRecommendations({ actions }: ActionRecommendationsProps) {
  return (
    <div className="ontix-card h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="label">RECOMMENDED ACTIONS</span>
        <span className="label-sm">{actions.length} ITEMS</span>
      </div>

      {/* Action List */}
      <div className="flex-1 space-y-3">
        {actions.map((action, index) => (
          <div
            key={action.id}
            className="action-item animate-fade-in-up"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            {/* Icon */}
            <div className="action-icon">
              <ActionIcon type={action.type} />
            </div>

            {/* Content */}
            <div className="action-content">
              <div className="flex items-center gap-2 mb-1">
                <PriorityIndicator priority={action.priority} />
                <span className="action-title">{action.title}</span>
              </div>
              <p className="action-description">{action.description}</p>
            </div>

            {/* Arrow */}
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="w-4 h-4 action-arrow flex-shrink-0"
            >
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="mt-4 pt-4 border-t border-[#0f172a]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <PriorityIndicator priority="high" />
              <span className="label-sm">HIGH</span>
            </div>
            <div className="flex items-center gap-1.5">
              <PriorityIndicator priority="medium" />
              <span className="label-sm">MEDIUM</span>
            </div>
            <div className="flex items-center gap-1.5">
              <PriorityIndicator priority="low" />
              <span className="label-sm">LOW</span>
            </div>
          </div>
          <button className="text-xs text-sapphire hover:text-[#0ea5e9] transition-colors">
            View All →
          </button>
        </div>
      </div>
    </div>
  );
}
