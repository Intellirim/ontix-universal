"use client";

import type { ReactElement } from "react";
import type { Channel, OntixMode, FilterState } from "@/types/dashboard";
import { CHANNELS, MODES } from "@/types/dashboard";

interface FilterModePanelProps {
  activeChannel: Channel;
  activeMode: OntixMode;
  filters: FilterState;
  onChannelChange: (channel: Channel) => void;
  onModeChange: (mode: OntixMode) => void;
  onFilterChange: (filters: Partial<FilterState>) => void;
}

// Simple icons as SVG
const ChannelIcon = ({ channel }: { channel: Channel }) => {
  const icons: Record<Channel, ReactElement> = {
    instagram: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <rect x="2" y="2" width="20" height="20" rx="5" />
        <circle cx="12" cy="12" r="4" />
        <circle cx="18" cy="6" r="1.5" fill="currentColor" />
      </svg>
    ),
    tiktok: (
      <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
        <path d="M19.59 6.69a4.83 4.83 0 01-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 01-5.2 1.74 2.89 2.89 0 012.31-4.64c.298-.002.595.042.88.13V9.4a6.33 6.33 0 00-1-.05A6.34 6.34 0 005 20.1a6.34 6.34 0 0010.86-4.43v-7a8.16 8.16 0 004.77 1.52v-3.4a4.85 4.85 0 01-1-.1z" />
      </svg>
    ),
    twitter: (
      <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
      </svg>
    ),
    youtube: (
      <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
        <path d="M23.498 6.186a3.016 3.016 0 00-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 00.502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 002.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 002.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
      </svg>
    ),
    web: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <circle cx="12" cy="12" r="10" />
        <path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z" />
      </svg>
    ),
  };
  return icons[channel];
};

const ModeIcon = ({ mode }: { mode: OntixMode }) => {
  const icons: Record<OntixMode, ReactElement> = {
    advisor: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mode-tab-icon">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
    analytics: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mode-tab-icon">
        <path d="M3 3v18h18" />
        <path d="M18 17V9M13 17V5M8 17v-3" />
      </svg>
    ),
    product: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mode-tab-icon">
        <path d="M6 2L3 6v14a2 2 0 002 2h14a2 2 0 002-2V6l-3-4z" />
        <path d="M3 6h18M16 10a4 4 0 01-8 0" />
      </svg>
    ),
    content: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mode-tab-icon">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
        <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
      </svg>
    ),
    onboarding: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mode-tab-icon">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 16v-4M12 8h.01" />
      </svg>
    ),
    monitoring: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mode-tab-icon">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
  };
  return icons[mode];
};

export function FilterModePanel({
  activeChannel,
  activeMode,
  filters,
  onChannelChange,
  onModeChange,
  onFilterChange,
}: FilterModePanelProps) {
  return (
    <div className="ontix-card h-full flex flex-col">
      {/* Channel Tabs */}
      <div className="mb-6">
        <span className="label mb-3 block">CHANNEL</span>
        <div className="channel-tabs">
          {CHANNELS.map((ch) => (
            <button
              key={ch.id}
              onClick={() => onChannelChange(ch.id)}
              className={`channel-tab flex items-center gap-2 ${
                activeChannel === ch.id ? "channel-tab-active" : ""
              }`}
            >
              <ChannelIcon channel={ch.id} />
              <span className="hidden lg:inline">{ch.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Filters */}
      <div className="mb-6 space-y-4">
        <div className="filter-group">
          <label className="filter-label">PERIOD</label>
          <select
            className="filter-select"
            value={filters.period}
            onChange={(e) =>
              onFilterChange({ period: e.target.value as FilterState["period"] })
            }
          >
            <option value="7d">Last 7 Days</option>
            <option value="14d">Last 14 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">FORMAT</label>
          <select
            className="filter-select"
            value={filters.format}
            onChange={(e) =>
              onFilterChange({ format: e.target.value as FilterState["format"] })
            }
          >
            <option value="all">All Formats</option>
            <option value="image">Image</option>
            <option value="video">Video</option>
            <option value="carousel">Carousel</option>
            <option value="story">Story</option>
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">SEGMENT</label>
          <select
            className="filter-select"
            value={filters.segment}
            onChange={(e) =>
              onFilterChange({
                segment: e.target.value as FilterState["segment"],
              })
            }
          >
            <option value="all">All Content</option>
            <option value="organic">Organic</option>
            <option value="paid">Paid</option>
            <option value="ugc">UGC</option>
          </select>
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-[#0f172a] my-2" />

      {/* Mode Tabs */}
      <div className="flex-1">
        <span className="label mb-3 block">AI MODE</span>
        <div className="mode-tabs">
          {MODES.map((mode) => (
            <button
              key={mode.id}
              onClick={() => onModeChange(mode.id)}
              className={`mode-tab ${
                activeMode === mode.id ? "mode-tab-active" : ""
              }`}
            >
              <ModeIcon mode={mode.id} />
              <div>
                <div>{mode.label}</div>
                <div className="text-[9px] text-muted font-normal tracking-normal normal-case">
                  {mode.description}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
