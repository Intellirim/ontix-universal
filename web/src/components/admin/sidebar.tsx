"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Building2,
  Database,
  Play,
  BarChart3,
  Globe,
  MessageSquare,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  {
    label: "대시보드",
    href: "/brand-dashboard",
    icon: LayoutDashboard,
  },
  {
    label: "브랜드 관리",
    href: "/brand-dashboard/brands",
    icon: Building2,
  },
  {
    label: "데이터 수집",
    href: "/brand-dashboard/pipeline",
    icon: Play,
  },
  {
    label: "Neo4j 데이터",
    href: "/brand-dashboard/neo4j",
    icon: Database,
  },
  {
    label: "소셜 모니터링",
    href: "/brand-dashboard/social",
    icon: Globe,
  },
  {
    label: "대화 분석",
    href: "/brand-dashboard/analytics",
    icon: BarChart3,
  },
  {
    label: "채팅 세션",
    href: "/brand-dashboard/sessions",
    icon: MessageSquare,
  },
];

export function AdminSidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-14 bottom-0 w-64 z-40">
      {/* Glass Background Layer */}
      <div className="absolute inset-0 bg-black/30 backdrop-blur-2xl" />

      {/* Subtle Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#2563EB]/5 via-transparent to-transparent" />

      {/* Border */}
      <div className="absolute right-0 top-0 bottom-0 w-px bg-gradient-to-b from-white/[0.08] via-white/[0.04] to-transparent" />

      {/* Content */}
      <div className="relative h-full overflow-y-auto">
        <nav className="px-4 py-6 space-y-1">
          <span className="block px-3 mb-3 text-[10px] font-medium tracking-widest uppercase text-zinc-600">
            브랜드 메뉴
          </span>

          {navItems.map((item) => {
            const isActive =
              pathname === item.href ||
              (item.href !== "/brand-dashboard" &&
                pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "group relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200",
                  isActive
                    ? "text-white"
                    : "text-zinc-500 hover:text-zinc-200"
                )}
              >
                {/* Active Background */}
                {isActive && (
                  <>
                    <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-[#2563EB]/20 via-[#2563EB]/10 to-transparent" />
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-full bg-[#2563EB] shadow-lg shadow-[#2563EB]/50" />
                  </>
                )}

                {/* Hover Background */}
                {!isActive && (
                  <div className="absolute inset-0 rounded-xl bg-white/[0.03] opacity-0 group-hover:opacity-100 transition-opacity" />
                )}

                {/* Icon */}
                <div className={cn(
                  "relative z-10 w-8 h-8 rounded-lg flex items-center justify-center transition-all",
                  isActive
                    ? "bg-[#2563EB]/20 text-[#2563EB]"
                    : "bg-white/[0.03] text-zinc-500 group-hover:text-zinc-300 group-hover:bg-white/[0.05]"
                )}>
                  <item.icon className="w-4 h-4" />
                </div>

                {/* Label */}
                <span className="relative z-10 flex-1">{item.label}</span>

                {/* Active Indicator */}
                {isActive && (
                  <ChevronRight className="relative z-10 w-4 h-4 text-[#2563EB]" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Bottom Glow */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-32 h-32 bg-[#2563EB]/10 rounded-full blur-3xl pointer-events-none" />
      </div>
    </aside>
  );
}
