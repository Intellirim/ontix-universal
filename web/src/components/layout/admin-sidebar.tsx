"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Bell,
  BarChart3,
  MessageSquare,
  Settings,
  LogOut,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/dashboard", label: "대시보드", icon: LayoutDashboard },
  { href: "/alerts", label: "알림", icon: Bell },
  { href: "/analytics", label: "분석", icon: BarChart3 },
  { href: "/chat", label: "챗 테스트", icon: MessageSquare },
];

export function AdminSidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-64 z-30">
      {/* Glass Background Layer */}
      <div className="absolute inset-0 bg-black/30 backdrop-blur-2xl" />

      {/* Subtle Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#2563EB]/5 via-transparent to-transparent" />

      {/* Border */}
      <div className="absolute right-0 top-0 bottom-0 w-px bg-gradient-to-b from-white/[0.08] via-white/[0.04] to-transparent" />

      {/* Content */}
      <div className="relative flex flex-col h-full">
        {/* Logo Header */}
        <div className="px-6 py-6">
          <Link href="/dashboard" className="flex items-center gap-3 group">
            {/* Logo Icon */}
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#2563EB] to-[#1D4ED8] flex items-center justify-center shadow-lg shadow-[#2563EB]/20">
                <span className="text-white font-bold text-sm">O</span>
              </div>
              {/* Glow Effect */}
              <div className="absolute inset-0 w-10 h-10 rounded-xl bg-[#2563EB]/30 blur-xl -z-10" />
            </div>

            {/* Logo Text */}
            <div>
              <h1 className="text-lg font-semibold tracking-tight text-zinc-100 group-hover:text-white transition-colors">
                ONTIX
              </h1>
              <span className="text-[10px] font-medium tracking-widest uppercase text-zinc-500">
                Admin Panel
              </span>
            </div>
          </Link>
        </div>

        {/* Divider */}
        <div className="mx-4 h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-1">
          <span className="block px-3 mb-3 text-[10px] font-medium tracking-widest uppercase text-zinc-600">
            메뉴
          </span>

          {navItems.map((item) => {
            const isActive = pathname === item.href;
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

        {/* Divider */}
        <div className="mx-4 h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />

        {/* Footer */}
        <div className="px-4 py-6 space-y-1">
          <span className="block px-3 mb-3 text-[10px] font-medium tracking-widest uppercase text-zinc-600">
            설정
          </span>

          <Link
            href="/settings"
            className="group relative flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium text-zinc-500 hover:text-zinc-200 transition-all"
          >
            <div className="absolute inset-0 rounded-xl bg-white/[0.03] opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="relative z-10 w-8 h-8 rounded-lg bg-white/[0.03] flex items-center justify-center group-hover:bg-white/[0.05] transition-all">
              <Settings className="w-4 h-4" />
            </div>
            <span className="relative z-10">설정</span>
          </Link>

          <button className="group relative w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium text-zinc-500 hover:text-red-400 transition-all">
            <div className="absolute inset-0 rounded-xl bg-white/[0.03] opacity-0 group-hover:opacity-100 group-hover:bg-red-500/5 transition-all" />
            <div className="relative z-10 w-8 h-8 rounded-lg bg-white/[0.03] flex items-center justify-center group-hover:bg-red-500/10 transition-all">
              <LogOut className="w-4 h-4" />
            </div>
            <span className="relative z-10">로그아웃</span>
          </button>
        </div>

        {/* Bottom Glow */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-32 h-32 bg-[#2563EB]/10 rounded-full blur-3xl pointer-events-none" />
      </div>
    </aside>
  );
}
