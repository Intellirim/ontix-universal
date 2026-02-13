import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function generateId(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
}

export function getGradeClass(grade: string): string {
  const gradeMap: Record<string, string> = {
    A: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
    B: "bg-blue-500/20 text-blue-400 border border-blue-500/30",
    C: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
    D: "bg-orange-500/20 text-orange-400 border border-orange-500/30",
    F: "bg-red-500/20 text-red-400 border border-red-500/30",
  }
  return gradeMap[grade] || "bg-white/10 text-white/60 border border-white/20"
}

export function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M"
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + "K"
  }
  return num.toString()
}

export function formatMs(ms: number): string {
  if (ms >= 1000) {
    return (ms / 1000).toFixed(2) + "s"
  }
  return ms.toFixed(0) + "ms"
}

export function formatDateTime(date: string | Date): string {
  const d = typeof date === "string" ? new Date(date) : date
  return d.toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function formatDate(date: string | Date): string {
  const d = typeof date === "string" ? new Date(date) : date
  return d.toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  })
}

export function formatTime(date: string | Date): string {
  const d = typeof date === "string" ? new Date(date) : date
  return d.toLocaleTimeString("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function formatRelativeTime(date: string | Date): string {
  const d = typeof date === "string" ? new Date(date) : date
  const now = new Date()
  const diff = now.getTime() - d.getTime()

  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)

  if (days > 0) return `${days}일 전`
  if (hours > 0) return `${hours}시간 전`
  if (minutes > 0) return `${minutes}분 전`
  return "방금 전"
}
