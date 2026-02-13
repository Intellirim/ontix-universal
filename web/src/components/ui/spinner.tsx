import { cn } from "@/lib/utils";

export interface SpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function Spinner({ size = "md", className }: SpinnerProps) {
  return (
    <div
      className={cn(
        "border-2 border-zinc-700 border-t-[var(--sapphire)] rounded-full animate-spin",
        size === "sm" && "w-4 h-4",
        size === "md" && "w-6 h-6",
        size === "lg" && "w-8 h-8",
        className
      )}
    />
  );
}

export function LoadingDots({ className }: { className?: string }) {
  return (
    <div className={cn("flex gap-1", className)}>
      <span className="typing-dot w-1.5 h-1.5 rounded-full bg-zinc-500" />
      <span className="typing-dot w-1.5 h-1.5 rounded-full bg-zinc-500" />
      <span className="typing-dot w-1.5 h-1.5 rounded-full bg-zinc-500" />
    </div>
  );
}
