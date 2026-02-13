import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "sapphire";
  size?: "sm" | "md" | "lg" | "icon";
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center font-medium transition-all duration-200 disabled:opacity-40 disabled:pointer-events-none",
          // Variants - Zero Tackiness: minimal, translucent
          variant === "default" &&
            "bg-white/[0.04] text-zinc-200 hover:bg-white/[0.06] border border-white/[0.06]",
          variant === "outline" &&
            "border border-white/[0.06] text-zinc-400 hover:bg-white/[0.03] hover:text-zinc-200",
          variant === "ghost" &&
            "text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.03]",
          variant === "sapphire" &&
            "bg-[var(--sapphire)] text-white hover:bg-[var(--sapphire-light)] border border-[var(--sapphire-light)]/20",
          // Sizes
          size === "sm" && "h-7 px-2.5 text-xs rounded-md",
          size === "md" && "h-9 px-4 text-sm rounded-lg",
          size === "lg" && "h-11 px-5 text-sm rounded-lg",
          size === "icon" && "h-9 w-9 rounded-lg",
          className
        )}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
