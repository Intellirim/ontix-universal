import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", ...props }, ref) => {
    return (
      <input
        type={type}
        ref={ref}
        className={cn(
          "w-full glass-input rounded-lg px-4 py-2.5 text-zinc-100 placeholder-zinc-500",
          "focus:outline-none focus:ring-2 focus:ring-[var(--sapphire)] focus:border-transparent",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          className
        )}
        {...props}
      />
    );
  }
);

Input.displayName = "Input";

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={cn(
          "w-full glass-input rounded-lg px-4 py-2.5 text-zinc-100 placeholder-zinc-500 resize-none",
          "focus:outline-none focus:ring-2 focus:ring-[var(--sapphire)] focus:border-transparent",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          className
        )}
        {...props}
      />
    );
  }
);

Textarea.displayName = "Textarea";
