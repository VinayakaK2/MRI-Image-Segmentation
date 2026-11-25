import React from "react";

export type StatusTone = "success" | "warning" | "error" | "neutral" | "accent";

interface StatusBadgeProps {
  tone: StatusTone;
  children: React.ReactNode;
  dot?: boolean;
  title?: string;
}

const TONE_CLASSES: Record<StatusTone, string> = {
  success: "border-transparent bg-[var(--status-success-bg)] text-status-success",
  warning: "border-transparent bg-[var(--status-warning-bg)] text-status-warning",
  error: "border-transparent bg-[var(--status-error-bg)] text-status-error",
  accent: "border-transparent bg-accent-subtle text-accent",
  neutral: "border-border-subtle bg-surface-secondary text-ink-secondary",
};

const DOT_CLASSES: Record<StatusTone, string> = {
  success: "bg-status-success",
  warning: "bg-status-warning",
  error: "bg-status-error",
  accent: "bg-accent",
  neutral: "bg-ink-secondary",
};

/** Compact semantic status pill with accessible contrast in light and dark themes. */
export function StatusBadge({ tone, children, dot = false, title }: StatusBadgeProps) {
  return (
    <span
      title={title}
      className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-0.5 text-xs font-medium ${TONE_CLASSES[tone]}`}
    >
      {dot && <span className={`h-1.5 w-1.5 rounded-full ${DOT_CLASSES[tone]}`} aria-hidden="true" />}
      {children}
    </span>
  );
}
