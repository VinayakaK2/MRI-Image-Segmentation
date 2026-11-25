import { AlertTriangle, Inbox, RefreshCw } from "lucide-react";

interface ErrorStatePanelProps {
  title: string;
  message: string;
  code?: string;
  onRetry?: () => void;
  isRetrying?: boolean;
}

/** Explicit error state displaying failure reason and retry action without fake fallback values. */
export function ErrorStatePanel({
  title,
  message,
  code,
  onRetry,
  isRetrying = false,
}: ErrorStatePanelProps) {
  return (
    <div
      role="alert"
      className="rounded-card border border-border-subtle bg-surface p-6"
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-lg bg-[var(--status-error-bg)] p-2 text-status-error">
            <AlertTriangle className="h-5 w-5" aria-hidden="true" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-ink">{title}</h3>
            <p className="mt-1 text-sm text-ink-secondary">{message}</p>
            {code && (
              <p className="mt-1.5 font-mono text-xs text-ink-secondary">
                Code: {code}
              </p>
            )}
          </div>
        </div>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            disabled={isRetrying}
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg border border-border-subtle bg-surface-secondary px-3.5 py-2 text-sm font-medium text-ink transition-colors hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw className={`h-4 w-4 ${isRetrying ? "animate-spin" : ""}`} aria-hidden="true" />
            {isRetrying ? "Retrying…" : "Retry"}
          </button>
        )}
      </div>
    </div>
  );
}

interface EmptyStatePanelProps {
  title: string;
  message: string;
}

/** Clean informative empty state when backend reports missing or unavailable artifacts. */
export function EmptyStatePanel({ title, message }: EmptyStatePanelProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-card border border-dashed border-border-subtle bg-surface p-8 text-center">
      <div className="rounded-xl bg-surface-secondary p-3 text-ink-secondary">
        <Inbox className="h-5 w-5" aria-hidden="true" />
      </div>
      <h3 className="mt-3 text-sm font-semibold text-ink">{title}</h3>
      <p className="mt-1 max-w-md text-xs leading-relaxed text-ink-secondary">{message}</p>
    </div>
  );
}
