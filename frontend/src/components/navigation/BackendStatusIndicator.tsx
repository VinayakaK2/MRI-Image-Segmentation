import { RefreshCw } from "lucide-react";
import { StatusBadge, type StatusTone } from "@/components/common/StatusBadge";
import { ApiClientError } from "@/services/api/client";
import type { HealthResponse } from "@/types/api";

export type BackendConnectionState = "Initializing" | "Connected" | "Disconnected" | "Error";

interface BackendStatusIndicatorProps {
  health: HealthResponse | undefined;
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  onRefresh: () => void;
  isFetching: boolean;
}

/** Derives explicit backend connectivity state strictly from GET /api/v1/health response. */
export function resolveBackendConnectionState(
  health: HealthResponse | undefined,
  isLoading: boolean,
  isError: boolean,
  error: unknown
): { state: BackendConnectionState; tone: StatusTone; detail: string } {
  if (isLoading && !health) {
    return {
      state: "Initializing",
      tone: "warning",
      detail: "Checking local FastAPI backend health…",
    };
  }

  if (isError || !health) {
    const isNetworkError =
      error instanceof ApiClientError && error.kind === "network_unreachable";
    if (isNetworkError) {
      return {
        state: "Disconnected",
        tone: "error",
        detail: "Local FastAPI backend is unreachable at http://127.0.0.1:8000.",
      };
    }
    return {
      state: "Error",
      tone: "error",
      detail: error instanceof Error ? error.message : "Backend health check failed.",
    };
  }

  if (health.status === "ok" && health.model_initialized) {
    return {
      state: "Connected",
      tone: "success",
      detail: `FastAPI ${health.backend_version} · PyTorch ${health.pytorch_version ?? "N/A"} · Device ${health.device_active.toUpperCase()}`,
    };
  }

  return {
    state: "Error",
    tone: "warning",
    detail:
      health.model_load_error ??
      `Backend status is '${health.status}' (model_initialized=${String(health.model_initialized)}).`,
  };
}

/** Header indicator displaying live FastAPI health state and manual refresh button. */
export function BackendStatusIndicator({
  health,
  isLoading,
  isError,
  error,
  onRefresh,
  isFetching,
}: BackendStatusIndicatorProps) {
  const resolved = resolveBackendConnectionState(health, isLoading, isError, error);

  return (
    <div className="flex items-center gap-1.5">
      <StatusBadge tone={resolved.tone} dot title={resolved.detail}>
        <span className="hidden md:inline">Backend:</span>
        <span>{resolved.state}</span>
      </StatusBadge>
      <button
        type="button"
        onClick={onRefresh}
        aria-label="Refresh backend health status"
        title={resolved.detail}
        className="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-border-subtle bg-surface text-ink-secondary transition-colors hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <RefreshCw className={`h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`} aria-hidden="true" />
      </button>
    </div>
  );
}
