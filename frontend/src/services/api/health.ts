import { fetchApiJson } from "@/services/api/client";
import type { HealthResponse } from "@/types/api";

/** Fetches live backend, PyTorch, model initialization, and research artifact readiness. */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return fetchApiJson<HealthResponse>("/api/v1/health", { signal });
}
