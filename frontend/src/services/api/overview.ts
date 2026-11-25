import { fetchApiJson } from "@/services/api/client";
import type { OverviewResponse } from "@/types/api";

/** Fetches project specifications, checkpoint SHA-256 verification, dataset counts, and baseline metrics. */
export function getOverview(signal?: AbortSignal): Promise<OverviewResponse> {
  return fetchApiJson<OverviewResponse>("/api/v1/overview", { signal });
}
