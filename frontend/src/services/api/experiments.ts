import { fetchApiJson } from "@/services/api/client";
import type { ExperimentsResponse } from "@/types/api";

export interface ExperimentsQueryParams {
  phase?: 4 | 5;
  family?: string;
  patientId?: string;
}

/** Fetches Phase 4 robustness and Phase 5 mitigation experiment results from backend artifacts. */
export function getExperiments(
  params: ExperimentsQueryParams = {},
  signal?: AbortSignal
): Promise<ExperimentsResponse> {
  const search = new URLSearchParams();
  if (params.phase !== undefined) {
    search.set("phase", String(params.phase));
  }
  if (params.family) {
    search.set("family", params.family);
  }
  if (params.patientId) {
    search.set("patient_id", params.patientId);
  }
  const qs = search.toString();
  const path = qs ? `/api/v1/experiments?${qs}` : "/api/v1/experiments";
  return fetchApiJson<ExperimentsResponse>(path, { signal });
}
