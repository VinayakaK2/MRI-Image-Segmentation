import { useQuery } from "@tanstack/react-query";
import { ApiClientError } from "@/services/api/client";
import { getExperiments, type ExperimentsQueryParams } from "@/services/api/experiments";
import type { ExperimentsResponse } from "@/types/api";

/** Queries GET /api/v1/experiments for Phase 4 and Phase 5 canonical artifacts. */
export function useExperiments(params: ExperimentsQueryParams = {}) {
  return useQuery<ExperimentsResponse>({
    queryKey: ["api", "experiments", params] as const,
    queryFn: ({ signal }) => getExperiments(params, signal),
    retry: (failureCount, error) =>
      !(error instanceof ApiClientError && error.kind === "network_unreachable") && failureCount < 1,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });
}
