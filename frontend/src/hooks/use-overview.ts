import { useQuery } from "@tanstack/react-query";
import { ApiClientError } from "@/services/api/client";
import { getOverview } from "@/services/api/overview";
import type { OverviewResponse } from "@/types/api";

export const OVERVIEW_QUERY_KEY = ["api", "overview"] as const;

/** Queries GET /api/v1/overview once and caches verified project specifications. */
export function useOverview() {
  return useQuery<OverviewResponse>({
    queryKey: OVERVIEW_QUERY_KEY,
    queryFn: ({ signal }) => getOverview(signal),
    retry: (failureCount, error) =>
      !(error instanceof ApiClientError && error.kind === "network_unreachable") && failureCount < 1,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });
}
