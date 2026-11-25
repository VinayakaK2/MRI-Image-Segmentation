import { useQuery } from "@tanstack/react-query";
import { getHealth } from "@/services/api/health";
import type { HealthResponse } from "@/types/api";

export const HEALTH_QUERY_KEY = ["api", "health"] as const;

/** Queries GET /api/v1/health with fast failure reporting when backend is offline. */
export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: HEALTH_QUERY_KEY,
    queryFn: ({ signal }) => getHealth(signal),
    retry: false,
    refetchInterval: 30_000,
    refetchOnWindowFocus: false,
  });
}
