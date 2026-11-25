import type { AggregatedMetricSummary, CheckpointOverview } from "@/types/api";

/** Formats a floating-point score (e.g. Dice or IoU) with fixed decimals. */
export function formatScore(value: number | null | undefined, decimals = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "Not available";
  }
  return value.toFixed(decimals);
}

/** Formats a signed metric difference (e.g. +0.0738 or -0.1182). */
export function formatSignedDelta(value: number | null | undefined, decimals = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}`;
}

/** Formats a signed percentage change (e.g. -13.15%). */
export function formatSignedPercent(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}%`;
}

/** Formats a p-value in scientific notation for small probabilities or 4 decimals otherwise. */
export function formatPValue(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "N/A";
  }
  if (value < 0.001) {
    return value.toExponential(2);
  }
  return value.toFixed(4);
}

/** Formats an integer parameter or sample count with locale grouping. */
export function formatInteger(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "Not available";
  }
  return Math.trunc(value).toLocaleString("en-US");
}

/** Shortens a 64-char SHA-256 hex digest for compact display while preserving full value in titles. */
export function shortenSha256(hash: string | null | undefined, prefixLen = 12): string {
  if (!hash) {
    return "Unavailable";
  }
  if (hash.length <= prefixLen) {
    return hash;
  }
  return `${hash.slice(0, prefixLen)}…`;
}

/** Finds a clean baseline metric record matching a specific aggregation level. */
export function findBaselineMetricByAggregation(
  metrics: AggregatedMetricSummary[] | undefined,
  aggregationLevel: "patient_level_3d_pooled" | "slice_level"
): AggregatedMetricSummary | null {
  if (!metrics || metrics.length === 0) {
    return null;
  }
  return (
    metrics.find(
      (item) =>
        item.condition_id === "clean_baseline" &&
        item.aggregation_level === aggregationLevel
    ) ?? null
  );
}

/** Returns the canonical checkpoint entry from the overview checkpoint list. */
export function findCanonicalCheckpoint(
  checkpoints: CheckpointOverview[] | undefined
): CheckpointOverview | null {
  if (!checkpoints || checkpoints.length === 0) {
    return null;
  }
  return checkpoints.find((c) => c.is_canonical) ?? null;
}
