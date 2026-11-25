import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { EmptyStatePanel } from "@/components/common/StateMessage";
import type { Phase4ConditionRecord, Phase4Payload } from "@/types/api";
import { formatPValue, formatScore, formatSignedDelta } from "@/utils/formatters";

interface PerformanceComparisonChartProps {
  phase4: Phase4Payload | null | undefined;
}

type AggregationMode = "patient_level_3d_pooled" | "slice_level";
type MetricMode = "dice" | "iou";

interface ChartDatum {
  conditionId: string;
  shortLabel: string;
  displayName: string;
  family: string;
  severity: string;
  paramValue: number;
  units: string;
  value: number;
  std: number | null;
  deltaVsClean: number | null;
  holmPValue: number | null;
  significance: string | null;
  fillVar: string;
}

const SHORT_LABELS: Record<string, string> = {
  clean_baseline: "Clean",
  gaussian_noise_mild: "Noise\nMild",
  gaussian_noise_moderate: "Noise\nMod",
  gaussian_noise_severe: "Noise\nSev",
  gaussian_blur_mild: "Blur\nMild",
  gaussian_blur_moderate: "Blur\nMod",
  gaussian_blur_severe: "Blur\nSev",
  contrast_reduction_mild: "Contrast\nMild",
  contrast_reduction_moderate: "Contrast\nMod",
  contrast_reduction_severe: "Contrast\nSev",
};

function getFamilyColorVar(family: string): string {
  switch (family) {
    case "clean":
      return "var(--chart-baseline)";
    case "gaussian_noise":
      return "var(--chart-noise)";
    case "gaussian_blur":
      return "var(--chart-blur)";
    case "contrast_reduction":
      return "var(--chart-contrast)";
    default:
      return "var(--accent)";
  }
}

function extractDatum(
  cond: Phase4ConditionRecord,
  aggregation: AggregationMode,
  metric: MetricMode
): ChartDatum | null {
  const metricKey = metric === "dice" ? "mean_dice_foreground" : "mean_iou_foreground";
  let value: number | undefined;
  let std: number | null = null;

  if (aggregation === "patient_level_3d_pooled") {
    const summary = cond.patient_level_summary?.[metricKey];
    value = summary?.mean;
    std = summary?.std ?? null;
  } else {
    value = cond.slice_level?.[metricKey];
  }

  if (value === undefined || !Number.isFinite(value)) {
    return null;
  }

  const comp = cond.statistical_comparison_vs_clean;
  return {
    conditionId: cond.condition_id,
    shortLabel: SHORT_LABELS[cond.condition_id] ?? cond.condition_id,
    displayName: cond.display_name,
    family: cond.family,
    severity: cond.severity,
    paramValue: cond.param_value,
    units: cond.units,
    value,
    std,
    deltaVsClean:
      aggregation === "patient_level_3d_pooled" && metric === "dice" && comp
        ? comp.mean_difference
        : null,
    holmPValue: comp ? comp.paired_t_pvalue_holm : null,
    significance: comp ? comp.significance_decision : null,
    fillVar: getFamilyColorVar(cond.family),
  };
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: ChartDatum }>;
  aggregation: AggregationMode;
  metric: MetricMode;
}

function ChartTooltipContent({ active, payload, aggregation, metric }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const d = payload[0].payload;
  const metricLabel = metric === "dice" ? "Foreground Dice" : "Foreground IoU";
  const aggLabel =
    aggregation === "patient_level_3d_pooled" ? "Patient 3D Pooled (n=15)" : "Slice 2D Avg (n=131)";

  return (
    <div
      className="rounded-xl border p-3 text-xs shadow-lg"
      style={{
        backgroundColor: "var(--chart-tooltip-bg)",
        borderColor: "var(--chart-tooltip-border)",
        color: "var(--text-primary)",
      }}
    >
      <div className="font-semibold">{d.displayName}</div>
      <div className="mt-0.5 text-[11px] text-ink-secondary">{aggLabel}</div>
      <div className="mt-2 space-y-1">
        <div className="flex justify-between gap-4">
          <span className="text-ink-secondary">{metricLabel}:</span>
          <span className="font-mono font-semibold">
            {formatScore(d.value, 4)}
            {d.std !== null ? ` ± ${d.std.toFixed(4)}` : ""}
          </span>
        </div>
        {d.deltaVsClean !== null && (
          <div className="flex justify-between gap-4">
            <span className="text-ink-secondary">Paired Δ vs Clean:</span>
            <span className="font-mono">{formatSignedDelta(d.deltaVsClean, 4)}</span>
          </div>
        )}
        {d.holmPValue !== null && (
          <div className="flex justify-between gap-4">
            <span className="text-ink-secondary">Holm p (m=9):</span>
            <span className="font-mono">
              {formatPValue(d.holmPValue)} ({d.significance})
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/** Section G: Interactive Recharts Phase 4 comparison separating patient-level and slice-level series. */
export function PerformanceComparisonChart({ phase4 }: PerformanceComparisonChartProps) {
  const [aggregation, setAggregation] = useState<AggregationMode>("patient_level_3d_pooled");
  const [metric, setMetric] = useState<MetricMode>("dice");

  const chartData = useMemo(() => {
    if (!phase4 || !phase4.available) {
      return [];
    }
    return phase4.conditions
      .map((cond) => extractDatum(cond, aggregation, metric))
      .filter((item): item is ChartDatum => item !== null);
  }, [phase4, aggregation, metric]);

  const cleanValue = chartData.find((d) => d.conditionId === "clean_baseline")?.value;

  if (!phase4 || !phase4.available || chartData.length === 0) {
    return (
      <section aria-labelledby="perf-chart-heading">
        <h2 id="perf-chart-heading" className="mb-3 text-[18px] font-semibold text-ink">
          Performance Comparison Across Phase 4 Conditions
        </h2>
        <EmptyStatePanel
          title="Performance comparison chart unavailable"
          message={phase4?.error ?? "Phase 4 condition metrics were not returned by the backend."}
        />
      </section>
    );
  }

  return (
    <section
      aria-labelledby="perf-chart-heading"
      className="rounded-card border border-border-subtle bg-surface p-5"
    >
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 id="perf-chart-heading" className="text-[18px] font-semibold text-ink">
            Performance Comparison Across Phase 4 Conditions
          </h2>
          <p className="mt-0.5 text-xs text-ink-secondary">
            {metric === "dice" ? "Macro Foreground Dice (RV, MYO, LV)" : "Macro Foreground IoU (RV, MYO, LV)"} ·{" "}
            {aggregation === "patient_level_3d_pooled"
              ? `Patient-level 3D volume pooled (n=${phase4.n_patients ?? 15})`
              : `Slice-level 2D arithmetic mean (n=${phase4.total_test_slices ?? 131})`}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div
            role="group"
            aria-label="Aggregation level selector"
            className="inline-flex rounded-xl border border-border-subtle bg-surface-secondary p-0.5 text-xs"
          >
            <button
              type="button"
              aria-pressed={aggregation === "patient_level_3d_pooled"}
              onClick={() => setAggregation("patient_level_3d_pooled")}
              className={`rounded-[10px] px-2.5 py-1.5 font-medium transition-colors ${
                aggregation === "patient_level_3d_pooled"
                  ? "bg-surface text-ink shadow-sm"
                  : "text-ink-secondary hover:text-ink"
              }`}
            >
              Patient 3D (n=15)
            </button>
            <button
              type="button"
              aria-pressed={aggregation === "slice_level"}
              onClick={() => setAggregation("slice_level")}
              className={`rounded-[10px] px-2.5 py-1.5 font-medium transition-colors ${
                aggregation === "slice_level"
                  ? "bg-surface text-ink shadow-sm"
                  : "text-ink-secondary hover:text-ink"
              }`}
            >
              Slice 2D (n=131)
            </button>
          </div>

          <div
            role="group"
            aria-label="Metric selector"
            className="inline-flex rounded-xl border border-border-subtle bg-surface-secondary p-0.5 text-xs"
          >
            <button
              type="button"
              aria-pressed={metric === "dice"}
              onClick={() => setMetric("dice")}
              className={`rounded-[10px] px-2.5 py-1.5 font-medium transition-colors ${
                metric === "dice"
                  ? "bg-surface text-ink shadow-sm"
                  : "text-ink-secondary hover:text-ink"
              }`}
            >
              Dice
            </button>
            <button
              type="button"
              aria-pressed={metric === "iou"}
              onClick={() => setMetric("iou")}
              className={`rounded-[10px] px-2.5 py-1.5 font-medium transition-colors ${
                metric === "iou"
                  ? "bg-surface text-ink shadow-sm"
                  : "text-ink-secondary hover:text-ink"
              }`}
            >
              IoU
            </button>
          </div>
        </div>
      </div>

      {/* Explicit Color Legend */}
      <div className="mb-3 flex flex-wrap items-center gap-4 text-xs text-ink-secondary">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "var(--chart-baseline)" }} />
          Clean Baseline
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "var(--chart-noise)" }} />
          Gaussian Noise (0.02 / 0.05 / 0.10)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "var(--chart-blur)" }} />
          Gaussian Blur (0.75 / 1.50 / 2.50 px)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "var(--chart-contrast)" }} />
          Contrast Reduction (0.75 / 0.50 / 0.25)
        </span>
      </div>

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 12, right: 16, left: 0, bottom: 20 }}>
            <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="shortLabel"
              tick={{ fill: "var(--chart-axis)", fontSize: 11 }}
              interval={0}
              height={42}
              tickLine={false}
              axisLine={{ stroke: "var(--chart-grid)" }}
            />
            <YAxis
              domain={[0.5, 1.0]}
              tick={{ fill: "var(--chart-axis)", fontSize: 11 }}
              tickFormatter={(v: number) => v.toFixed(2)}
              tickLine={false}
              axisLine={{ stroke: "var(--chart-grid)" }}
              width={44}
            />
            <Tooltip
              cursor={{ fill: "var(--accent-subtle)" }}
              content={<ChartTooltipContent aggregation={aggregation} metric={metric} />}
            />
            {cleanValue !== undefined && (
              <ReferenceLine
                y={cleanValue}
                stroke="var(--chart-baseline)"
                strokeDasharray="4 4"
                strokeWidth={1.5}
              />
            )}
            <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={44}>
              {chartData.map((entry) => (
                <Cell key={entry.conditionId} fill={entry.fillVar} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
