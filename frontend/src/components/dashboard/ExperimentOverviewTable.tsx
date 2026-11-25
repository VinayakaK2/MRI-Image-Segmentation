import { useState } from "react";
import { StatusBadge, type StatusTone } from "@/components/common/StatusBadge";
import { EmptyStatePanel } from "@/components/common/StateMessage";
import type { ExperimentsResponse } from "@/types/api";
import { formatPValue, formatScore, formatSignedDelta } from "@/utils/formatters";

interface ExperimentOverviewTableProps {
  experiments: ExperimentsResponse | undefined;
}

type ExperimentTab = "phase4" | "phase5";

function getSigTone(decision: string): StatusTone {
  if (decision === "Significant") {
    return "accent";
  }
  if (decision === "Nominal Only (Unadjusted)") {
    return "warning";
  }
  return "neutral";
}

/** Section H: Factual Research Experiment Overview backed strictly by GET /api/v1/experiments. */
export function ExperimentOverviewTable({ experiments }: ExperimentOverviewTableProps) {
  const [activeTab, setActiveTab] = useState<ExperimentTab>("phase4");

  const phase4 = experiments?.phase4;
  const phase5 = experiments?.phase5;

  const phase4Count = phase4?.conditions.length ?? 0;
  const phase5Rows = phase5?.families.flatMap((f) => f.hypotheses) ?? [];

  return (
    <section
      aria-labelledby="experiment-overview-heading"
      className="rounded-card border border-border-subtle bg-surface p-5"
    >
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 id="experiment-overview-heading" className="text-[18px] font-semibold text-ink">
            Research Experiment Overview
          </h2>
          <p className="mt-0.5 text-xs text-ink-secondary">
            Verified Phase 4 robustness conditions and Phase 5 pre-registered hypotheses from canonical artifacts
          </p>
        </div>

        <div
          role="tablist"
          aria-label="Experiment phase selector"
          className="inline-flex rounded-xl border border-border-subtle bg-surface-secondary p-0.5 text-xs"
        >
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "phase4"}
            onClick={() => setActiveTab("phase4")}
            className={`whitespace-nowrap rounded-[10px] px-3 py-1.5 font-medium transition-colors ${
              activeTab === "phase4"
                ? "bg-surface text-ink shadow-sm"
                : "text-ink-secondary hover:text-ink"
            }`}
          >
            Phase 4 · Robustness ({phase4Count})
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "phase5"}
            onClick={() => setActiveTab("phase5")}
            className={`whitespace-nowrap rounded-[10px] px-3 py-1.5 font-medium transition-colors ${
              activeTab === "phase5"
                ? "bg-surface text-ink shadow-sm"
                : "text-ink-secondary hover:text-ink"
            }`}
          >
            Phase 5 · Hypotheses ({phase5Rows.length})
          </button>
        </div>
      </div>

      {activeTab === "phase4" && (
        <>
          {!phase4 || !phase4.available || phase4.conditions.length === 0 ? (
            <EmptyStatePanel
              title="No Phase 4 condition records"
              message={phase4?.error ?? "Phase 4 robustness artifacts were not returned."}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-border-subtle text-ink-secondary">
                    <th className="py-2.5 pr-4 font-semibold">Condition</th>
                    <th className="py-2.5 pr-4 font-semibold">Family / Severity</th>
                    <th className="py-2.5 pr-4 font-semibold">Patient 3D Dice (n=15)</th>
                    <th className="py-2.5 pr-4 font-semibold">Slice 2D Dice (n=131)</th>
                    <th className="py-2.5 pr-4 font-semibold">Paired Δ (3D)</th>
                    <th className="py-2.5 pr-4 font-semibold">Holm p (m=9)</th>
                    <th className="py-2.5 font-semibold">Statistical Result</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle">
                  {phase4.conditions.map((cond) => {
                    const p3d = cond.patient_level_summary?.mean_dice_foreground;
                    const s2d = cond.slice_level?.mean_dice_foreground;
                    const comp = cond.statistical_comparison_vs_clean;
                    return (
                      <tr key={cond.condition_id} className="hover:bg-surface-secondary/40">
                        <td className="py-2.5 pr-4 font-medium text-ink">{cond.display_name}</td>
                        <td className="py-2.5 pr-4 text-ink-secondary">
                          {cond.family} · {cond.severity}
                        </td>
                        <td className="py-2.5 pr-4 font-mono text-ink">
                          {p3d ? `${formatScore(p3d.mean, 4)} ± ${p3d.std.toFixed(4)}` : "—"}
                        </td>
                        <td className="py-2.5 pr-4 font-mono text-ink">{formatScore(s2d, 4)}</td>
                        <td className="py-2.5 pr-4 font-mono text-ink">
                          {comp ? formatSignedDelta(comp.mean_difference, 4) : "Reference"}
                        </td>
                        <td className="py-2.5 pr-4 font-mono text-ink">
                          {comp ? formatPValue(comp.paired_t_pvalue_holm) : "N/A"}
                        </td>
                        <td className="py-2.5">
                          <StatusBadge
                            tone={comp ? getSigTone(comp.significance_decision) : "neutral"}
                          >
                            {comp ? comp.significance_decision : "Clean Benchmark"}
                          </StatusBadge>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {activeTab === "phase5" && (
        <>
          {!phase5 || !phase5.available || phase5Rows.length === 0 ? (
            <EmptyStatePanel
              title="No Phase 5 hypothesis records"
              message={phase5?.error ?? "Phase 5 mitigation artifacts were not returned."}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[780px] border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-border-subtle text-ink-secondary">
                    <th className="py-2.5 pr-4 font-semibold">Holm Family</th>
                    <th className="py-2.5 pr-4 font-semibold">Condition</th>
                    <th className="py-2.5 pr-4 font-semibold">Patient 3D Dice (n=15)</th>
                    <th className="py-2.5 pr-4 font-semibold">Slice 2D Dice (n=131)</th>
                    <th className="py-2.5 pr-4 font-semibold">Paired Δ</th>
                    <th className="py-2.5 pr-4 font-semibold">Holm p (Family-wise)</th>
                    <th className="py-2.5 font-semibold">Decision</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle">
                  {phase5Rows.map((hyp) => (
                    <tr
                      key={`${hyp.family_id}-${hyp.condition_id}`}
                      className="hover:bg-surface-secondary/40"
                    >
                      <td className="py-2.5 pr-4 font-medium text-ink">{hyp.family_name}</td>
                      <td className="py-2.5 pr-4 text-ink-secondary">{hyp.condition_id}</td>
                      <td className="py-2.5 pr-4 font-mono text-ink">
                        {formatScore(hyp.patient_dice_fg_mean, 4)} ±{" "}
                        {hyp.patient_dice_fg_std.toFixed(4)}
                      </td>
                      <td className="py-2.5 pr-4 font-mono text-ink">
                        {formatScore(hyp.slice_dice_fg, 4)}
                      </td>
                      <td className="py-2.5 pr-4 font-mono text-ink">
                        {formatSignedDelta(hyp.mean_difference, 4)}
                      </td>
                      <td className="py-2.5 pr-4 font-mono text-ink">
                        {formatPValue(hyp.paired_t_pvalue_holm)}
                      </td>
                      <td className="py-2.5">
                        <StatusBadge tone={getSigTone(hyp.significance_decision)}>
                          {hyp.significance_decision}
                        </StatusBadge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}
