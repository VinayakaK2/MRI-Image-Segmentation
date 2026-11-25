import { StatusBadge, type StatusTone } from "@/components/common/StatusBadge";
import { EmptyStatePanel } from "@/components/common/StateMessage";
import type { Phase4ConditionRecord, Phase4Payload } from "@/types/api";
import { formatPValue, formatScore, formatSignedDelta, formatSignedPercent } from "@/utils/formatters";

interface RobustnessSummarySectionProps {
  phase4: Phase4Payload | null | undefined;
}

interface FamilyDescriptor {
  familyKey: string;
  title: string;
  interpretation: string;
}

const CORRUPTION_FAMILIES: FamilyDescriptor[] = [
  {
    familyKey: "gaussian_noise",
    title: "Gaussian Noise",
    interpretation:
      "Additive zero-mean noise caused comparatively small degradation on patient-level 3D foreground Dice, with no Holm-significant loss across tested tiers.",
  },
  {
    familyKey: "gaussian_blur",
    title: "Gaussian Blur",
    interpretation:
      "Isotropic spatial blur produced progressive, statistically significant degradation at moderate and severe levels as myocardial boundaries softened.",
  },
  {
    familyKey: "contrast_reduction",
    title: "Contrast Reduction",
    interpretation:
      "Mild and moderate contrast scaling remained stable, while severe contrast attenuation (α=0.25) significantly impaired segmentation accuracy.",
  },
];

function decisionTone(decision: string | undefined): StatusTone {
  if (decision === "Significant") {
    return "error";
  }
  if (decision === "Nominal Only (Unadjusted)") {
    return "warning";
  }
  return "neutral";
}

function getConditionPatientDice(cond: Phase4ConditionRecord | undefined): number | null {
  return cond?.patient_level_summary?.mean_dice_foreground?.mean ?? null;
}

function getConditionSliceDice(cond: Phase4ConditionRecord | undefined): number | null {
  return cond?.slice_level?.mean_dice_foreground ?? null;
}

/** Section E: Phase 4 controlled robustness findings derived from backend JSON artifacts. */
export function RobustnessSummarySection({ phase4 }: RobustnessSummarySectionProps) {
  if (!phase4 || !phase4.available || phase4.conditions.length === 0) {
    return (
      <section aria-labelledby="robustness-summary-heading">
        <h2 id="robustness-summary-heading" className="mb-3 text-[18px] font-semibold text-ink">
          Phase 4 Robustness Findings
        </h2>
        <EmptyStatePanel
          title="Phase 4 robustness artifacts unavailable"
          message={phase4?.error ?? "No verified Phase 4 robustness results were returned by the backend."}
        />
      </section>
    );
  }

  const cleanCond = phase4.conditions.find((c) => c.condition_id === "clean_baseline");
  const cleanPatientDice = getConditionPatientDice(cleanCond);
  const cleanSliceDice = getConditionSliceDice(cleanCond);

  return (
    <section aria-labelledby="robustness-summary-heading">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="robustness-summary-heading" className="text-[18px] font-semibold text-ink">
          Phase 4 Robustness Findings
        </h2>
        <span className="text-xs text-ink-secondary">
          Clean baseline: {formatScore(cleanPatientDice, 4)} (Patient 3D, n={phase4.n_patients ?? 15}) ·{" "}
          {formatScore(cleanSliceDice, 4)} (Slice 2D, n={phase4.total_test_slices ?? 131})
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {CORRUPTION_FAMILIES.map((fam) => {
          const conditions = phase4.conditions.filter((c) => c.family === fam.familyKey);
          const severeCond =
            conditions.find((c) => c.severity === "severe") ?? conditions[conditions.length - 1];
          const severePatientDice = getConditionPatientDice(severeCond);
          const severeSliceDice = getConditionSliceDice(severeCond);
          const comp = severeCond?.statistical_comparison_vs_clean;

          return (
            <article
              key={fam.familyKey}
              className="flex flex-col justify-between rounded-card border border-border-subtle bg-surface p-5"
            >
              <div>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="text-base font-semibold text-ink">{fam.title}</h3>
                    <p className="mt-0.5 text-xs text-ink-secondary">
                      Severities:{" "}
                      {conditions.map((c) => `${c.severity} (${c.param_value})`).join(" · ")}
                    </p>
                  </div>
                  {comp && (
                    <StatusBadge tone={decisionTone(comp.significance_decision)}>
                      {comp.significance_decision}
                    </StatusBadge>
                  )}
                </div>

                <div className="mt-4 grid grid-cols-2 gap-3 rounded-xl border border-border-subtle bg-surface-secondary/50 p-3">
                  <div>
                    <div className="text-[11px] font-medium text-ink-secondary">
                      Severe (Patient 3D)
                    </div>
                    <div className="mt-0.5 font-mono text-lg font-semibold text-ink">
                      {formatScore(severePatientDice, 4)}
                    </div>
                    {comp && (
                      <div className="text-[11px] text-ink-secondary">
                        Δ {formatSignedDelta(comp.mean_difference, 4)} (
                        {formatSignedPercent(comp.mean_relative_change_pct_patient_wise, 2)})
                      </div>
                    )}
                  </div>
                  <div>
                    <div className="text-[11px] font-medium text-ink-secondary">
                      Severe (Slice 2D)
                    </div>
                    <div className="mt-0.5 font-mono text-lg font-semibold text-ink">
                      {formatScore(severeSliceDice, 4)}
                    </div>
                    {comp && (
                      <div className="text-[11px] text-ink-secondary">
                        Holm p = {formatPValue(comp.paired_t_pvalue_holm)}
                      </div>
                    )}
                  </div>
                </div>

                <p className="mt-3 text-xs leading-relaxed text-ink-secondary">
                  {fam.interpretation}
                </p>
              </div>

              <div className="mt-4 border-t border-border-subtle pt-2.5 text-[11px] text-ink-secondary">
                Clean ref: {formatScore(cleanPatientDice, 4)} Patient 3D /{" "}
                {formatScore(cleanSliceDice, 4)} Slice 2D
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
