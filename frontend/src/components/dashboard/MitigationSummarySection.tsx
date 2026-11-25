import { AlertCircle } from "lucide-react";
import { StatusBadge, type StatusTone } from "@/components/common/StatusBadge";
import { EmptyStatePanel } from "@/components/common/StateMessage";
import type { Phase5FamilyGroup, Phase5Payload } from "@/types/api";
import { formatPValue, formatScore, formatSignedDelta } from "@/utils/formatters";

interface MitigationSummarySectionProps {
  phase5: Phase5Payload | null | undefined;
}

interface FamilyInterpretation {
  verdict: string;
  tone: StatusTone;
  summary: string;
  limitation: string;
}

function interpretFamily(family: Phase5FamilyGroup): FamilyInterpretation {
  const sigItems = family.hypotheses.filter(
    (h) => h.significance_decision === "Significant"
  );
  const nominalItems = family.hypotheses.filter(
    (h) => h.significance_decision === "Nominal Only (Unadjusted)"
  );

  if (family.family_id === "family_1_postproc_alone") {
    return {
      verdict: sigItems.length > 0 ? "Supported" : "Inconclusive",
      tone: "warning",
      summary:
        nominalItems.length > 0
          ? `0 of ${family.hypothesis_count_m} tests reached family-wise Holm significance; ${nominalItems.length} reached nominal unadjusted Wilcoxon significance (${nominalItems[0].condition_id}).`
          : `No statistically significant improvement across ${family.hypothesis_count_m} conditions after family-wise Holm correction.`,
      limitation:
        "Modest positive deltas (+0.0014 to +0.0140 Dice) do not establish reliable recovery across conditions.",
    };
  }

  if (
    family.family_id === "family_2_clahe_alone" ||
    family.family_id === "family_3_clahe_plus_postproc"
  ) {
    return {
      verdict: "Confounded (Invariance)",
      tone: "warning",
      summary:
        sigItems.length > 0
          ? `${sigItems.length} of ${family.hypothesis_count_m} severe-contrast comparisons passed Holm correction (p_holm=${formatPValue(sigItems[0].paired_t_pvalue_holm)}), but forensic audit shows recovery stems from min-max re-normalization.`
          : `No significant difference under mild/moderate contrast; severe contrast recovery is attributable to input re-normalization.`,
      limitation:
        "to_unit_interval() is mathematically affine-invariant to positive contrast scaling; CLAHE itself added a net -0.0062 Dice penalty relative to pure z-score.",
    };
  }

  return {
    verdict: "Negative Result",
    tone: "neutral",
    summary: `0 of ${family.hypothesis_count_m} noise conditions improved (mean Δ from ${formatSignedDelta(
      Math.min(...family.hypotheses.map((h) => h.mean_difference))
    )} to ${formatSignedDelta(
      Math.max(...family.hypotheses.map((h) => h.mean_difference))
    )} Dice).`,
    limitation:
      "3×3 median filtering slightly degraded clean and noisy slices by smoothing subtle anatomical edges.",
  };
}

/** Section F: Phase 5 mitigation findings preserving the four registered Holm-Bonferroni families. */
export function MitigationSummarySection({ phase5 }: MitigationSummarySectionProps) {
  if (!phase5 || !phase5.available || phase5.families.length === 0) {
    return (
      <section aria-labelledby="mitigation-summary-heading">
        <h2 id="mitigation-summary-heading" className="mb-3 text-[18px] font-semibold text-ink">
          Phase 5 Mitigation Findings
        </h2>
        <EmptyStatePanel
          title="Phase 5 mitigation artifacts unavailable"
          message={phase5?.error ?? "No verified Phase 5 mitigation results were returned by the backend."}
        />
      </section>
    );
  }

  const claheAudit = phase5.clahe_invariance_summary;
  const causal = claheAudit?.causal_decomposition;

  return (
    <section aria-labelledby="mitigation-summary-heading" className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="mitigation-summary-heading" className="text-[18px] font-semibold text-ink">
          Phase 5 Mitigation Findings
        </h2>
        <span className="text-xs text-ink-secondary">
          4 pre-registered Holm-Bonferroni families ({phase5.total_primary_hypotheses} hypotheses, family-wise α=0.05)
        </span>
      </div>

      {/* Forensic CLAHE normalization-invariance callout derived directly from preprocessing_stage_analysis.json */}
      {causal && (
        <div className="rounded-card border border-border-subtle bg-surface p-4">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 rounded-lg bg-[var(--status-warning-bg)] p-2 text-status-warning">
              <AlertCircle className="h-4 w-4" aria-hidden="true" />
            </div>
            <div className="text-xs leading-relaxed text-ink-secondary">
              <span className="font-semibold text-ink">
                Forensic Attribution Note (CLAHE Normalization Invariance):{" "}
              </span>
              On severe contrast reduction (unmitigated {formatScore(causal.severe_unmitigated_dice, 4)}),
              reported CLAHE rescue ({formatSignedDelta(causal.clahe_reported_rescue, 4)} to{" "}
              {formatScore(causal.severe_unmitigated_dice + causal.clahe_reported_rescue, 4)}) is{" "}
              <strong className="text-ink">100% reproduced by z-score re-normalization alone</strong>{" "}
              ({formatSignedDelta(causal.zscore_alone_rescue, 4)} to{" "}
              {formatScore(causal.clean_baseline_dice, 4)}), while CLAHE introduced a{" "}
              <strong className="text-ink">
                {formatSignedDelta(causal.clahe_net_effect_relative_to_zscore_alone, 4)}
              </strong>{" "}
              net penalty.
              {claheAudit?.stage_C_vs_D_unit_interval && (
                <span>
                  {" "}
                  (Clean vs. severe-contrast tensor max |diff| after min-max scaling:{" "}
                  <span className="font-mono">
                    {claheAudit.stage_C_vs_D_unit_interval.max_absolute_difference.toExponential(2)}
                  </span>
                  ).
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {phase5.families.map((fam) => {
          const interp = interpretFamily(fam);
          const bestHyp = fam.hypotheses.reduce((best, curr) =>
            curr.mean_difference > best.mean_difference ? curr : best
          );

          return (
            <article
              key={fam.family_id}
              className="flex flex-col justify-between rounded-card border border-border-subtle bg-surface p-5"
            >
              <div>
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold text-ink">{fam.family_name}</h3>
                    <p className="mt-0.5 text-xs text-ink-secondary">{fam.intervention_arm}</p>
                  </div>
                  <StatusBadge tone={interp.tone}>{interp.verdict}</StatusBadge>
                </div>

                <div className="mt-3 grid grid-cols-3 gap-2 rounded-xl border border-border-subtle bg-surface-secondary/50 p-3 text-xs">
                  <div>
                    <div className="text-[11px] text-ink-secondary">Hypotheses</div>
                    <div className="mt-0.5 font-mono font-semibold text-ink">
                      m = {fam.hypothesis_count_m}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-ink-secondary">Largest Δ (3D)</div>
                    <div className="mt-0.5 font-mono font-semibold text-ink">
                      {formatSignedDelta(bestHyp.mean_difference, 4)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-ink-secondary">Holm p (best)</div>
                    <div className="mt-0.5 font-mono font-semibold text-ink">
                      {formatPValue(bestHyp.paired_t_pvalue_holm)}
                    </div>
                  </div>
                </div>

                <p className="mt-3 text-xs leading-relaxed text-ink">{interp.summary}</p>
                <p className="mt-1.5 text-xs leading-relaxed text-ink-secondary">
                  <span className="font-medium text-ink">Limitation:</span> {interp.limitation}
                </p>
              </div>

              <div className="mt-4 border-t border-border-subtle pt-2.5 text-[11px] text-ink-secondary">
                Correction scope: {fam.correction_scope}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
