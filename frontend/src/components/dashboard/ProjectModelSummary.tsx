import { StatusBadge } from "@/components/common/StatusBadge";
import { ANATOMICAL_CLASS_SPEC } from "@/theme/theme-context";
import type { OverviewResponse } from "@/types/api";
import { findCanonicalCheckpoint, formatInteger, formatScore, shortenSha256 } from "@/utils/formatters";

interface ProjectModelSummaryProps {
  overview: OverviewResponse;
}

/** Section D: Compact Project, Model, Canonical Checkpoint, Dataset, and Class Legend summary. */
export function ProjectModelSummary({ overview }: ProjectModelSummaryProps) {
  const { model, dataset } = overview;
  const canonical = findCanonicalCheckpoint(overview.checkpoints);

  const inputDims =
    model.input_dimensions.length >= 4
      ? `${model.input_dimensions[2]} × ${model.input_dimensions[3]} 2D MRI slice (${model.in_channels} ch)`
      : `${model.input_dimensions.join(" × ")}`;

  const pathologyEntries = Object.entries(dataset.pathology_distribution_test);

  return (
    <section aria-labelledby="project-summary-heading" className="rounded-card border border-border-subtle bg-surface p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 id="project-summary-heading" className="text-[18px] font-semibold text-ink">
            Project &amp; Model Summary
          </h2>
          <p className="mt-0.5 text-xs text-ink-secondary">
            {overview.project_title}
          </p>
        </div>
        <StatusBadge tone="neutral">
          Depth {model.depth} · Base {model.base_channels}ch · {model.bilinear_upsample ? "Bilinear" : "Transposed"}
        </StatusBadge>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {/* Architecture & Input */}
        <div className="rounded-xl border border-border-subtle bg-surface-secondary/50 p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-secondary">
            Model Architecture
          </div>
          <dl className="mt-2.5 space-y-1.5 text-sm">
            <div className="flex justify-between gap-2">
              <dt className="text-ink-secondary">Model</dt>
              <dd className="font-medium text-ink">{model.architecture}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-ink-secondary">Parameters</dt>
              <dd className="font-mono text-xs font-medium text-ink">
                {formatInteger(model.parameter_count_total)}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-ink-secondary">Input</dt>
              <dd className="text-right text-xs font-medium text-ink">{inputDims}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-ink-secondary">BatchNorm / Dropout</dt>
              <dd className="text-xs font-medium text-ink">
                {model.use_batchnorm ? "Yes" : "No"} · {model.dropout}
              </dd>
            </div>
          </dl>
        </div>

        {/* Canonical Checkpoint */}
        <div className="rounded-xl border border-border-subtle bg-surface-secondary/50 p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-secondary">
            Canonical Checkpoint
          </div>
          {canonical ? (
            <dl className="mt-2.5 space-y-1.5 text-sm">
              <div className="flex justify-between gap-2">
                <dt className="text-ink-secondary">ID</dt>
                <dd className="font-mono text-xs font-medium text-ink">{canonical.checkpoint_id}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-ink-secondary">Selected Epoch</dt>
                <dd className="text-xs font-medium text-ink">
                  Epoch {canonical.best_epoch ?? "N/A"} / {canonical.epochs_completed}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-ink-secondary">Val FG Dice</dt>
                <dd className="font-mono text-xs font-medium text-ink">
                  {formatScore(canonical.best_val_foreground_dice, 4)}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-ink-secondary">SHA-256</dt>
                <dd
                  className="font-mono text-xs font-medium text-ink"
                  title={canonical.verified_sha256 ?? canonical.expected_sha256}
                >
                  {shortenSha256(canonical.verified_sha256 ?? canonical.expected_sha256, 14)}
                </dd>
              </div>
            </dl>
          ) : (
            <p className="mt-2.5 text-xs text-ink-secondary">No canonical checkpoint metadata available.</p>
          )}
        </div>

        {/* Dataset & Test Split */}
        <div className="rounded-xl border border-border-subtle bg-surface-secondary/50 p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-secondary">
            Dataset &amp; Evaluation Cohort
          </div>
          <dl className="mt-2.5 space-y-1.5 text-sm">
            <div className="flex justify-between gap-2">
              <dt className="text-ink-secondary">Dataset</dt>
              <dd className="text-xs font-medium text-ink">ACDC</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-ink-secondary">Phase Scope</dt>
              <dd className="text-xs font-medium text-ink">{dataset.evaluated_phase}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-ink-secondary">Held-Out Test</dt>
              <dd className="text-xs font-medium text-ink">
                {dataset.test_patient_count} patients · {dataset.test_slice_count} slices
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-ink-secondary">Pathology Mix</dt>
              <dd className="text-right text-xs font-medium text-ink">
                {pathologyEntries.map(([grp, count]) => `${grp}:${count}`).join(" · ")}
              </dd>
            </div>
          </dl>
        </div>

        {/* Theme-Invariant Output Classes */}
        <div className="rounded-xl border border-border-subtle bg-surface-secondary/50 p-4">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-ink-secondary">
              Output Classes ({model.out_channels})
            </span>
            <span className="text-[10px] text-ink-secondary">Theme-invariant</span>
          </div>
          <ul className="mt-2.5 space-y-2">
            {ANATOMICAL_CLASS_SPEC.map((cls) => (
              <li key={cls.id} className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2 text-ink">
                  <span
                    className="inline-block h-3 w-3 rounded-sm border border-white/20"
                    style={{ backgroundColor: cls.cssVar }}
                    aria-hidden="true"
                  />
                  <span className="font-medium">
                    {cls.id}: {model.class_mapping[String(cls.id)] ?? cls.code}
                  </span>
                </span>
                <span className="text-ink-secondary">{cls.label}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
