import { InfoTooltip } from "@/components/common/InfoTooltip";
import type { OverviewResponse } from "@/types/api";
import {
  findBaselineMetricByAggregation,
  findCanonicalCheckpoint,
  formatInteger,
  formatScore,
} from "@/utils/formatters";

interface BaselineMetricCardsProps {
  overview: OverviewResponse;
}

interface MetricCardSpec {
  id: string;
  label: string;
  value: string;
  unit: string;
  aggregationLabel: string;
  context: string;
  tooltip: string;
}

/** Section C: Five baseline metric cards populated strictly from verified backend data. */
export function BaselineMetricCards({ overview }: BaselineMetricCardsProps) {
  const canonical = findCanonicalCheckpoint(overview.checkpoints);
  const checkpointVerified = Boolean(canonical?.integrity_verified);

  const patientPooled = findBaselineMetricByAggregation(
    overview.baseline_metrics,
    "patient_level_3d_pooled"
  );
  const sliceLevel = findBaselineMetricByAggregation(
    overview.baseline_metrics,
    "slice_level"
  );

  const verifiedDice =
    checkpointVerified && patientPooled?.mean_dice_foreground !== null && patientPooled?.mean_dice_foreground !== undefined
      ? formatScore(patientPooled.mean_dice_foreground, 4)
      : "Not verified";

  const verifiedIou =
    checkpointVerified && patientPooled?.mean_iou_foreground !== null && patientPooled?.mean_iou_foreground !== undefined
      ? formatScore(patientPooled.mean_iou_foreground, 4)
      : "Not verified";

  const cards: MetricCardSpec[] = [
    {
      id: "baseline-dice",
      label: "Baseline Dice",
      value: verifiedDice,
      unit:
        checkpointVerified && patientPooled?.std_dice_foreground != null
          ? `± ${patientPooled.std_dice_foreground.toFixed(4)} SD`
          : "Foreground (RV, MYO, LV)",
      aggregationLabel: patientPooled
        ? `Patient-level 3D Pooled (n=${patientPooled.sample_count})`
        : "Patient-level 3D Pooled",
      context: sliceLevel
        ? `Slice-level avg (n=${sliceLevel.sample_count}): ${formatScore(sliceLevel.mean_dice_foreground, 4)}`
        : "Clean held-out test cohort",
      tooltip:
        "3D voxel-pooled macro foreground Dice across RV, MYO, and LV summed over all slices per patient (n=15), rather than unweighted 2D slice averaging (n=131).",
    },
    {
      id: "baseline-iou",
      label: "Baseline IoU",
      value: verifiedIou,
      unit: "Jaccard Index (FG)",
      aggregationLabel: patientPooled
        ? `Patient-level 3D Pooled (n=${patientPooled.sample_count})`
        : "Patient-level 3D Pooled",
      context: sliceLevel
        ? `Slice-level avg (n=${sliceLevel.sample_count}): ${formatScore(sliceLevel.mean_iou_foreground, 4)}`
        : "Clean held-out test cohort",
      tooltip:
        "Macro-averaged Intersection-over-Union across RV, MYO, and LV computed from per-patient 3D pooled voxel counts.",
    },
    {
      id: "test-patients",
      label: "Test Patients",
      value: formatInteger(overview.dataset.test_patient_count),
      unit: "Held-out ACDC patients",
      aggregationLabel: "Strict Patient-Level Split",
      context:
        overview.dataset.total_patients != null
          ? `${overview.dataset.test_patient_count} of ${overview.dataset.total_patients} total ACDC patients`
          : "Zero identity overlap with train/val",
      tooltip:
        "Stratified 15% held-out test cohort (3 patients per pathology group: DCM, HCM, MINF, NOR, RV) excluded from training and validation tuning.",
    },
    {
      id: "test-slices",
      label: "Test Slices",
      value: formatInteger(overview.dataset.test_slice_count),
      unit: `${overview.dataset.evaluated_phase} 2D slices`,
      aggregationLabel: "2D Short-Axis ED Slices",
      context:
        overview.dataset.test_foreground_slice_count != null
          ? `${overview.dataset.test_foreground_slice_count} foreground / ${overview.dataset.test_slice_count} total`
          : "Extracted from frame01 volumes",
      tooltip:
        "Total End-Diastolic 2D short-axis cine MRI slices across the 15 held-out test patients from acdc_slice_manifest.json.",
    },
    {
      id: "model-parameters",
      label: "Model Parameters",
      value: formatInteger(overview.model.parameter_count_total),
      unit: `${(overview.model.parameter_count_total / 1_000_000).toFixed(2)}M Float32`,
      aggregationLabel: "LightweightUNet2D",
      context: `${formatInteger(overview.model.parameter_count_trainable)} trainable params`,
      tooltip:
        "Exact parameter count measured from LightweightUNet2D (base_channels=16, depth=4, bilinear upsampling, BatchNorm).",
    },
  ];

  return (
    <section aria-labelledby="baseline-metrics-heading">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="baseline-metrics-heading" className="text-[18px] font-semibold text-ink">
          Verified Clean-Test Baseline
        </h2>
        <span className="text-xs text-ink-secondary">
          Patient-level 3D pooling reported separately from 2D slice averages
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {cards.map((card) => (
          <article
            key={card.id}
            className="flex flex-col justify-between rounded-card border border-border-subtle bg-surface p-4"
          >
            <div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-ink-secondary">{card.label}</span>
                <InfoTooltip label={`About ${card.label}`} content={card.tooltip} />
              </div>
              <div className="mt-2 text-2xl font-semibold tracking-tight text-ink tabular-nums">
                {card.value}
              </div>
              <div className="mt-0.5 text-xs text-ink-secondary">{card.unit}</div>
            </div>

            <div className="mt-4 border-t border-border-subtle pt-3">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-accent">
                {card.aggregationLabel}
              </div>
              <div className="mt-1 text-xs text-ink-secondary">{card.context}</div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
