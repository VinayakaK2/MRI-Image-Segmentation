import { StatusBadge } from "@/components/common/StatusBadge";
import type { OverviewResponse } from "@/types/api";

interface OverviewHeaderProps {
  overview: OverviewResponse | undefined;
}

/** Section A: Compact header displaying title, scope, and verified model/dataset metadata. */
export function OverviewHeader({ overview }: OverviewHeaderProps) {
  return (
    <section aria-labelledby="overview-page-title" className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1
          id="overview-page-title"
          className="text-2xl font-semibold tracking-tight text-ink sm:text-[26px]"
        >
          Research Overview
        </h1>
        <p className="mt-1 text-sm text-ink-secondary">
          Cardiac MRI segmentation and robustness evaluation.
        </p>
      </div>

      {overview && (
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge tone="accent">{overview.model.architecture}</StatusBadge>
          <StatusBadge tone="neutral">
            {overview.dataset.dataset_name.split(" ")[0]} · {overview.dataset.evaluated_phase}
          </StatusBadge>
        </div>
      )}
    </section>
  );
}
