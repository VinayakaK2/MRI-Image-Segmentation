import { ErrorStatePanel } from "@/components/common/StateMessage";
import { BaselineMetricCards } from "@/components/dashboard/BaselineMetricCards";
import { ExperimentOverviewTable } from "@/components/dashboard/ExperimentOverviewTable";
import { MitigationSummarySection } from "@/components/dashboard/MitigationSummarySection";
import { OverviewHeader } from "@/components/dashboard/OverviewHeader";
import { PerformanceComparisonChart } from "@/components/dashboard/PerformanceComparisonChart";
import { ProjectModelSummary } from "@/components/dashboard/ProjectModelSummary";
import { QuickActionsSection } from "@/components/dashboard/QuickActionsSection";
import { RobustnessSummarySection } from "@/components/dashboard/RobustnessSummarySection";
import { useExperiments } from "@/hooks/use-experiments";
import { useHealth } from "@/hooks/use-health";
import { useOverview } from "@/hooks/use-overview";
import { ApiClientError } from "@/services/api/client";

function OverviewLoadingSkeleton() {
  return (
    <div aria-busy="true" aria-label="Loading research overview" className="space-y-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {Array.from({ length: 5 }).map((_, idx) => (
          <div
            key={idx}
            className="h-36 animate-pulse rounded-card border border-border-subtle bg-surface"
          />
        ))}
      </div>
      <div className="h-56 animate-pulse rounded-card border border-border-subtle bg-surface" />
    </div>
  );
}

/** Primary Overview Dashboard (/overview) driven strictly by FastAPI backend responses. */
export function OverviewPage() {
  const healthQuery = useHealth();
  const overviewQuery = useOverview();
  const experimentsQuery = useExperiments();

  const overviewError = overviewQuery.error;
  const overviewCode =
    overviewError instanceof ApiClientError ? overviewError.errorCode : undefined;

  const experimentsError = experimentsQuery.error;
  const experimentsCode =
    experimentsError instanceof ApiClientError ? experimentsError.errorCode : undefined;

  return (
    <div className="space-y-8">
      {/* Section A: Page Header */}
      <OverviewHeader overview={overviewQuery.data} />

      {/* Sections C & D: Baseline Metrics & Project Summary */}
      {overviewQuery.isLoading ? (
        <OverviewLoadingSkeleton />
      ) : overviewQuery.isError || !overviewQuery.data ? (
        <ErrorStatePanel
          title="Unable to load backend overview metadata"
          message={
            overviewError instanceof Error
              ? overviewError.message
              : "Failed to retrieve GET /api/v1/overview from the local FastAPI backend."
          }
          code={overviewCode}
          onRetry={() => {
            void healthQuery.refetch();
            void overviewQuery.refetch();
            void experimentsQuery.refetch();
          }}
          isRetrying={overviewQuery.isFetching}
        />
      ) : (
        <>
          <BaselineMetricCards overview={overviewQuery.data} />
          <ProjectModelSummary overview={overviewQuery.data} />
        </>
      )}

      {/* Sections E, F, G, H: Phase 4 & 5 Findings, Chart, and Experiment Overview */}
      {experimentsQuery.isLoading ? (
        <div
          aria-busy="true"
          aria-label="Loading Phase 4 and Phase 5 experiments"
          className="h-72 animate-pulse rounded-card border border-border-subtle bg-surface"
        />
      ) : experimentsQuery.isError || !experimentsQuery.data ? (
        <ErrorStatePanel
          title="Unable to load Phase 4 and Phase 5 experiment artifacts"
          message={
            experimentsError instanceof Error
              ? experimentsError.message
              : "Failed to retrieve GET /api/v1/experiments from the local FastAPI backend."
          }
          code={experimentsCode}
          onRetry={() => void experimentsQuery.refetch()}
          isRetrying={experimentsQuery.isFetching}
        />
      ) : (
        <>
          <RobustnessSummarySection phase4={experimentsQuery.data.phase4} />
          <MitigationSummarySection phase5={experimentsQuery.data.phase5} />
          <PerformanceComparisonChart phase4={experimentsQuery.data.phase4} />
          <ExperimentOverviewTable experiments={experimentsQuery.data} />
        </>
      )}

      {/* Section I: Quick Actions */}
      <QuickActionsSection />
    </div>
  );
}
