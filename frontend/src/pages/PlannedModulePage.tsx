import { ArrowLeft, Construction } from "lucide-react";
import { Link } from "react-router-dom";
import { StatusBadge } from "@/components/common/StatusBadge";
import type { NavItemConfig } from "@/components/navigation/nav-config";

interface PlannedModulePageProps {
  item: NavItemConfig;
}

/** Honest structural placeholder for workspace routes planned after Phase 1. */
export function PlannedModulePage({ item }: PlannedModulePageProps) {
  const Icon = item.icon;
  const isReports = item.id === "reports";

  return (
    <div className="mx-auto max-w-2xl py-8">
      <div className="rounded-card border border-border-subtle bg-surface p-6 sm:p-8">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-surface-secondary p-2.5 text-ink-secondary">
              <Icon className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-ink">{item.label}</h1>
              <p className="text-xs text-ink-secondary">Route: {item.path}</p>
            </div>
          </div>
          <StatusBadge tone="neutral">
            {isReports ? "Not Available" : "Not Yet Implemented"}
          </StatusBadge>
        </div>

        <div className="mt-5 flex items-start gap-3 rounded-xl border border-border-subtle bg-surface-secondary/50 p-4 text-sm text-ink-secondary">
          <Construction className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden="true" />
          <div className="space-y-1.5 text-xs leading-relaxed">
            <p className="font-medium text-ink">{item.description}</p>
            <p>
              This structural route is reserved in the CardioAI Research navigation shell. No
              placeholder metrics, mock predictions, or synthetic export workflows are displayed.
            </p>
          </div>
        </div>

        <div className="mt-6">
          <Link
            to="/overview"
            className="inline-flex items-center gap-2 rounded-xl border border-border-subtle bg-surface-secondary px-4 py-2 text-xs font-medium text-ink transition-colors hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
            Return to Research Overview
          </Link>
        </div>
      </div>
    </div>
  );
}
