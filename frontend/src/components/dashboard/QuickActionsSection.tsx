import { Activity, ArrowRight, FileText, ShieldAlert, Wrench } from "lucide-react";
import { Link } from "react-router-dom";
import { StatusBadge } from "@/components/common/StatusBadge";

interface QuickActionItem {
  title: string;
  description: string;
  destination: string;
  icon: typeof Activity;
  availabilityLabel: string;
}

const QUICK_ACTIONS: QuickActionItem[] = [
  {
    title: "Start New Segmentation",
    description: "Run single-slice inference on held-out ACDC test patients.",
    destination: "/segmentation",
    icon: Activity,
    availabilityLabel: "Planned · Phase 2",
  },
  {
    title: "View Robustness Experiments",
    description: "Inspect noise, blur, and contrast curves across all 10 conditions.",
    destination: "/robustness",
    icon: ShieldAlert,
    availabilityLabel: "Planned · Phase 2",
  },
  {
    title: "Compare Mitigation Techniques",
    description: "Explore CLAHE, median filter, and morphological ablations.",
    destination: "/mitigation",
    icon: Wrench,
    availabilityLabel: "Planned · Phase 2",
  },
  {
    title: "View Research Reports",
    description: "No backend report-export endpoint exists in the current API.",
    destination: "/reports",
    icon: FileText,
    availabilityLabel: "Not Available",
  },
];

/** Section I: Honest navigation quick actions linking to structural workspace routes. */
export function QuickActionsSection() {
  return (
    <section aria-labelledby="quick-actions-heading">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="quick-actions-heading" className="text-[18px] font-semibold text-ink">
          Workspace Modules
        </h2>
        <span className="text-xs text-ink-secondary">
          Navigation routes prepared; interactive modules follow in subsequent phases
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {QUICK_ACTIONS.map((action) => {
          const Icon = action.icon;
          return (
            <Link
              key={action.destination}
              to={action.destination}
              className="group flex flex-col justify-between rounded-card border border-border-subtle bg-surface p-4 transition-colors hover:border-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <div>
                <div className="flex items-center justify-between gap-2">
                  <div className="rounded-lg bg-surface-secondary p-2 text-ink-secondary transition-colors group-hover:bg-accent-subtle group-hover:text-accent">
                    <Icon className="h-4 w-4" aria-hidden="true" />
                  </div>
                  <StatusBadge tone="neutral">{action.availabilityLabel}</StatusBadge>
                </div>
                <h3 className="mt-3 text-sm font-semibold text-ink group-hover:text-accent">
                  {action.title}
                </h3>
                <p className="mt-1 text-xs leading-relaxed text-ink-secondary">
                  {action.description}
                </p>
              </div>

              <div className="mt-4 flex items-center gap-1 text-xs font-medium text-ink-secondary group-hover:text-accent">
                <span>Open route ({action.destination})</span>
                <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" aria-hidden="true" />
              </div>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
