import { HeartPulse, X } from "lucide-react";
import { NavLink } from "react-router-dom";
import { SETTINGS_NAV_ITEM, WORKSPACE_NAV_ITEMS, type NavItemConfig } from "@/components/navigation/nav-config";

interface SidebarProps {
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

function SidebarNavItem({
  item,
  onNavigate,
}: {
  item: NavItemConfig;
  onNavigate: () => void;
}) {
  const Icon = item.icon;
  return (
    <NavLink
      to={item.path}
      onClick={onNavigate}
      className={({ isActive }) =>
        [
          "group flex items-center justify-between rounded-xl px-3 py-2.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
          isActive
            ? "bg-accent-subtle text-accent"
            : "text-ink-secondary hover:bg-surface-secondary hover:text-ink",
        ].join(" ")
      }
    >
      {({ isActive }) => (
        <>
          <span className="flex items-center gap-3">
            <Icon
              className={`h-4 w-4 shrink-0 ${isActive ? "text-accent" : "text-ink-secondary group-hover:text-ink"}`}
              aria-hidden="true"
            />
            <span>{item.label}</span>
          </span>
          {!item.implemented && (
            <span
              className="rounded-md border border-border-subtle bg-surface-secondary px-1.5 py-0.5 text-[10px] font-medium text-ink-secondary"
              title="Module structure reserved; not yet implemented"
            >
              Planned
            </span>
          )}
        </>
      )}
    </NavLink>
  );
}

/** Persistent desktop sidebar and responsive mobile slide-over drawer. */
export function Sidebar({ mobileOpen, onCloseMobile }: SidebarProps) {
  const sidebarContent = (
    <div className="flex h-full flex-col justify-between bg-surface px-4 py-5">
      <div>
        <div className="flex items-center justify-between px-1 pb-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-subtle text-accent">
              <HeartPulse className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <div className="text-sm font-semibold tracking-tight text-ink">
                CardioAI Research
              </div>
              <div className="text-[11px] text-ink-secondary">
                ACDC Robustness Workstation
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close navigation menu"
            className="rounded-lg p-1.5 text-ink-secondary hover:bg-surface-secondary hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent lg:hidden"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <div className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-wider text-ink-secondary">
          Workspace
        </div>
        <nav aria-label="Primary workspace navigation" className="space-y-1">
          {WORKSPACE_NAV_ITEMS.map((item) => (
            <SidebarNavItem key={item.id} item={item} onNavigate={onCloseMobile} />
          ))}
        </nav>
      </div>

      <div className="border-t border-border-subtle pt-4">
        <SidebarNavItem item={SETTINGS_NAV_ITEM} onNavigate={onCloseMobile} />
        <p className="mt-3 px-3 text-[11px] leading-relaxed text-ink-secondary">
          Local research workstation · Not for clinical diagnosis.
        </p>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop persistent sidebar */}
      <aside className="hidden w-64 shrink-0 border-r border-border-subtle bg-surface lg:block">
        <div className="sticky top-0 h-screen overflow-y-auto">{sidebarContent}</div>
      </aside>

      {/* Mobile/Tablet drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation drawer">
          <div
            className="fixed inset-0 bg-black/50 transition-opacity"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <div className="relative z-10 h-full w-72 max-w-[85vw] border-r border-border-subtle bg-surface shadow-xl">
            {sidebarContent}
          </div>
        </div>
      )}
    </>
  );
}
