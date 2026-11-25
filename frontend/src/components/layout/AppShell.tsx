import { Menu } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { BackendStatusIndicator } from "@/components/navigation/BackendStatusIndicator";
import { Sidebar } from "@/components/navigation/Sidebar";
import { ThemeSelector } from "@/components/navigation/ThemeSelector";
import { findNavItemByPath } from "@/components/navigation/nav-config";
import { useHealth } from "@/hooks/use-health";

interface AppShellProps {
  children: React.ReactNode;
}

/** Global application shell combining persistent sidebar, header, health status, and theme selector. */
export function AppShell({ children }: AppShellProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const activeNav = findNavItemByPath(location.pathname);

  const healthQuery = useHealth();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && mobileOpen) {
        setMobileOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [mobileOpen]);

  return (
    <div className="flex min-h-screen bg-canvas text-ink">
      <Sidebar mobileOpen={mobileOpen} onCloseMobile={() => setMobileOpen(false)} />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 border-b border-border-subtle bg-surface/95 backdrop-blur-sm">
          <div className="mx-auto flex max-w-7xl items-center justify-between gap-2 px-3 py-3 sm:gap-3 sm:px-6 lg:px-8">
            <div className="flex min-w-0 items-center gap-2 sm:gap-3">
              <button
                type="button"
                onClick={() => setMobileOpen(true)}
                aria-label="Open navigation menu"
                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-border-subtle bg-surface text-ink-secondary transition-colors hover:bg-surface-secondary hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent sm:h-9 sm:w-9 lg:hidden"
              >
                <Menu className="h-4 w-4 sm:h-5 sm:w-5" aria-hidden="true" />
              </button>

              <nav aria-label="Breadcrumb" className="min-w-0">
                <ol className="flex items-center gap-1.5 text-xs text-ink-secondary">
                  <li className="hidden sm:inline">Workspace</li>
                  <li className="hidden sm:inline" aria-hidden="true">
                    /
                  </li>
                  <li className="truncate text-sm font-semibold text-ink">
                    {activeNav.label}
                  </li>
                </ol>
              </nav>
            </div>

            <div className="flex shrink-0 items-center gap-1.5 sm:gap-3">
              <BackendStatusIndicator
                health={healthQuery.data}
                isLoading={healthQuery.isLoading}
                isError={healthQuery.isError}
                error={healthQuery.error}
                onRefresh={() => void healthQuery.refetch()}
                isFetching={healthQuery.isFetching}
              />
              <ThemeSelector />
            </div>
          </div>
        </header>

        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8">
          {children}
        </main>
      </div>
    </div>
  );
}
