import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { SETTINGS_NAV_ITEM, WORKSPACE_NAV_ITEMS } from "@/components/navigation/nav-config";
import { OverviewPage } from "@/pages/OverviewPage";
import { PlannedModulePage } from "@/pages/PlannedModulePage";
import { ThemeProvider } from "@/theme/theme-context";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
});

/** Application route tree rendered inside AppShell. */
export function AppRoutes() {
  const plannedWorkspaceRoutes = WORKSPACE_NAV_ITEMS.filter((item) => !item.implemented);

  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/overview" replace />} />
        <Route path="/overview" element={<OverviewPage />} />
        {plannedWorkspaceRoutes.map((item) => (
          <Route
            key={item.id}
            path={item.path}
            element={<PlannedModulePage item={item} />}
          />
        ))}
        <Route
          path={SETTINGS_NAV_ITEM.path}
          element={<PlannedModulePage item={SETTINGS_NAV_ITEM} />}
        />
        <Route path="*" element={<Navigate to="/overview" replace />} />
      </Routes>
    </AppShell>
  );
}

/** Root application component configuring QueryClient, ThemeProvider, and BrowserRouter. */
export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
