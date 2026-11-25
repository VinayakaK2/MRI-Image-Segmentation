import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "@/app/App";
import { BackendStatusIndicator } from "@/components/navigation/BackendStatusIndicator";
import { ApiClientError } from "@/services/api/client";
import * as experimentsApi from "@/services/api/experiments";
import * as healthApi from "@/services/api/health";
import * as overviewApi from "@/services/api/overview";
import { MOCK_EXPERIMENTS_OK, MOCK_HEALTH_OK, MOCK_OVERVIEW_OK } from "@/test/fixtures";
import { ThemeProvider } from "@/theme/theme-context";

function renderAppAt(initialPath: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <MemoryRouter initialEntries={[initialPath]}>
          <AppRoutes />
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

describe("Application shell, navigation, theme persistence, and backend status", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.classList.remove("dark");
    vi.spyOn(healthApi, "getHealth").mockResolvedValue(MOCK_HEALTH_OK);
    vi.spyOn(overviewApi, "getOverview").mockResolvedValue(MOCK_OVERVIEW_OK);
    vi.spyOn(experimentsApi, "getExperiments").mockResolvedValue(MOCK_EXPERIMENTS_OK);
  });

  it("navigates across sidebar links, preserves active state, and shows honest planned routes", async () => {
    const user = userEvent.setup();
    renderAppAt("/overview");

    expect(await screen.findByRole("heading", { name: "Research Overview" })).toBeInTheDocument();

    const nav = screen.getByRole("navigation", { name: "Primary workspace navigation" });
    const overviewLink = within(nav).getByRole("link", { name: /Overview/i });
    expect(overviewLink).toHaveAttribute("aria-current", "page");

    // Navigate to unimplemented route (/segmentation) and verify honest state
    await user.click(within(nav).getByRole("link", { name: /Segmentation/i }));
    expect(await screen.findByRole("heading", { name: "Segmentation" })).toBeInTheDocument();
    expect(screen.getByText("Not Yet Implemented")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Return to Research Overview/i })).toBeInTheDocument();
  });

  it("persists Light, Dark, and System theme modes via localStorage", async () => {
    const user = userEvent.setup();
    renderAppAt("/overview");

    const darkBtn = await screen.findByRole("button", { name: /Dark/i });
    const lightBtn = screen.getByRole("button", { name: /Light/i });
    const systemBtn = screen.getByRole("button", { name: /System/i });

    expect(systemBtn).toHaveAttribute("aria-pressed", "true");

    await user.click(darkBtn);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(window.localStorage.getItem("cardioai-theme-mode")).toBe("dark");

    await user.click(lightBtn);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(window.localStorage.getItem("cardioai-theme-mode")).toBe("light");

    await user.click(systemBtn);
    expect(window.localStorage.getItem("cardioai-theme-mode")).toBe("system");
  });

  it("follows OS preference in System mode when matchMedia indicates dark", async () => {
    const origMatchMedia = window.matchMedia;
    window.matchMedia = ((query: string) => ({
      matches: query === "(prefers-color-scheme: dark)",
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    })) as typeof window.matchMedia;

    try {
      renderAppAt("/overview");
      await screen.findByRole("heading", { name: "Research Overview" });
      expect(document.documentElement.classList.contains("dark")).toBe(true);
    } finally {
      window.matchMedia = origMatchMedia;
    }
  });

  it("displays explicit Initializing, Connected, Disconnected, and Error backend states", () => {
    const noop = () => {};

    const { rerender } = render(
      <BackendStatusIndicator
        health={undefined}
        isLoading={true}
        isError={false}
        error={null}
        onRefresh={noop}
        isFetching={true}
      />
    );
    expect(screen.getByText("Initializing")).toBeInTheDocument();

    rerender(
      <BackendStatusIndicator
        health={MOCK_HEALTH_OK}
        isLoading={false}
        isError={false}
        error={null}
        onRefresh={noop}
        isFetching={false}
      />
    );
    expect(screen.getByText("Connected")).toBeInTheDocument();

    rerender(
      <BackendStatusIndicator
        health={undefined}
        isLoading={false}
        isError={true}
        error={new ApiClientError("Unreachable", "network_unreachable", null, "BACKEND_UNREACHABLE")}
        onRefresh={noop}
        isFetching={false}
      />
    );
    expect(screen.getByText("Disconnected")).toBeInTheDocument();

    rerender(
      <BackendStatusIndicator
        health={{ ...MOCK_HEALTH_OK, status: "degraded", model_initialized: false, model_load_error: "Missing ckpt" }}
        isLoading={false}
        isError={false}
        error={null}
        onRefresh={noop}
        isFetching={false}
      />
    );
    expect(screen.getByText("Error")).toBeInTheDocument();
  });
});
