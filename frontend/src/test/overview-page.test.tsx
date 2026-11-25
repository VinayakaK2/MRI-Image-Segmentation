import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "@/app/App";
import { ApiClientError } from "@/services/api/client";
import * as experimentsApi from "@/services/api/experiments";
import * as healthApi from "@/services/api/health";
import * as overviewApi from "@/services/api/overview";
import { MOCK_EXPERIMENTS_OK, MOCK_HEALTH_OK, MOCK_OVERVIEW_OK } from "@/test/fixtures";
import { ThemeProvider } from "@/theme/theme-context";

function renderOverview() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <MemoryRouter initialEntries={["/overview"]}>
          <AppRoutes />
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

describe("OverviewPage data integration, aggregation clarity, and error states", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders backend-verified metrics, separated aggregation levels, Phase 4/5 findings, and invariant class colors", async () => {
    const user = userEvent.setup();
    vi.spyOn(healthApi, "getHealth").mockResolvedValue(MOCK_HEALTH_OK);
    vi.spyOn(overviewApi, "getOverview").mockResolvedValue(MOCK_OVERVIEW_OK);
    vi.spyOn(experimentsApi, "getExperiments").mockResolvedValue(MOCK_EXPERIMENTS_OK);

    renderOverview();

    // Baseline metrics from backend
    expect(await screen.findByText("0.8958")).toBeInTheDocument();
    expect(screen.getByText("0.8161")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
    expect(screen.getByText("131")).toBeInTheDocument();
    expect(screen.getAllByText("1,963,860").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText(/Patient-level 3D Pooled \(n=15\)/i).length).toBeGreaterThanOrEqual(2);

    // Phase 4 robustness & Phase 5 mitigation sections
    expect(screen.getByText("Gaussian Blur")).toBeInTheDocument();
    expect(screen.getByText("Family 1 (Post-Proc, m=10)")).toBeInTheDocument();
    expect(screen.getByText(/Forensic Attribution Note \(CLAHE Normalization Invariance\)/i)).toBeInTheDocument();

    // Switch chart aggregation level cleanly
    await user.click(screen.getByRole("button", { name: "Slice 2D (n=131)" }));
    expect(screen.getByText(/Slice-level 2D arithmetic mean \(n=131\)/i)).toBeInTheDocument();

    // Switch Experiment Overview table to Phase 5 hypotheses
    await user.click(screen.getByRole("tab", { name: /Phase 5 · Hypotheses/i }));
    expect(screen.getAllByText("contrast_reduction_severe").length).toBeGreaterThanOrEqual(2);
  });

  it("displays 'Not verified' when canonical checkpoint integrity fails instead of hardcoded fallback", async () => {
    vi.spyOn(healthApi, "getHealth").mockResolvedValue(MOCK_HEALTH_OK);
    vi.spyOn(overviewApi, "getOverview").mockResolvedValue({
      ...MOCK_OVERVIEW_OK,
      checkpoints: [
        {
          ...MOCK_OVERVIEW_OK.checkpoints[0],
          integrity_verified: false,
          verified_sha256: null,
        },
      ],
    });
    vi.spyOn(experimentsApi, "getExperiments").mockResolvedValue(MOCK_EXPERIMENTS_OK);

    renderOverview();

    expect(await screen.findAllByText("Not verified")).toHaveLength(2);
    expect(screen.queryByText("0.8958")).not.toBeInTheDocument();
  });

  it("renders informative error state with retry action when backend is unreachable", async () => {
    const user = userEvent.setup();
    const err = new ApiClientError(
      "Unable to connect to the local FastAPI backend",
      "network_unreachable",
      null,
      "BACKEND_UNREACHABLE"
    );
    const healthSpy = vi.spyOn(healthApi, "getHealth").mockRejectedValue(err);
    const overviewSpy = vi.spyOn(overviewApi, "getOverview").mockRejectedValue(err);
    const expSpy = vi.spyOn(experimentsApi, "getExperiments").mockRejectedValue(err);

    renderOverview();

    expect(await screen.findByText("Unable to load backend overview metadata")).toBeInTheDocument();
    expect(screen.getByText("Disconnected")).toBeInTheDocument();

    // Retry invokes backend calls again
    const retryButtons = screen.getAllByRole("button", { name: /Retry/i });
    await user.click(retryButtons[0]);
    expect(healthSpy).toHaveBeenCalledTimes(2);
    expect(overviewSpy).toHaveBeenCalledTimes(2);
    expect(expSpy).toHaveBeenCalledTimes(2);
  });

  it("renders empty states when Phase 4 and Phase 5 artifacts are unavailable", async () => {
    vi.spyOn(healthApi, "getHealth").mockResolvedValue(MOCK_HEALTH_OK);
    vi.spyOn(overviewApi, "getOverview").mockResolvedValue(MOCK_OVERVIEW_OK);
    vi.spyOn(experimentsApi, "getExperiments").mockResolvedValue({
      phase_filter: null,
      family_filter: null,
      patient_id_filter: null,
      phase4: {
        available: false,
        error: "Phase 4 missing",
        experiment_id: null,
        checkpoint_sha256: null,
        total_test_slices: null,
        n_patients: null,
        conditions: [],
      },
      phase5: {
        available: false,
        error: "Phase 5 missing",
        experiment_id: null,
        checkpoint_sha256: null,
        total_primary_hypotheses: 0,
        locked_mitigation_hyperparameters: {},
        statistical_protocol: {},
        families: [],
        arm_metrics: [],
        negative_controls: [],
        clahe_invariance_summary: null,
      },
    });

    renderOverview();

    expect(await screen.findByText("Phase 4 robustness artifacts unavailable")).toBeInTheDocument();
    expect(screen.getByText("Phase 5 mitigation artifacts unavailable")).toBeInTheDocument();
    expect(screen.getByText("Performance comparison chart unavailable")).toBeInTheDocument();
  });
});
