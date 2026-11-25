"""Tests for GET /api/v1/experiments."""

from pathlib import Path
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import build_research_paths


def test_experiments_loads_phase4_and_phase5_with_19_hypotheses(client: TestClient) -> None:
    """Verifies Phase 4 (10 conditions) and Phase 5 (4 families, 19 hypotheses) load accurately."""
    resp = client.get("/api/v1/experiments")
    assert resp.status_code == 200
    data = resp.json()

    # Phase 4 checks
    p4 = data["phase4"]
    assert p4["available"] is True
    assert p4["n_patients"] == 15
    assert p4["total_test_slices"] == 131
    assert len(p4["conditions"]) == 10
    clean = next(c for c in p4["conditions"] if c["condition_id"] == "clean_baseline")
    assert clean["statistical_comparison_vs_clean"] is None
    assert abs(clean["slice_level"]["mean_dice_foreground"] - 0.8296715765) < 1e-5
    assert abs(clean["patient_level_summary"]["mean_dice_foreground"]["mean"] - 0.8957515247) < 1e-5

    blur_sev = next(c for c in p4["conditions"] if c["condition_id"] == "gaussian_blur_severe")
    assert blur_sev["statistical_comparison_vs_clean"] is not None
    assert blur_sev["statistical_comparison_vs_clean"]["n_patients"] == 15
    assert blur_sev["statistical_comparison_vs_clean"]["degrees_of_freedom"] == 14

    # Phase 5 checks
    p5 = data["phase5"]
    assert p5["available"] is True
    assert p5["total_primary_hypotheses"] == 19
    assert len(p5["families"]) == 4

    family_sizes = {f["family_id"]: len(f["hypotheses"]) for f in p5["families"]}
    assert family_sizes == {
        "family_1_postproc_alone": 10,
        "family_2_clahe_alone": 3,
        "family_3_clahe_plus_postproc": 3,
        "family_4_median_plus_postproc": 3,
    }
    for fam in p5["families"]:
        assert fam["hypothesis_count_m"] == len(fam["hypotheses"])
        for hyp in fam["hypotheses"]:
            assert hyp["n_patients"] == 15
            assert hyp["degrees_of_freedom"] == 14
            assert "Holm-Bonferroni" in hyp["correction_method"]
            assert fam["correction_scope"] == hyp["correction_family_scope"]

    assert len(p5["negative_controls"]) == 9
    assert p5["clahe_invariance_summary"] is not None


def test_experiments_phase_family_and_patient_filters(client: TestClient) -> None:
    """Verifies phase, family, and patient_id query filters work and preserve Holm boundaries."""
    # Phase 5 Family 2 filter
    r_f2 = client.get("/api/v1/experiments", params={"phase": 5, "family": "family_2_clahe_alone"})
    assert r_f2.status_code == 200
    d_f2 = r_f2.json()
    assert d_f2["phase4"] is None
    assert len(d_f2["phase5"]["families"]) == 1
    assert d_f2["phase5"]["families"][0]["hypothesis_count_m"] == 3
    assert len(d_f2["phase5"]["families"][0]["hypotheses"]) == 3

    # Phase 4 + patient_id filter
    r_p4 = client.get("/api/v1/experiments", params={"phase": 4, "family": "gaussian_blur", "patient_id": "patient001"})
    assert r_p4.status_code == 200
    d_p4 = r_p4.json()
    assert d_p4["phase5"] is None
    assert len(d_p4["phase4"]["conditions"]) == 3
    for cond in d_p4["phase4"]["conditions"]:
        assert list(cond["per_patient_metrics"].keys()) == ["patient001"]

    # Invalid phase
    assert client.get("/api/v1/experiments", params={"phase": 3}).status_code == 422
    # Invalid family
    assert client.get("/api/v1/experiments", params={"phase": 5, "family": "invalid_fam"}).status_code == 422
    # Train patient rejected
    assert client.get("/api/v1/experiments", params={"patient_id": "patient002"}).status_code == 404


def test_experiments_missing_artifacts_returns_explicit_error(tmp_path: Path) -> None:
    """Verifies missing Phase 4/5 artifacts produce explicit errors instead of fake data."""
    app = create_app(paths=build_research_paths(project_root=tmp_path))
    with TestClient(app) as iso_client:
        r_all = iso_client.get("/api/v1/experiments")
        assert r_all.status_code == 200
        body = r_all.json()
        assert body["phase4"]["available"] is False
        assert body["phase4"]["error"] is not None
        assert body["phase5"]["available"] is False
        assert body["phase5"]["error"] is not None

        r_p5 = iso_client.get("/api/v1/experiments", params={"phase": 5})
        assert r_p5.status_code == 503
        assert r_p5.json()["error"]["code"] == "ARTIFACT_UNAVAILABLE"
