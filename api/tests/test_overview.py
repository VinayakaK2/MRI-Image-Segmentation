"""Tests for GET /api/v1/overview."""

from fastapi.testclient import TestClient


def test_overview_metadata_and_counts(client: TestClient) -> None:
    """Verifies model specs, SHA-256, dataset counts, and separated aggregation levels."""
    resp = client.get("/api/v1/overview")
    assert resp.status_code == 200
    data = resp.json()

    model = data["model"]
    assert model["architecture"] == "LightweightUNet2D"
    assert model["parameter_count_total"] == 1963860
    assert model["parameter_count_trainable"] == 1963860
    assert model["input_dimensions"] == [1, 1, 256, 256]
    assert model["output_dimensions"] == [1, 4, 256, 256]
    assert model["class_mapping"] == {
        "0": "Background",
        "1": "RV",
        "2": "MYO",
        "3": "LV",
    }

    canonical_ckpts = [c for c in data["checkpoints"] if c["checkpoint_id"] == "canonical_50epoch"]
    assert len(canonical_ckpts) == 1
    assert canonical_ckpts[0]["expected_sha256"] == "a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38"
    assert canonical_ckpts[0]["verified_sha256"] == canonical_ckpts[0]["expected_sha256"]
    assert canonical_ckpts[0]["integrity_verified"] is True

    dataset = data["dataset"]
    assert dataset["test_patient_count"] == 15
    assert dataset["test_slice_count"] == 131
    assert len(dataset["test_patient_ids"]) == 15

    # Verify slice-level and patient-level metrics are kept strictly separate
    agg_levels = {m["aggregation_level"]: m for m in data["baseline_metrics"]}
    assert "slice_level" in agg_levels
    assert "patient_level_3d_pooled" in agg_levels
    assert agg_levels["slice_level"]["sample_count"] == 131
    assert agg_levels["patient_level_3d_pooled"]["sample_count"] == 15
    assert abs(agg_levels["slice_level"]["mean_dice_foreground"] - 0.8296715765) < 1e-5
    assert abs(agg_levels["patient_level_3d_pooled"]["mean_dice_foreground"] - 0.8957515247) < 1e-5

    phases = {e["phase"]: e for e in data["experiment_summaries"]}
    assert phases[4]["available"] is True
    assert phases[4]["conditions_count"] == 10
    assert phases[5]["available"] is True
    assert phases[5]["total_primary_hypotheses"] == 19
