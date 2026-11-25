"""Tests for GET /api/v1/patients."""

import json
from pathlib import Path
from fastapi.testclient import TestClient


def test_patients_returns_only_test_cohort_matching_manifest(client: TestClient) -> None:
    """Verifies all 15 test patients and 131 slices match acdc_slice_manifest.json."""
    resp = client.get("/api/v1/patients")
    assert resp.status_code == 200
    data = resp.json()

    assert data["split"] == "test"
    assert data["phase"] == "ED"
    assert data["patient_count"] == 15
    assert data["total_slices"] == 131
    assert len(data["patients"]) == 15

    manifest_path = Path("data/processed/acdc_slice_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_test_patients = {
        s["patient_id"] for s in manifest["slices"] if s.get("split") == "test"
    }
    manifest_non_test_patients = {
        s["patient_id"] for s in manifest["slices"] if s.get("split") != "test"
    }

    returned_ids = {p["patient_id"] for p in data["patients"]}
    assert returned_ids == manifest_test_patients
    assert returned_ids.isdisjoint(manifest_non_test_patients)

    # Verify patients retain distinct slice counts and shapes (e.g. patient038 has 8 slices of 276x384)
    by_id = {p["patient_id"]: p for p in data["patients"]}
    assert by_id["patient001"]["slice_count"] == 10
    assert by_id["patient038"]["slice_count"] == 8
    assert by_id["patient038"]["raw_slice_shape"] == [276, 384]
    assert by_id["patient068"]["slice_count"] == 7
    assert all(p["split"] == "test" for p in data["patients"])


def test_patients_single_patient_filter(client: TestClient) -> None:
    """Verifies filtering by a valid test patient_id returns only that patient's slices."""
    resp = client.get("/api/v1/patients", params={"patient_id": "patient001"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_count"] == 1
    assert data["total_slices"] == 10
    assert data["patients"][0]["patient_id"] == "patient001"
    assert data["patients"][0]["available_slice_indices"] == list(range(10))
    assert data["patients"][0]["voxel_spacing"] is not None


def test_patients_rejects_train_val_and_unknown_patients(client: TestClient) -> None:
    """Verifies unknown, train, val, and path-traversal patient IDs are rejected cleanly."""
    # patient002 is in train split, patient003 is in val split
    for blocked_id in ("patient002", "patient003", "patient999"):
        resp = client.get("/api/v1/patients", params={"patient_id": blocked_id})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "PATIENT_NOT_FOUND"

    # Path traversal must be rejected cleanly with 400
    trav_resp = client.get("/api/v1/patients", params={"patient_id": "../patient001"})
    assert trav_resp.status_code == 400
    assert trav_resp.json()["error"]["code"] == "INVALID_PATIENT_ID"
