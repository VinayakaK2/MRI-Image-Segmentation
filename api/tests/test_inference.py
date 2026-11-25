"""Tests for POST /api/v1/inference/segment."""

import base64
from io import BytesIO
import json
from pathlib import Path
import shutil
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from api.app import create_app
from api.config import build_research_paths
import api.utils.checkpoint as ckpt_utils


def _decode_png_array(b64_str: str) -> np.ndarray:
    """Decodes a base64-encoded PNG image into a NumPy array."""
    raw_bytes = base64.b64decode(b64_str)
    with Image.open(BytesIO(raw_bytes)) as img:
        return np.asarray(img)


def test_clean_inference_matches_phase5_slice_benchmark(client: TestClient) -> None:
    """Verifies clean inference works, returns valid mask/overlay, and matches Phase 5 slice Dice."""
    resp = client.post(
        "/api/v1/inference/segment",
        json={
            "patient_id": "patient001",
            "slice_index": 4,
            "checkpoint_id": "canonical_50epoch",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    meta = data["metadata"]
    assert meta["patient_id"] == "patient001"
    assert meta["slice_index"] == 4
    assert meta["checkpoint_id"] == "canonical_50epoch"
    assert meta["checkpoint_sha256"] == "a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38"
    assert meta["applied_corruption"]["family"] == "clean"
    assert meta["registered_experiment_arm"] == "unmitigated"
    assert meta["inference_duration_ms"] > 0.0

    # Geometry checks: patient001 is 216x256 -> padded symmetrically to 256x256
    geom = data["geometry"]
    assert geom["original_height"] == 216
    assert geom["original_width"] == 256
    assert geom["model_height"] == 256
    assert geom["model_width"] == 256
    assert geom["was_padded"] is True
    assert geom["was_cropped"] is False

    # Model-space mask validation
    ms_mask = data["model_space_mask"]
    assert ms_mask["height"] == 256
    assert ms_mask["width"] == 256
    arr_ms = _decode_png_array(ms_mask["data_base64"])
    assert arr_ms.shape == (256, 256)
    assert set(np.unique(arr_ms).tolist()).issubset({0, 1, 2, 3})

    # Display-space mask and overlay dimensions match original slice dimensions
    ds_mask = data["display_space_mask"]
    ds_overlay = data["display_space_overlay"]
    assert ds_mask["height"] == 216
    assert ds_mask["width"] == 256
    assert ds_overlay["height"] == 216
    assert ds_overlay["width"] == 256

    arr_ds = _decode_png_array(ds_mask["data_base64"])
    rgba_ds = _decode_png_array(ds_overlay["png_base64"])
    assert arr_ds.shape == (216, 256)
    assert rgba_ds.shape == (216, 256, 4)
    # Background pixels (label 0) must be fully transparent in RGBA overlay
    assert np.all(rgba_ds[arr_ds == 0, 3] == 0)
    assert np.all(rgba_ds[arr_ds > 0, 3] > 0)

    # Ground-truth metrics match Phase 5 slice_level_results.csv for patient001_slice04
    assert data["ground_truth_available"] is True
    metrics = data["metrics"]
    assert metrics["aggregation_level"] == "single_2d_slice"
    assert abs(metrics["mean_dice_foreground"] - 0.9102) < 1e-3
    assert abs(metrics["dice_per_class"]["rv"] - 0.9155) < 1e-3
    assert abs(metrics["dice_per_class"]["myo"] - 0.8435) < 1e-3
    assert abs(metrics["dice_per_class"]["lv"] - 0.9716) < 1e-3


def test_cropped_patient_geometry_reported_honestly(client: TestClient) -> None:
    """Verifies larger slice (patient038: 276x384) records cropping and keeps model/display space separate."""
    resp = client.post(
        "/api/v1/inference/segment",
        json={"patient_id": "patient038", "slice_index": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    geom = data["geometry"]
    assert geom["original_height"] == 276
    assert geom["original_width"] == 384
    assert geom["was_cropped"] is True
    assert geom["cropped_regions_unrecoverable"] is True
    assert geom["crop_bounds"] is not None
    assert data["model_space_mask"]["height"] == 256
    assert data["display_space_mask"]["height"] == 276
    assert data["display_space_mask"]["width"] == 384


@pytest.mark.parametrize(
    "corruption,mitigation,expected_family,expected_arm",
    [
        ({"type": "noise", "severity": 0.05}, None, "gaussian_noise", "unmitigated"),
        ({"type": "gaussian_blur", "severity": "moderate"}, None, "gaussian_blur", "unmitigated"),
        ({"type": "contrast_reduction", "severity": 0.25}, None, "contrast_reduction", "unmitigated"),
        ({"type": "contrast_reduction", "severity": "severe"}, {"clahe": True}, "contrast_reduction", "clahe_alone"),
        ({"type": "gaussian_noise", "severity": "severe"}, {"median": True}, "gaussian_noise", "median_alone"),
        ({"type": "gaussian_blur", "severity": "severe"}, {"morphology": True}, "gaussian_blur", "postproc_alone"),
        ({"type": "contrast_reduction", "severity": "severe"}, {"clahe": True, "morphology": True}, "contrast_reduction", "combined_contrast"),
        ({"type": "gaussian_noise", "severity": "moderate"}, {"median": True, "morphology": True}, "gaussian_noise", "combined_noise"),
    ],
)
def test_corruptions_and_mitigations_supported(
    client: TestClient,
    corruption: dict,
    mitigation: dict,
    expected_family: str,
    expected_arm: str,
) -> None:
    """Verifies noise, blur, contrast, CLAHE, median, morphology, and supported combinations."""
    payload = {
        "patient_id": "patient001",
        "slice_index": 4,
        "checkpoint_id": "canonical_50epoch",
        "corruption": corruption,
    }
    if mitigation is not None:
        payload["mitigation"] = mitigation

    resp = client.post("/api/v1/inference/segment", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["applied_corruption"]["family"] == expected_family
    assert data["metadata"]["registered_experiment_arm"] == expected_arm
    arr = _decode_png_array(data["model_space_mask"]["data_base64"])
    assert set(np.unique(arr).tolist()).issubset({0, 1, 2, 3})


def test_invalid_inputs_and_unsupported_combinations_rejected(client: TestClient) -> None:
    """Verifies invalid severity, unsupported mitigation combo, bad patient/slice, and bad checkpoint are rejected."""
    # 1. Invalid numeric severity
    r1 = client.post(
        "/api/v1/inference/segment",
        json={"patient_id": "patient001", "slice_index": 4, "corruption": {"type": "noise", "severity": 0.99}},
    )
    assert r1.status_code == 422
    assert r1.json()["error"]["code"] == "UNSUPPORTED_SEVERITY"

    # 2. Invalid corruption type
    r2 = client.post(
        "/api/v1/inference/segment",
        json={"patient_id": "patient001", "slice_index": 4, "corruption": {"type": "motion", "severity": "mild"}},
    )
    assert r2.status_code == 422
    assert r2.json()["error"]["code"] == "UNSUPPORTED_CORRUPTION"

    # 3. Unsupported dual preprocessing (CLAHE + Median)
    r3 = client.post(
        "/api/v1/inference/segment",
        json={
            "patient_id": "patient001",
            "slice_index": 4,
            "mitigation": {"clahe": True, "median": True, "morphology": False},
        },
    )
    assert r3.status_code == 422
    assert r3.json()["error"]["code"] == "UNSUPPORTED_MITIGATION_COMBINATION"

    # 4. Non-test patient (patient002 is in train split)
    r4 = client.post(
        "/api/v1/inference/segment",
        json={"patient_id": "patient002", "slice_index": 0},
    )
    assert r4.status_code == 404
    assert r4.json()["error"]["code"] == "PATIENT_NOT_FOUND"

    # 5. Out-of-range slice index
    r5 = client.post(
        "/api/v1/inference/segment",
        json={"patient_id": "patient001", "slice_index": 99},
    )
    assert r5.status_code == 404
    assert r5.json()["error"]["code"] == "SLICE_NOT_FOUND"

    # 6. Unregistered checkpoint ID
    r6 = client.post(
        "/api/v1/inference/segment",
        json={"patient_id": "patient001", "slice_index": 4, "checkpoint_id": "arbitrary_evil.pt"},
    )
    assert r6.status_code == 400
    assert r6.json()["error"]["code"] == "INVALID_CHECKPOINT_ID"


def test_model_not_reloaded_across_requests(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies singleton model session is reused across requests without reloading checkpoint."""
    calls = 0

    def fail_if_reloaded(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Model checkpoint must not be reloaded per request")

    monkeypatch.setattr(ckpt_utils, "load_verified_model", fail_if_reloaded)

    for s_idx in (3, 4):
        resp = client.post(
            "/api/v1/inference/segment",
            json={"patient_id": "patient001", "slice_index": s_idx},
        )
        assert resp.status_code == 200
    assert calls == 0


def test_missing_ground_truth_returns_no_fake_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies ground_truth_available=false and metrics=null when GT mask file is absent."""
    # Build isolated minimal workspace copy containing image volume ONLY (no _gt file)
    repo_root = Path(__file__).resolve().parent.parent.parent
    (tmp_path / "configs").mkdir()
    shutil.copy2(repo_root / "configs" / "baseline_unet.yaml", tmp_path / "configs" / "baseline_unet.yaml")

    ckpt_dest = tmp_path / "checkpoints" / "50epochs"
    ckpt_dest.mkdir(parents=True)
    shutil.copy2(
        repo_root / "checkpoints" / "50epochs" / "best_model_50epochs.pt",
        ckpt_dest / "best_model_50epochs.pt",
    )

    raw_p = tmp_path / "data" / "raw" / "ACDC" / "training" / "patient001"
    raw_p.mkdir(parents=True)
    shutil.copy2(
        repo_root / "data" / "raw" / "ACDC" / "training" / "patient001" / "patient001_frame01.nii.gz",
        raw_p / "patient001_frame01.nii.gz",
    )

    manifest_data = json.loads((repo_root / "data" / "processed" / "acdc_slice_manifest.json").read_text(encoding="utf-8"))
    p1_slices = [s for s in manifest_data["slices"] if s["patient_id"] == "patient001"]
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "acdc_slice_manifest.json").write_text(
        json.dumps({"slices": p1_slices}),
        encoding="utf-8",
    )

    monkeypatch.setenv("ACDC_RAW_DIR", str(tmp_path / "data" / "raw" / "ACDC"))
    isolated_app = create_app(paths=build_research_paths(project_root=tmp_path))
    with TestClient(isolated_app) as iso_client:
        resp = iso_client.post(
            "/api/v1/inference/segment",
            json={"patient_id": "patient001", "slice_index": 4},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ground_truth_available"] is False
        assert body["ground_truth_model_space_mask"] is None
        assert body["ground_truth_model_space_overlay"] is None
        assert body["metrics"] is None
