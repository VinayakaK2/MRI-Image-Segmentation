"""Service for reading and validating held-out ACDC test cohort patients and ED slices."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api.config import ResearchPaths, build_research_paths
from api.schemas.patient import PatientsResponse, SliceMetadataItem, TestPatientItem
from api.utils.validation import ApiError, resolve_safe_path_within, validate_patient_id_format
from src.utils.config_parser import load_yaml_config, resolve_dataset_raw_dir


class PatientService:
    """Provides read-only access strictly to held-out test split patients and ED slices."""

    def __init__(self, paths: Optional[ResearchPaths] = None) -> None:
        """Initializes service with canonical manifest and dataset path configuration."""
        self.paths = paths or build_research_paths()
        self._test_records_cache: Optional[List[Dict[str, Any]]] = None
        self._raw_dir_cache: Optional[Path] = None

    def _get_raw_dir(self) -> Optional[Path]:
        """Resolves the raw ACDC dataset root using the existing research helper."""
        if self._raw_dir_cache is not None:
            return self._raw_dir_cache
        try:
            cfg = load_yaml_config(str(self.paths.config_path))
            self._raw_dir_cache = resolve_dataset_raw_dir(cfg).resolve()
        except FileNotFoundError:
            self._raw_dir_cache = None
        return self._raw_dir_cache

    def _load_test_slice_records(self) -> List[Dict[str, Any]]:
        """Loads slice records from acdc_slice_manifest.json and filters strictly to split == 'test'."""
        if self._test_records_cache is not None:
            return self._test_records_cache

        manifest_path = self.paths.slice_manifest_path
        if not manifest_path.is_file():
            raise ApiError(
                status_code=503,
                code="MANIFEST_NOT_FOUND",
                message="ACDC slice manifest is unavailable.",
            )
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            raise ApiError(
                status_code=500,
                code="MANIFEST_MALFORMED",
                message="ACDC slice manifest could not be parsed.",
            ) from exc

        slices = payload.get("slices")
        if not isinstance(slices, list):
            raise ApiError(
                status_code=500,
                code="MANIFEST_MALFORMED",
                message="ACDC slice manifest is missing the 'slices' list.",
            )

        # Strictly enforce test-only boundary: never expose train or val records
        test_slices = [
            rec for rec in slices
            if isinstance(rec, dict) and str(rec.get("split", "")).strip().lower() == "test"
        ]
        self._test_records_cache = test_slices
        return test_slices

    def _check_file_availability(self, rel_path: str) -> bool:
        """Safely checks whether a dataset-relative NIfTI file exists inside raw_dir."""
        raw_dir = self._get_raw_dir()
        if raw_dir is None or not rel_path:
            return False
        try:
            resolved = resolve_safe_path_within(raw_dir, rel_path)
            return resolved.is_file()
        except ApiError:
            return False

    def list_test_patients(self, patient_id: Optional[str] = None) -> PatientsResponse:
        """Returns all 15 held-out test patients or a single verified test patient."""
        test_records = self._load_test_slice_records()

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in test_records:
            grouped[str(rec["patient_id"])].append(rec)

        if patient_id is not None:
            clean_id = validate_patient_id_format(patient_id)
            if clean_id not in grouped:
                raise ApiError(
                    status_code=404,
                    code="PATIENT_NOT_FOUND",
                    message=f"Test patient '{clean_id}' was not found in the held-out test cohort.",
                )
            selected_ids = [clean_id]
        else:
            selected_ids = sorted(grouped.keys())

        patient_items: List[TestPatientItem] = []
        total_slices = 0

        for pid in selected_ids:
            recs = sorted(grouped[pid], key=lambda r: int(r["slice_idx"]))
            first = recs[0]
            img_avail = self._check_file_availability(str(first.get("image_rel_path", "")))
            mask_avail = self._check_file_availability(str(first.get("mask_rel_path", "")))

            slice_items: List[SliceMetadataItem] = []
            for r in recs:
                spacing = [float(x) for x in r["voxel_spacing"]] if r.get("voxel_spacing") else None
                slice_items.append(
                    SliceMetadataItem(
                        slice_index=int(r["slice_idx"]),
                        total_slices=int(r["total_slices"]),
                        raw_slice_shape=[int(x) for x in r["raw_slice_shape"]],
                        voxel_spacing=spacing,
                        image_available=img_avail,
                        mask_available=mask_avail,
                        has_foreground=bool(r.get("has_foreground", False)),
                        foreground_labels=[int(x) for x in r.get("foreground_labels", [])],
                        pixel_counts={str(k): int(v) for k, v in r.get("pixel_counts", {}).items()},
                    )
                )

            first_spacing = [float(x) for x in first["voxel_spacing"]] if first.get("voxel_spacing") else None
            patient_items.append(
                TestPatientItem(
                    patient_id=pid,
                    pathology_group=str(first.get("group", "UNKNOWN")),
                    ed_frame=int(first["ed_frame"]),
                    split="test",
                    slice_count=len(slice_items),
                    available_slice_indices=[s.slice_index for s in slice_items],
                    raw_slice_shape=[int(x) for x in first["raw_slice_shape"]],
                    voxel_spacing=first_spacing,
                    image_available=img_avail,
                    mask_available=mask_avail,
                    slices=slice_items,
                )
            )
            total_slices += len(slice_items)

        return PatientsResponse(
            split="test",
            phase="ED",
            patient_count=len(patient_items),
            total_slices=total_slices,
            patients=patient_items,
        )

    def get_test_slice_context(self, patient_id: str, slice_index: int) -> Tuple[Dict[str, Any], int, Path]:
        """Resolves a test slice record, its manifest test-loader index, and safe dataset raw_dir."""
        clean_id = validate_patient_id_format(patient_id)
        test_records = self._load_test_slice_records()

        patient_exists = False
        for idx, rec in enumerate(test_records):
            if str(rec["patient_id"]) == clean_id:
                patient_exists = True
                if int(rec["slice_idx"]) == slice_index:
                    raw_dir = self._get_raw_dir()
                    if raw_dir is None:
                        raise ApiError(
                            status_code=503,
                            code="DATASET_UNAVAILABLE",
                            message="ACDC raw NIfTI dataset directory is unavailable.",
                        )
                    return rec, idx, raw_dir

        if not patient_exists:
            raise ApiError(
                status_code=404,
                code="PATIENT_NOT_FOUND",
                message=f"Test patient '{clean_id}' was not found in the held-out test cohort.",
            )

        raise ApiError(
            status_code=404,
            code="SLICE_NOT_FOUND",
            message=f"Slice index {slice_index} does not exist for test patient '{clean_id}'.",
        )
