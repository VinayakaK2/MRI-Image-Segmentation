"""Manifest generation, split compilation, and dataset-level integrity reporting."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.data.discovery import discover_all_training_patients, PatientEDRecord
from src.data.patient_split import create_stratified_patient_splits, save_patient_splits
from src.data.slice_extractor import extract_slice_metadata_from_volumes, SliceRecord
from src.utils.config_parser import resolve_dataset_raw_dir, get_project_root


def build_acdc_pipeline_artifacts(
    raw_dataset_dir: Optional[Path] = None,
    output_splits_path: Optional[Path] = None,
    output_manifest_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Discovers patients, creates stratified splits, extracts slice manifests, and writes audit reports."""
    project_root = get_project_root()
    raw_dir = raw_dataset_dir if raw_dataset_dir is not None else resolve_dataset_raw_dir()
    training_dir = raw_dir / "training"

    splits_path = (
        output_splits_path
        if output_splits_path is not None
        else project_root / "data" / "splits" / "acdc_patient_splits.json"
    )
    manifest_path = (
        output_manifest_path
        if output_manifest_path is not None
        else project_root / "data" / "processed" / "acdc_slice_manifest.json"
    )
    report_path = (
        output_report_path
        if output_report_path is not None
        else project_root / "reports" / "acdc_dataset_integrity_report.json"
    )

    # 1. Discover all training patients
    patient_records: List[PatientEDRecord] = discover_all_training_patients(training_dir)
    patient_to_group = {p.patient_id: p.group for p in patient_records}

    # 2. Stratified Patient Splitting
    splits = create_stratified_patient_splits(patient_to_group=patient_to_group, seed=seed)
    patient_to_split: Dict[str, str] = {}
    for s_name, p_list in splits.items():
        for p_id in p_list:
            patient_to_split[p_id] = s_name

    # Save splits
    save_patient_splits(
        splits=splits,
        output_path=splits_path,
        metadata={"seed": seed, "total_patients": len(patient_records)},
    )

    # 3. Extract 2D Slice Metadata
    all_slices: List[Dict[str, Any]] = []
    split_slice_counts = {"train": 0, "val": 0, "test": 0}
    split_fg_counts = {"train": 0, "val": 0, "test": 0}

    for p in patient_records:
        p_split = patient_to_split[p.patient_id]
        slice_recs = extract_slice_metadata_from_volumes(
            patient_id=p.patient_id,
            group=p.group,
            ed_frame=p.ed_frame_number,
            image_path=p.image_path,
            mask_path=p.mask_path,
            base_dir=raw_dir,
        )
        for s in slice_recs:
            s_dict = s.to_dict()
            s_dict["split"] = p_split
            all_slices.append(s_dict)
            split_slice_counts[p_split] += 1
            if s.has_foreground:
                split_fg_counts[p_split] += 1

    # 4. Generate Machine-Readable Slice Manifest
    slice_manifest_data = {
        "version": "1.0",
        "description": "ACDC 2D ED-phase slice manifest with patient-level splits",
        "seed": seed,
        "total_patients": len(patient_records),
        "total_slices": len(all_slices),
        "split_counts": {
            "patients": {k: len(v) for k, v in splits.items()},
            "slices": split_slice_counts,
            "foreground_slices": split_fg_counts,
        },
        "slices": all_slices,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(slice_manifest_data, f, indent=2)

    # 5. Generate Integrity & Audit Report
    integrity_report_data = {
        "status": "PASS",
        "cohort": "ACDC Primary Training Cohort (ED Phase)",
        "random_seed": seed,
        "splits_file": str(splits_path.relative_to(project_root)).replace("\\", "/"),
        "slice_manifest_file": str(manifest_path.relative_to(project_root)).replace("\\", "/"),
        "summary": {
            "total_patients": len(patient_records),
            "total_ed_volumes": len(patient_records),
            "total_2d_slices": len(all_slices),
            "patient_distribution": {k: len(v) for k, v in splits.items()},
            "slice_distribution": split_slice_counts,
            "foreground_slice_distribution": split_fg_counts,
            "pathology_distribution_by_split": _compute_pathology_by_split(splits, patient_to_group),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(integrity_report_data, f, indent=2)

    return integrity_report_data


def _compute_pathology_by_split(
    splits: Dict[str, List[str]],
    patient_to_group: Dict[str, str],
) -> Dict[str, Dict[str, int]]:
    """Tallies patient count per pathology group within each partition."""
    dist: Dict[str, Dict[str, int]] = {}
    for s_name, p_list in splits.items():
        dist[s_name] = {}
        for p_id in p_list:
            grp = patient_to_group[p_id]
            dist[s_name][grp] = dist[s_name].get(grp, 0) + 1
    return dist
