"""Data loading, patient-level splitting, and dataset handling modules."""

from src.data.discovery import (
    PatientEDRecord,
    discover_all_training_patients,
    discover_single_patient_ed,
    parse_ed_frame_indices,
    parse_info_cfg,
)
from src.data.patient_split import (
    create_stratified_patient_splits,
    load_patient_splits,
    save_patient_splits,
    verify_patient_splits_integrity,
)
from src.data.slice_extractor import (
    SliceRecord,
    extract_slice_metadata_from_volumes,
    load_raw_2d_slice_pair,
)
from src.data.dataset import ACDCSliceDataset, acdc_collate_fn
from src.data.manifest import build_acdc_pipeline_artifacts

__all__ = [
    "PatientEDRecord",
    "discover_all_training_patients",
    "discover_single_patient_ed",
    "parse_ed_frame_indices",
    "parse_info_cfg",
    "create_stratified_patient_splits",
    "load_patient_splits",
    "save_patient_splits",
    "verify_patient_splits_integrity",
    "SliceRecord",
    "extract_slice_metadata_from_volumes",
    "load_raw_2d_slice_pair",
    "ACDCSliceDataset",
    "acdc_collate_fn",
    "build_acdc_pipeline_artifacts",
]
