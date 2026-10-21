"""Dataset discovery, Info.cfg metadata extraction, and NIfTI volume validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import nibabel as nib
import numpy as np


VALID_SEGMENTATION_LABELS: Set[int] = {0, 1, 2, 3}
EXPECTED_NUM_CLASSES: int = 4


@dataclass(frozen=True)
class PatientEDRecord:
    """Immutable data record representing an individual patient's verified ED-phase data."""
    patient_id: str
    group: str
    ed_frame_number: int
    ed_frame_index_0based: int
    image_path: Path
    mask_path: Path
    spatial_shape: Tuple[int, int, int]
    voxel_spacing: Tuple[float, float, float]
    labels_present: Tuple[int, ...]
    num_slices: int


def parse_info_cfg(info_file_path: Path) -> Dict[str, str]:
    """Parses patient metadata from an ACDC Info.cfg text file into key-value pairs."""
    info_file_path = Path(info_file_path)
    if not info_file_path.exists():
        raise FileNotFoundError(f"Info.cfg file not found at: {info_file_path}")

    metadata: Dict[str, str] = {}
    with open(info_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            if ":" not in cleaned:
                raise ValueError(f"Malformed line {line_num} in {info_file_path}: '{line}'")
            key, val = cleaned.split(":", 1)
            metadata[key.strip()] = val.strip()

    required_keys = {"ED", "Group"}
    missing = required_keys - set(metadata.keys())
    if missing:
        raise KeyError(f"Info.cfg at {info_file_path} missing mandatory fields: {missing}")

    return metadata


def parse_ed_frame_indices(raw_ed_value: str) -> Tuple[int, int]:
    """Converts 1-based ED frame string from Info.cfg into (1-based frame_num, 0-based array_idx)."""
    try:
        frame_1based = int(raw_ed_value)
    except ValueError as err:
        raise ValueError(f"ED frame value must be integer, got: '{raw_ed_value}'") from err

    if frame_1based < 1:
        raise ValueError(f"ED frame number must be >= 1 (1-based), got: {frame_1based}")

    frame_0based = frame_1based - 1
    return frame_1based, frame_0based


def validate_nifti_ed_pair(
    image_path: Path,
    mask_path: Path,
    patient_id: str,
) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], Tuple[int, ...]]:
    """Loads and validates spatial shape, voxel spacing, and label integrity of an MRI/mask pair."""
    if not image_path.exists():
        raise FileNotFoundError(f"Patient {patient_id} MRI file missing: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Patient {patient_id} Ground Truth mask missing: {mask_path}")

    try:
        img_nii = nib.load(str(image_path))
        mask_nii = nib.load(str(mask_path))
    except Exception as err:
        raise IOError(f"Failed to load NIfTI volumes for {patient_id}: {err}") from err

    img_shape = tuple(int(s) for s in img_nii.shape)
    mask_shape = tuple(int(s) for s in mask_nii.shape)

    if len(img_shape) != 3:
        raise ValueError(f"Expected 3D spatial volume for {patient_id}, got shape: {img_shape}")
    if img_shape != mask_shape:
        raise ValueError(
            f"Shape mismatch for {patient_id}: image={img_shape} vs mask={mask_shape}"
        )

    # Validate voxel zoom / spacing
    zooms = tuple(float(z) for z in img_nii.header.get_zooms()[:3])

    # Validate mask integer labels
    mask_data = mask_nii.get_fdata()
    unique_labels = sorted([int(x) for x in np.unique(mask_data)])
    invalid_labels = set(unique_labels) - VALID_SEGMENTATION_LABELS
    if invalid_labels:
        raise ValueError(
            f"Unexpected mask label(s) {invalid_labels} in {patient_id}. Allowed: {VALID_SEGMENTATION_LABELS}"
        )

    return img_shape, zooms, tuple(unique_labels)


def discover_single_patient_ed(patient_dir: Path) -> PatientEDRecord:
    """Discovers and validates the ED frame and segmentation mask for a single patient directory."""
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name
    info_path = patient_dir / "Info.cfg"
    metadata = parse_info_cfg(info_path)

    group = metadata["Group"]
    frame_1based, frame_0based = parse_ed_frame_indices(metadata["ED"])

    image_filename = f"{patient_id}_frame{frame_1based:02d}.nii.gz"
    mask_filename = f"{patient_id}_frame{frame_1based:02d}_gt.nii.gz"

    image_path = patient_dir / image_filename
    mask_path = patient_dir / mask_filename

    shape, spacing, labels = validate_nifti_ed_pair(image_path, mask_path, patient_id)
    num_slices = shape[2]  # Spatial axis 2 is slice depth in short-axis cine MRI

    return PatientEDRecord(
        patient_id=patient_id,
        group=group,
        ed_frame_number=frame_1based,
        ed_frame_index_0based=frame_0based,
        image_path=image_path,
        mask_path=mask_path,
        spatial_shape=shape,
        voxel_spacing=spacing,
        labels_present=labels,
        num_slices=num_slices,
    )


def discover_all_training_patients(training_dir: Path) -> List[PatientEDRecord]:
    """Discovers and validates all primary training cohort patients."""
    training_dir = Path(training_dir)
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory does not exist: {training_dir}")

    patient_dirs = sorted(
        [d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith("patient")]
    )
    records: List[PatientEDRecord] = []
    errors: List[str] = []

    for p_dir in patient_dirs:
        try:
            record = discover_single_patient_ed(p_dir)
            records.append(record)
        except Exception as err:
            errors.append(f"{p_dir.name}: {err}")

    if errors:
        error_msg = "\n".join(errors)
        raise RuntimeError(f"Errors encountered during training patient discovery:\n{error_msg}")

    return records
