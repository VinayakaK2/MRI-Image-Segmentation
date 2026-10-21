"""3D spatial volume to 2D slice extraction, pairing, and metadata indexing."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import nibabel as nib
import numpy as np


# Short-axis cine MRI slice depth dimension index in NIfTI format
SLICE_DEPTH_AXIS: int = 2


@dataclass(frozen=True)
class SliceRecord:
    """Metadata and filesystem reference for an individual 2D MRI and mask slice pair."""
    patient_id: str
    group: str
    ed_frame: int
    slice_idx: int
    total_slices: int
    image_rel_path: str
    mask_rel_path: str
    raw_slice_shape: Tuple[int, int]
    voxel_spacing: Tuple[float, float, float]
    has_foreground: bool
    foreground_labels: Tuple[int, ...]
    pixel_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Converts slice record to a serializable dictionary."""
        d = asdict(self)
        d["raw_slice_shape"] = list(self.raw_slice_shape)
        d["voxel_spacing"] = list(self.voxel_spacing)
        d["foreground_labels"] = list(self.foreground_labels)
        return d


def extract_slice_metadata_from_volumes(
    patient_id: str,
    group: str,
    ed_frame: int,
    image_path: Path,
    mask_path: Path,
    base_dir: Path,
) -> List[SliceRecord]:
    """Inspects a patient's 3D NIfTI volumes and generates slice metadata records without modifying data."""
    img_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))

    mask_data = mask_nii.get_fdata().astype(np.int64)
    num_slices = mask_data.shape[SLICE_DEPTH_AXIS]
    raw_h, raw_w = mask_data.shape[0], mask_data.shape[1]
    zooms = tuple(float(z) for z in img_nii.header.get_zooms()[:3])

    # Convert paths to relative strings for portability across environments
    img_rel = str(image_path.resolve().relative_to(base_dir.resolve())).replace("\\", "/")
    mask_rel = str(mask_path.resolve().relative_to(base_dir.resolve())).replace("\\", "/")

    records: List[SliceRecord] = []
    for s_idx in range(num_slices):
        mask_slice = mask_data[:, :, s_idx]
        unique_labels = sorted([int(x) for x in np.unique(mask_slice)])
        fg_labels = tuple(lbl for lbl in unique_labels if lbl != 0)
        has_fg = len(fg_labels) > 0

        # Tally pixel distribution
        counts: Dict[str, int] = {}
        for c in range(4):
            counts[str(c)] = int(np.sum(mask_slice == c))

        rec = SliceRecord(
            patient_id=patient_id,
            group=group,
            ed_frame=ed_frame,
            slice_idx=s_idx,
            total_slices=num_slices,
            image_rel_path=img_rel,
            mask_rel_path=mask_rel,
            raw_slice_shape=(raw_h, raw_w),
            voxel_spacing=zooms,
            has_foreground=has_fg,
            foreground_labels=fg_labels,
            pixel_counts=counts,
        )
        records.append(rec)

    return records


def load_raw_2d_slice_pair(
    image_full_path: Path,
    mask_full_path: Path,
    slice_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a specific 2D slice pair from NIfTI volumes along the slice depth axis."""
    img_nii = nib.load(str(image_full_path))
    mask_nii = nib.load(str(mask_full_path))

    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    total_slices = img_data.shape[SLICE_DEPTH_AXIS]
    if slice_idx < 0 or slice_idx >= total_slices:
        raise IndexError(
            f"Requested slice_idx {slice_idx} out of range [0, {total_slices - 1}] in {image_full_path}"
        )

    # Short-axis slice extraction along axis 2
    img_slice = img_data[:, :, slice_idx].astype(np.float32)
    mask_slice = mask_data[:, :, slice_idx].astype(np.int64)

    return img_slice, mask_slice
