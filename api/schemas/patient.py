"""Pydantic schemas for the held-out test patients and slices endpoint."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class SliceMetadataItem(BaseModel):
    """Metadata for a single 2D End-Diastole slice from the test manifest."""

    model_config = ConfigDict(extra="forbid")

    slice_index: int
    total_slices: int
    raw_slice_shape: List[int]
    voxel_spacing: Optional[List[float]] = None
    image_available: bool
    mask_available: bool
    has_foreground: bool
    foreground_labels: List[int]
    pixel_counts: Dict[str, int]


class TestPatientItem(BaseModel):
    """Patient-level summary and slice list restricted strictly to the test split."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    pathology_group: str
    ed_frame: int
    split: str
    slice_count: int
    available_slice_indices: List[int]
    raw_slice_shape: List[int]
    voxel_spacing: Optional[List[float]] = None
    image_available: bool
    mask_available: bool
    slices: List[SliceMetadataItem]


class PatientsResponse(BaseModel):
    """Response for GET /api/v1/patients covering the 15-patient held-out test cohort."""

    model_config = ConfigDict(extra="forbid")

    split: str
    phase: str
    patient_count: int
    total_slices: int
    patients: List[TestPatientItem]
