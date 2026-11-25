"""Pydantic schemas for synchronous segmentation and stress-test inference."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class CorruptionSpec(BaseModel):
    """Controlled image-quality degradation configuration."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        default="clean",
        description="Corruption family: 'clean', 'gaussian_noise'/'noise', 'gaussian_blur'/'blur', or 'contrast_reduction'/'contrast'.",
    )
    severity: Optional[Union[str, float]] = Field(
        default=None,
        description="Registered severity tier ('mild', 'moderate', 'severe') or exact registered numeric parameter.",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional non-negative seed for deterministic Gaussian noise.",
    )


class MitigationSpec(BaseModel):
    """Inference-time preprocessing and morphological post-processing configuration."""

    model_config = ConfigDict(extra="forbid")

    clahe: bool = Field(default=False, description="Apply locked CLAHE (clip=0.01, grid=16x16, re-standardize=True).")
    median: bool = Field(default=False, description="Apply locked 2D median filter (size=3, re-standardize=False).")
    morphology: bool = Field(default=False, description="Apply anatomical morphological post-processing.")


class SegmentRequest(BaseModel):
    """Request body for POST /api/v1/inference/segment."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(..., min_length=1, description="Held-out test patient identifier (e.g. 'patient001').")
    slice_index: int = Field(..., ge=0, description="Zero-based ED slice index for the selected patient.")
    checkpoint_id: str = Field(
        default="canonical_50epoch",
        min_length=1,
        description="Registered checkpoint identifier from CHECKPOINT_REGISTRY.",
    )
    corruption: Optional[CorruptionSpec] = Field(default=None)
    mitigation: Optional[MitigationSpec] = Field(default=None)


class AppliedCorruptionInfo(BaseModel):
    """Resolved corruption metadata applied during inference."""

    model_config = ConfigDict(extra="forbid")

    family: str
    severity: str
    parameter_value: float
    units: str
    seed: Optional[int] = None


class AppliedMitigationInfo(BaseModel):
    """Resolved mitigation parameters applied during inference."""

    model_config = ConfigDict(extra="forbid")

    preprocessing_mode: str
    clahe: bool
    median: bool
    morphology: bool
    parameters: Dict[str, object]


class CategoricalMaskPayload(BaseModel):
    """Compact lossless PNG encoding of integer class labels {0, 1, 2, 3}."""

    model_config = ConfigDict(extra="forbid")

    space: str
    height: int
    width: int
    encoding: str
    data_base64: str
    valid_labels: List[int]
    pixel_counts_per_class: Dict[str, int]


class OverlayPayload(BaseModel):
    """Transparent RGBA PNG overlay with class-consistent anatomical colors."""

    model_config = ConfigDict(extra="forbid")

    space: str
    height: int
    width: int
    format: str
    png_base64: str
    color_legend: Dict[str, Dict[str, object]]


class ImagePreviewPayload(BaseModel):
    """8-bit grayscale PNG preview of the MRI slice for visual inspection."""

    model_config = ConfigDict(extra="forbid")

    space: str
    height: int
    width: int
    format: str
    png_base64: str


class GeometryMetadata(BaseModel):
    """Explicit spatial transformation record between display-space and model-space."""

    model_config = ConfigDict(extra="forbid")

    original_height: int
    original_width: int
    model_height: int
    model_width: int
    voxel_spacing: Optional[List[float]] = None
    pad_widths: Dict[str, int]
    crop_bounds: Optional[Dict[str, int]] = None
    was_padded: bool
    was_cropped: bool
    cropped_regions_unrecoverable: bool
    affine_preserved: bool
    orientation_note: str


class SegmentationMetrics(BaseModel):
    """Per-class and foreground segmentation metrics computed on standardized model-space slice."""

    model_config = ConfigDict(extra="forbid")

    aggregation_level: str
    sample_description: str
    evaluation_space: str
    dice_per_class: Dict[str, float]
    iou_per_class: Dict[str, float]
    mean_dice_foreground: float
    mean_iou_foreground: float
    mean_dice_all: float
    mean_iou_all: float


class InferenceMetadata(BaseModel):
    """Provenance and execution telemetry for a synchronous inference run."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    slice_index: int
    ed_frame: int
    pathology_group: str
    checkpoint_id: str
    checkpoint_sha256: str
    device: str
    inference_duration_ms: float
    applied_corruption: AppliedCorruptionInfo
    applied_mitigation: AppliedMitigationInfo
    registered_experiment_arm: str
    execution_order: List[str]


class SegmentResponse(BaseModel):
    """Complete synchronous segmentation and stress-test response."""

    model_config = ConfigDict(extra="forbid")

    metadata: InferenceMetadata
    geometry: GeometryMetadata
    input_preview_model_space: ImagePreviewPayload
    input_preview_display_space: ImagePreviewPayload
    model_space_mask: CategoricalMaskPayload
    display_space_mask: CategoricalMaskPayload
    model_space_overlay: OverlayPayload
    display_space_overlay: OverlayPayload
    ground_truth_available: bool
    ground_truth_model_space_mask: Optional[CategoricalMaskPayload] = None
    ground_truth_model_space_overlay: Optional[OverlayPayload] = None
    metrics: Optional[SegmentationMetrics] = None
