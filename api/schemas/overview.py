"""Pydantic schemas for the research overview and model specification endpoint."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class ModelSpecification(BaseModel):
    """Architectural specification for LightweightUNet2D."""

    model_config = ConfigDict(extra="forbid")

    architecture: str
    parameter_count_total: int
    parameter_count_trainable: int
    in_channels: int
    out_channels: int
    base_channels: int
    depth: int
    dropout: float
    use_batchnorm: bool
    bilinear_upsample: bool
    input_dimensions: List[int]
    output_dimensions: List[int]
    class_mapping: Dict[str, str]


class CheckpointOverview(BaseModel):
    """Verified checkpoint metadata derived from manifests and SHA-256 checks."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    display_name: str
    is_canonical: bool
    expected_sha256: str
    verified_sha256: Optional[str] = None
    integrity_verified: bool
    available: bool
    epochs_completed: int
    best_epoch: Optional[int] = None
    best_val_foreground_dice: Optional[float] = None


class DatasetOverview(BaseModel):
    """Dataset partition and cohort summary derived from dataset manifests."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    evaluated_phase: str
    total_patients: Optional[int] = None
    total_slices: Optional[int] = None
    test_patient_count: int
    test_slice_count: int
    test_foreground_slice_count: Optional[int] = None
    test_patient_ids: List[str]
    pathology_distribution_test: Dict[str, int]


class AggregatedMetricSummary(BaseModel):
    """Explicitly scoped metric record keeping slice-level and patient-level results separate."""

    model_config = ConfigDict(extra="forbid")

    condition_id: str
    aggregation_level: str
    sample_count: int
    sample_unit: str
    mean_dice_foreground: Optional[float] = None
    std_dice_foreground: Optional[float] = None
    mean_iou_foreground: Optional[float] = None
    dice_per_class: Dict[str, float]
    iou_per_class: Optional[Dict[str, float]] = None
    source_artifact: str


class ExperimentSummaryCard(BaseModel):
    """High-level summary of a completed research experiment phase."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    phase: int
    title: str
    available: bool
    checkpoint_sha256: Optional[str] = None
    conditions_count: Optional[int] = None
    hypothesis_families_count: Optional[int] = None
    total_primary_hypotheses: Optional[int] = None
    key_findings: List[str]


class OverviewResponse(BaseModel):
    """Complete dashboard overview and model metadata response."""

    model_config = ConfigDict(extra="forbid")

    project_title: str
    model: ModelSpecification
    checkpoints: List[CheckpointOverview]
    dataset: DatasetOverview
    baseline_metrics: List[AggregatedMetricSummary]
    experiment_summaries: List[ExperimentSummaryCard]
