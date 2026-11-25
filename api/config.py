"""Centralized configuration, registries, and frozen artifact paths for the local FastAPI backend."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.config_parser import get_project_root


# Backend metadata constants
BACKEND_VERSION: str = "0.1.0"
API_PREFIX: str = "/api/v1"
PROJECT_TITLE: str = (
    "Robustness Evaluation of a Lightweight U-Net for Cardiac MRI Segmentation "
    "Under Controlled Image-Quality Variations"
)

# Default local React/Vite development origins (no wildcard public origins allowed)
DEFAULT_CORS_ORIGINS: Tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def get_allowed_cors_origins() -> List[str]:
    """Returns explicitly allowed local frontend origins from env override or safe defaults."""
    raw = os.environ.get("CARDIAC_API_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    # Reject wildcard origins to enforce local-only access policy
    if "*" in origins:
        raise ValueError("Wildcard '*' CORS origin is not permitted for the local research backend.")
    return origins


@dataclass(frozen=True)
class CheckpointSpec:
    """Immutable specification for an explicitly registered model checkpoint."""

    checkpoint_id: str
    display_name: str
    relative_path: str
    expected_sha256: str
    is_canonical: bool
    epochs_completed: int
    best_epoch: int
    description: str


@dataclass(frozen=True)
class ResearchPaths:
    """Resolved filesystem paths to frozen research configs, manifests, and Phase 4/5 artifacts."""

    project_root: Path
    config_path: Path
    slice_manifest_path: Path
    splits_path: Path
    integrity_report_path: Path
    dataset_manifest_path: Path
    baseline_eval_50epochs_path: Path
    training_history_50epochs_path: Path
    phase4_dir: Path
    phase4_results_json: Path
    phase4_results_csv: Path
    phase4_metadata_json: Path
    phase5_dir: Path
    phase5_results_json: Path
    phase5_results_csv: Path
    phase5_primary_hypotheses_csv: Path
    phase5_holm_csv: Path
    phase5_statistical_tests_csv: Path
    phase5_negative_controls_csv: Path
    phase5_patient_level_csv: Path
    phase5_slice_level_csv: Path
    phase5_repro_manifest_json: Path
    phase5_config_snapshot_json: Path
    phase5_audit_json: Path
    phase5_clahe_stage_json: Path


def build_research_paths(project_root: Path | None = None) -> ResearchPaths:
    """Constructs all canonical read-only paths relative to the repository root."""
    root = (project_root or get_project_root()).resolve()
    phase4_dir = root / "experiments" / "phase4_robustness_50epoch_best_epoch44"
    phase5_dir = root / "experiments" / "phase5_validation_locked_mitigation"
    clahe_audit_dir = root / "experiments" / "phase5_clahe_invariance_audit"
    return ResearchPaths(
        project_root=root,
        config_path=root / "configs" / "baseline_unet.yaml",
        slice_manifest_path=root / "data" / "processed" / "acdc_slice_manifest.json",
        splits_path=root / "data" / "splits" / "acdc_patient_splits.json",
        integrity_report_path=root / "reports" / "acdc_dataset_integrity_report.json",
        dataset_manifest_path=root / "reports" / "acdc_dataset_manifest.json",
        baseline_eval_50epochs_path=root / "reports" / "baseline_test_evaluation_50epochs.json",
        training_history_50epochs_path=root / "reports" / "training_history_50epochs.json",
        phase4_dir=phase4_dir,
        phase4_results_json=phase4_dir / "robustness_results.json",
        phase4_results_csv=phase4_dir / "robustness_results.csv",
        phase4_metadata_json=phase4_dir / "experiment_metadata.json",
        phase5_dir=phase5_dir,
        phase5_results_json=phase5_dir / "mitigation_results.json",
        phase5_results_csv=phase5_dir / "mitigation_results.csv",
        phase5_primary_hypotheses_csv=phase5_dir / "primary_hypotheses_results.csv",
        phase5_holm_csv=phase5_dir / "holm_adjusted_results.csv",
        phase5_statistical_tests_csv=phase5_dir / "statistical_tests.csv",
        phase5_negative_controls_csv=phase5_dir / "negative_control_results.csv",
        phase5_patient_level_csv=phase5_dir / "patient_level_results.csv",
        phase5_slice_level_csv=phase5_dir / "slice_level_results.csv",
        phase5_repro_manifest_json=phase5_dir / "reproducibility_manifest.json",
        phase5_config_snapshot_json=phase5_dir / "experiment_config_snapshot.json",
        phase5_audit_json=phase5_dir / "phase5_independent_audit.json",
        phase5_clahe_stage_json=clahe_audit_dir / "preprocessing_stage_analysis.json",
    )


# Explicit allowlist of verified model checkpoints (no arbitrary file path loading)
CANONICAL_CHECKPOINT_ID: str = "canonical_50epoch"
CHECKPOINT_REGISTRY: Dict[str, CheckpointSpec] = {
    "canonical_50epoch": CheckpointSpec(
        checkpoint_id="canonical_50epoch",
        display_name="Canonical 50-Epoch LightweightUNet2D (Best Epoch 44)",
        relative_path="checkpoints/50epochs/best_model_50epochs.pt",
        expected_sha256="a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38",
        is_canonical=True,
        epochs_completed=50,
        best_epoch=44,
        description="Primary 50-epoch trained model selected at best validation Epoch 44.",
    ),
    "frozen_3epoch_baseline": CheckpointSpec(
        checkpoint_id="frozen_3epoch_baseline",
        display_name="Historical 3-Epoch Frozen Baseline (Epoch 3)",
        relative_path="checkpoints/best_model_frozen.pt",
        expected_sha256="e7bfa884bef0f4eafce30cd6948c8d60fc4e7c8c8ca2d0bb0e83f2977fc29c02",
        is_canonical=False,
        epochs_completed=3,
        best_epoch=3,
        description="Historical 3-epoch CPU baseline preserved for reference.",
    ),
}

# Anatomical class mapping (fixed across ACDC and LightweightUNet2D)
CLASS_MAPPING: Dict[int, str] = {
    0: "Background",
    1: "RV",
    2: "MYO",
    3: "LV",
}
VALID_MASK_LABELS: Tuple[int, ...] = (0, 1, 2, 3)

# Consistent RGBA color palette for transparent overlay PNGs (alpha=0 for Background)
OVERLAY_RGBA_PALETTE: Dict[int, Tuple[int, int, int, int]] = {
    0: (0, 0, 0, 0),
    1: (255, 127, 14, 180),    # RV - Orange (#FF7F0E)
    2: (148, 103, 189, 180),   # MYO - Purple (#9467BD)
    3: (214, 39, 40, 180),     # LV - Red (#D62728)
}

OVERLAY_COLOR_LEGEND: Dict[str, Dict[str, object]] = {
    "0": {"label": "Background", "hex": "#00000000", "rgba": [0, 0, 0, 0]},
    "1": {"label": "RV", "hex": "#FF7F0E", "rgba": [255, 127, 14, 180]},
    "2": {"label": "MYO", "hex": "#9467BD", "rgba": [148, 103, 189, 180]},
    "3": {"label": "LV", "hex": "#D62728", "rgba": [214, 39, 40, 180]},
}

# Accepted user-facing aliases mapped strictly to src.corruptions.engine family names
CORRUPTION_TYPE_ALIASES: Dict[str, str] = {
    "clean": "clean",
    "none": "clean",
    "identity": "clean",
    "noise": "gaussian_noise",
    "gaussian_noise": "gaussian_noise",
    "blur": "gaussian_blur",
    "gaussian_blur": "gaussian_blur",
    "contrast": "contrast_reduction",
    "contrast_reduction": "contrast_reduction",
}

# Locked Phase 5 mitigation hyperparameters (matching experiments/phase5_validation_locked_mitigation)
LOCKED_CLAHE_CLIP_LIMIT: float = 0.01
LOCKED_CLAHE_KERNEL_SIZE: Tuple[int, int] = (16, 16)
LOCKED_CLAHE_NBINS: int = 256
LOCKED_CLAHE_RE_STANDARDIZE: bool = True
LOCKED_MEDIAN_FILTER_SIZE: int = 3
LOCKED_MEDIAN_RE_STANDARDIZE: bool = False
LOCKED_MORPHOLOGY_MIN_SIZES: Dict[int, int] = {1: 30, 2: 50, 3: 30}
LOCKED_MORPHOLOGY_MAX_MYO_HOLE_SIZE: int = 16

# Deterministic noise seed offset constants matching scripts/evaluate_robustness.py & evaluate_mitigation_test.py
RESEARCH_BASE_SEED: int = 42
RESEARCH_BATCH_SEED_STRIDE: int = 17

# Registered Phase 5 hypothesis families for /api/v1/experiments
PHASE5_FAMILY_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "family_1_postproc_alone": {
        "family_id": "family_1_postproc_alone",
        "csv_family_name": "Family 1 (Post-Proc, m=10)",
        "arm_key": "postproc_alone",
        "intervention_arm": "Post-Processing Alone (LCC+HoleFill)",
        "hypothesis_count_m": 10,
        "correction_method": "Holm-Bonferroni step-down with monotonicity enforcement",
        "correction_scope": "Family-wise within Family 1 (m=10 conditions)",
    },
    "family_2_clahe_alone": {
        "family_id": "family_2_clahe_alone",
        "csv_family_name": "Family 2 (CLAHE, m=3)",
        "arm_key": "clahe_alone",
        "intervention_arm": "CLAHE Alone (clip=0.01)",
        "hypothesis_count_m": 3,
        "correction_method": "Holm-Bonferroni step-down with monotonicity enforcement",
        "correction_scope": "Family-wise within Family 2 (m=3 contrast conditions)",
    },
    "family_3_clahe_plus_postproc": {
        "family_id": "family_3_clahe_plus_postproc",
        "csv_family_name": "Family 3 (Combined Contrast, m=3)",
        "arm_key": "combined_contrast",
        "intervention_arm": "Combined (CLAHE + Post-Processing)",
        "hypothesis_count_m": 3,
        "correction_method": "Holm-Bonferroni step-down with monotonicity enforcement",
        "correction_scope": "Family-wise within Family 3 (m=3 contrast conditions)",
    },
    "family_4_median_plus_postproc": {
        "family_id": "family_4_median_plus_postproc",
        "csv_family_name": "Family 4 (Combined Noise, m=3)",
        "arm_key": "combined_noise",
        "intervention_arm": "Combined (Median + Post-Processing)",
        "hypothesis_count_m": 3,
        "correction_method": "Holm-Bonferroni step-down with monotonicity enforcement",
        "correction_scope": "Family-wise within Family 4 (m=3 noise conditions)",
    },
}
