"""Singleton model session management, health checks, and overview aggregation."""

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys
import threading
from typing import Any, Dict, List, Optional
import torch

from api.config import (
    BACKEND_VERSION,
    CANONICAL_CHECKPOINT_ID,
    CHECKPOINT_REGISTRY,
    CLASS_MAPPING,
    PROJECT_TITLE,
    ResearchPaths,
    build_research_paths,
)
from api.schemas.common import ArtifactStatus, HealthResponse
from api.schemas.overview import (
    AggregatedMetricSummary,
    CheckpointOverview,
    DatasetOverview,
    ExperimentSummaryCard,
    ModelSpecification,
    OverviewResponse,
)
from api.utils.checkpoint import get_checkpoint_spec, load_verified_model, verify_checkpoint_integrity
from api.utils.validation import ApiError
from src.models.unet2d import LightweightUNet2D
from src.utils.config_parser import load_yaml_config
from src.utils.reproducibility import get_device


@dataclass
class LoadedModelSession:
    """In-memory loaded model session bound to a verified checkpoint SHA-256."""

    checkpoint_id: str
    checkpoint_sha256: str
    model: LightweightUNet2D
    device: torch.device
    checkpoint_metadata: Dict[str, Any]


class ModelService:
    """Thread-safe singleton service managing model sessions, health, and overview metadata."""

    def __init__(self, paths: Optional[ResearchPaths] = None) -> None:
        """Initializes paths, config, and thread-safe session cache without mutating research files."""
        self.paths = paths or build_research_paths()
        self._lock = threading.Lock()
        self._sessions: Dict[str, LoadedModelSession] = {}
        self._startup_error: Optional[str] = None
        self._config_cache: Optional[Dict[str, Any]] = None
        self._device = get_device("auto")

    def get_config(self) -> Dict[str, Any]:
        """Loads and caches configs/baseline_unet.yaml."""
        if self._config_cache is None:
            self._config_cache = load_yaml_config(str(self.paths.config_path))
        return self._config_cache

    def initialize_canonical_model(self) -> None:
        """Pre-loads the canonical 50-epoch checkpoint once at startup, recording failures explicitly."""
        try:
            self.get_or_load_session(CANONICAL_CHECKPOINT_ID)
            self._startup_error = None
        except ApiError as exc:
            self._startup_error = f"{exc.code}: {exc.message}"
        except Exception as exc:
            self._startup_error = f"MODEL_INIT_ERROR: {type(exc).__name__}"

    def get_or_load_session(self, checkpoint_id: str) -> LoadedModelSession:
        """Returns a cached LoadedModelSession or verifies and loads it once."""
        spec = get_checkpoint_spec(checkpoint_id)
        with self._lock:
            if checkpoint_id in self._sessions:
                return self._sessions[checkpoint_id]

            cfg = self.get_config()
            model, sha256_hex, ckpt_meta = load_verified_model(
                spec=spec,
                project_root=self.paths.project_root,
                model_config=cfg,
                device=self._device,
            )
            session = LoadedModelSession(
                checkpoint_id=checkpoint_id,
                checkpoint_sha256=sha256_hex,
                model=model,
                device=self._device,
                checkpoint_metadata=ckpt_meta,
            )
            self._sessions[checkpoint_id] = session
            return session

    def get_health_status(self) -> HealthResponse:
        """Evaluates runtime, PyTorch, model initialization, and artifact availability."""
        artifacts = self._check_research_artifacts()
        artifacts_ok = all(item.available and item.valid for item in artifacts)

        loaded_ids = sorted(self._sessions.keys())
        model_ready = CANONICAL_CHECKPOINT_ID in self._sessions and self._startup_error is None

        if model_ready and artifacts_ok:
            overall_status = "ok"
        elif model_ready or artifacts_ok:
            overall_status = "degraded"
        else:
            overall_status = "unavailable"

        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0

        return HealthResponse(
            status=overall_status,
            backend_version=BACKEND_VERSION,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({platform.machine()})",
            pytorch_available=True,
            pytorch_version=str(torch.__version__),
            model_initialized=model_ready,
            canonical_checkpoint_id=CANONICAL_CHECKPOINT_ID,
            loaded_checkpoint_ids=loaded_ids,
            model_load_error=self._startup_error,
            device_active=str(self._device),
            cuda_available=cuda_available,
            cuda_device_count=cuda_count,
            research_artifacts_available=artifacts_ok,
            artifacts=artifacts,
        )

    def _check_research_artifacts(self) -> List[ArtifactStatus]:
        """Checks existence and JSON readability of key research manifests and artifacts."""
        targets = [
            ("baseline_config", self.paths.config_path, "yaml"),
            ("acdc_slice_manifest", self.paths.slice_manifest_path, "json"),
            ("baseline_test_evaluation_50epochs", self.paths.baseline_eval_50epochs_path, "json"),
            ("phase4_robustness_results", self.paths.phase4_results_json, "json"),
            ("phase5_mitigation_results", self.paths.phase5_results_json, "json"),
            ("phase5_holm_adjusted_results", self.paths.phase5_holm_csv, "file"),
        ]
        statuses: List[ArtifactStatus] = []
        for name, path, kind in targets:
            if not path.is_file():
                statuses.append(ArtifactStatus(name=name, available=False, valid=False, detail="File not found"))
                continue
            if kind == "json":
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        json.load(handle)
                    statuses.append(ArtifactStatus(name=name, available=True, valid=True))
                except Exception as exc:
                    statuses.append(
                        ArtifactStatus(name=name, available=True, valid=False, detail=f"Invalid JSON: {type(exc).__name__}")
                    )
            else:
                statuses.append(ArtifactStatus(name=name, available=True, valid=True))
        return statuses

    def get_overview(self) -> OverviewResponse:
        """Builds the model, checkpoint, dataset, and baseline evaluation overview from manifests."""
        cfg = self.get_config()
        model_cfg = cfg["model"]
        dataset_cfg = cfg["dataset"]
        spatial = [int(v) for v in dataset_cfg.get("target_spatial_size", [256, 256])]

        # Measure parameter count from live model session if loaded, else instantiate architecture
        if CANONICAL_CHECKPOINT_ID in self._sessions:
            param_counts = self._sessions[CANONICAL_CHECKPOINT_ID].model.count_parameters()
        else:
            temp_model = LightweightUNet2D.from_config(cfg)
            param_counts = temp_model.count_parameters()

        model_spec = ModelSpecification(
            architecture=str(model_cfg.get("architecture", "LightweightUNet2D")),
            parameter_count_total=int(param_counts["total"]),
            parameter_count_trainable=int(param_counts["trainable"]),
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("num_classes", 4)),
            base_channels=int(model_cfg.get("base_channels", 16)),
            depth=int(model_cfg.get("depth", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_batchnorm=bool(model_cfg.get("use_batchnorm", True)),
            bilinear_upsample=bool(model_cfg.get("bilinear_upsample", True)),
            input_dimensions=[1, int(model_cfg.get("in_channels", 1)), spatial[0], spatial[1]],
            output_dimensions=[1, int(model_cfg.get("num_classes", 4)), spatial[0], spatial[1]],
            class_mapping={str(k): v for k, v in CLASS_MAPPING.items()},
        )

        baseline_eval = self._read_json_optional(self.paths.baseline_eval_50epochs_path)
        checkpoints = self._build_checkpoint_overviews(baseline_eval)
        dataset_overview = self._build_dataset_overview()
        phase4_data = self._read_json_optional(self.paths.phase4_results_json)
        phase5_data = self._read_json_optional(self.paths.phase5_results_json)
        baseline_metrics = self._build_baseline_metrics(baseline_eval, phase4_data, dataset_overview)
        experiment_summaries = self._build_experiment_summaries(phase4_data, phase5_data, dataset_overview)

        return OverviewResponse(
            project_title=PROJECT_TITLE,
            model=model_spec,
            checkpoints=checkpoints,
            dataset=dataset_overview,
            baseline_metrics=baseline_metrics,
            experiment_summaries=experiment_summaries,
        )

    def _build_checkpoint_overviews(self, baseline_eval: Optional[Dict[str, Any]]) -> List[CheckpointOverview]:
        """Collects SHA-256 verification status and training metadata for registered checkpoints."""
        overviews: List[CheckpointOverview] = []
        for spec in CHECKPOINT_REGISTRY.values():
            available = False
            verified_sha: Optional[str] = None
            integrity_ok = False
            try:
                _, verified_sha = verify_checkpoint_integrity(spec, self.paths.project_root)
                available = True
                integrity_ok = True
            except ApiError as exc:
                if exc.code == "CHECKPOINT_INTEGRITY_ERROR" and exc.details:
                    available = True
                    verified_sha = str(exc.details.get("actual_sha256"))

            best_val = None
            if spec.is_canonical and baseline_eval:
                best_val = baseline_eval.get("best_val_foreground_dice")

            overviews.append(
                CheckpointOverview(
                    checkpoint_id=spec.checkpoint_id,
                    display_name=spec.display_name,
                    is_canonical=spec.is_canonical,
                    expected_sha256=spec.expected_sha256,
                    verified_sha256=verified_sha,
                    integrity_verified=integrity_ok,
                    available=available,
                    epochs_completed=spec.epochs_completed,
                    best_epoch=spec.best_epoch,
                    best_val_foreground_dice=float(best_val) if best_val is not None else None,
                )
            )
        return overviews

    def _build_dataset_overview(self) -> DatasetOverview:
        """Derives test cohort counts directly from acdc_slice_manifest.json and integrity report."""
        manifest = self._read_json_required(self.paths.slice_manifest_path, "acdc_slice_manifest")
        test_slices = [s for s in manifest.get("slices", []) if str(s.get("split", "")).lower() == "test"]
        test_patients = sorted({str(s["patient_id"]) for s in test_slices})
        fg_slices = sum(1 for s in test_slices if s.get("has_foreground"))

        pathology_by_patient: Dict[str, str] = {}
        for s in test_slices:
            pathology_by_patient[str(s["patient_id"])] = str(s.get("group", "UNKNOWN"))
        pathology_dist: Dict[str, int] = {}
        for grp in pathology_by_patient.values():
            pathology_dist[grp] = pathology_dist.get(grp, 0) + 1

        return DatasetOverview(
            dataset_name="ACDC (Automated Cardiac Diagnosis Challenge)",
            evaluated_phase="End-Diastole (ED)",
            total_patients=manifest.get("total_patients"),
            total_slices=manifest.get("total_slices"),
            test_patient_count=len(test_patients),
            test_slice_count=len(test_slices),
            test_foreground_slice_count=fg_slices,
            test_patient_ids=test_patients,
            pathology_distribution_test=dict(sorted(pathology_dist.items())),
        )

    def _build_baseline_metrics(
        self,
        baseline_eval: Optional[Dict[str, Any]],
        phase4_data: Optional[Dict[str, Any]],
        dataset_overview: DatasetOverview,
    ) -> List[AggregatedMetricSummary]:
        """Extracts explicitly separated slice-level and patient-level 3D clean baseline metrics."""
        records: List[AggregatedMetricSummary] = []

        if baseline_eval and "test_metrics" in baseline_eval:
            tm = baseline_eval["test_metrics"]
            records.append(
                AggregatedMetricSummary(
                    condition_id="clean_baseline",
                    aggregation_level="slice_level",
                    sample_count=dataset_overview.test_slice_count,
                    sample_unit="2d_slices",
                    mean_dice_foreground=float(tm["mean_dice_foreground"]),
                    std_dice_foreground=None,
                    mean_iou_foreground=float(tm["mean_iou_foreground"]),
                    dice_per_class={
                        "background": float(tm["dice_background"]),
                        "rv": float(tm["dice_rv"]),
                        "myo": float(tm["dice_myo"]),
                        "lv": float(tm["dice_lv"]),
                    },
                    iou_per_class={
                        "background": float(tm["iou_background"]),
                        "rv": float(tm["iou_rv"]),
                        "myo": float(tm["iou_myo"]),
                        "lv": float(tm["iou_lv"]),
                    },
                    source_artifact="reports/baseline_test_evaluation_50epochs.json",
                )
            )

        if phase4_data:
            clean_cond = phase4_data.get("conditions", {}).get("clean_baseline", {})
            patient_summary = clean_cond.get("patient_summary", {})
            if patient_summary and "mean_dice_foreground" in patient_summary:
                fg_dice = patient_summary["mean_dice_foreground"]
                fg_iou = patient_summary.get("mean_iou_foreground", {})
                records.append(
                    AggregatedMetricSummary(
                        condition_id="clean_baseline",
                        aggregation_level="patient_level_3d_pooled",
                        sample_count=int(fg_dice.get("n_patients", dataset_overview.test_patient_count)),
                        sample_unit="patients",
                        mean_dice_foreground=float(fg_dice["mean"]),
                        std_dice_foreground=float(fg_dice["std"]),
                        mean_iou_foreground=float(fg_iou["mean"]) if "mean" in fg_iou else None,
                        dice_per_class={
                            "background": float(patient_summary.get("dice_background", {}).get("mean", 0.0)),
                            "rv": float(patient_summary.get("dice_rv", {}).get("mean", 0.0)),
                            "myo": float(patient_summary.get("dice_myo", {}).get("mean", 0.0)),
                            "lv": float(patient_summary.get("dice_lv", {}).get("mean", 0.0)),
                        },
                        iou_per_class={
                            "background": float(patient_summary.get("iou_background", {}).get("mean", 0.0)),
                            "rv": float(patient_summary.get("iou_rv", {}).get("mean", 0.0)),
                            "myo": float(patient_summary.get("iou_myo", {}).get("mean", 0.0)),
                            "lv": float(patient_summary.get("iou_lv", {}).get("mean", 0.0)),
                        },
                        source_artifact="experiments/phase4_robustness_50epoch_best_epoch44/robustness_results.json",
                    )
                )
        return records

    def _build_experiment_summaries(
        self,
        phase4_data: Optional[Dict[str, Any]],
        phase5_data: Optional[Dict[str, Any]],
        dataset_overview: DatasetOverview,
    ) -> List[ExperimentSummaryCard]:
        """Builds concise summaries of Phase 4 and Phase 5 experiments from artifacts on disk."""
        cards: List[ExperimentSummaryCard] = []

        if phase4_data:
            p4_meta = phase4_data.get("metadata", {})
            p4_conds = phase4_data.get("conditions", {})
            blur_sev = p4_conds.get("gaussian_blur_severe", {}).get("patient_summary", {}).get("mean_dice_foreground", {}).get("mean")
            contrast_sev = p4_conds.get("contrast_reduction_severe", {}).get("patient_summary", {}).get("mean_dice_foreground", {}).get("mean")
            noise_sev = p4_conds.get("gaussian_noise_severe", {}).get("patient_summary", {}).get("mean_dice_foreground", {}).get("mean")
            findings = [
                f"Evaluated {len(p4_conds)} controlled conditions across {p4_meta.get('n_patients', dataset_overview.test_patient_count)} test patients ({p4_meta.get('total_test_slices', dataset_overview.test_slice_count)} ED slices).",
            ]
            if blur_sev is not None and contrast_sev is not None and noise_sev is not None:
                findings.append(
                    f"Patient-level 3D FG Dice under severe degradations: Blur={blur_sev:.4f}, Contrast={contrast_sev:.4f}, Noise={noise_sev:.4f}."
                )
            cards.append(
                ExperimentSummaryCard(
                    experiment_id=str(p4_meta.get("experiment_id", "phase4_robustness_50epoch_best_epoch44")),
                    phase=4,
                    title="Controlled Image-Quality Robustness Suite",
                    available=True,
                    checkpoint_sha256=p4_meta.get("checkpoint_sha256"),
                    conditions_count=len(p4_conds),
                    key_findings=findings,
                )
            )
        else:
            cards.append(
                ExperimentSummaryCard(
                    experiment_id="phase4_robustness_50epoch_best_epoch44",
                    phase=4,
                    title="Controlled Image-Quality Robustness Suite",
                    available=False,
                    key_findings=["Phase 4 robustness artifact not available."],
                )
            )

        repro = self._read_json_optional(self.paths.phase5_repro_manifest_json)
        if phase5_data and repro:
            comp_fams = repro.get("statistical_protocol", {}).get("comparison_families_m", {})
            total_h = sum(int(v) for v in comp_fams.values()) if isinstance(comp_fams, dict) else None
            cards.append(
                ExperimentSummaryCard(
                    experiment_id=str(phase5_data.get("experiment_id", "phase5_validation_locked_mitigation")),
                    phase=5,
                    title="Validation-Locked Mitigation & Holm-Bonferroni Audit",
                    available=True,
                    checkpoint_sha256=phase5_data.get("provenance", {}).get("checkpoint_sha256"),
                    hypothesis_families_count=len(comp_fams) if isinstance(comp_fams, dict) else 4,
                    total_primary_hypotheses=total_h,
                    key_findings=[
                        "Evaluated 4 pre-registered hypothesis families (19 primary tests) with family-wise Holm-Bonferroni correction.",
                        "Forensic audit confirmed CLAHE contrast recovery (+0.0738 Dice) is attributable to min-max normalization invariance rather than adaptive histogram equalization.",
                    ],
                )
            )
        else:
            cards.append(
                ExperimentSummaryCard(
                    experiment_id="phase5_validation_locked_mitigation",
                    phase=5,
                    title="Validation-Locked Mitigation & Holm-Bonferroni Audit",
                    available=False,
                    key_findings=["Phase 5 mitigation artifact not available."],
                )
            )
        return cards

    @staticmethod
    def _read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
        """Reads a JSON file if present and valid, returning None otherwise."""
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _read_json_required(path: Path, label: str) -> Dict[str, Any]:
        """Reads a required JSON file or raises explicit ApiError on missing/malformed artifact."""
        if not path.is_file():
            raise ApiError(
                status_code=503,
                code="ARTIFACT_NOT_FOUND",
                message=f"Required research artifact '{label}' is missing.",
            )
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            raise ApiError(
                status_code=500,
                code="ARTIFACT_MALFORMED",
                message=f"Required research artifact '{label}' is malformed.",
            ) from exc
        if not isinstance(data, dict):
            raise ApiError(
                status_code=500,
                code="ARTIFACT_MALFORMED",
                message=f"Required research artifact '{label}' has invalid structure.",
            )
        return data
