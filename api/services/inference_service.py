"""Synchronous segmentation and stress-test inference service reusing src/ research modules."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import nibabel as nib
import numpy as np
import torch

from api.config import (
    CLASS_MAPPING,
    LOCKED_CLAHE_CLIP_LIMIT,
    LOCKED_CLAHE_KERNEL_SIZE,
    LOCKED_CLAHE_NBINS,
    LOCKED_CLAHE_RE_STANDARDIZE,
    LOCKED_MEDIAN_FILTER_SIZE,
    LOCKED_MEDIAN_RE_STANDARDIZE,
    LOCKED_MORPHOLOGY_MAX_MYO_HOLE_SIZE,
    LOCKED_MORPHOLOGY_MIN_SIZES,
    OVERLAY_COLOR_LEGEND,
    RESEARCH_BASE_SEED,
    RESEARCH_BATCH_SEED_STRIDE,
    VALID_MASK_LABELS,
)
from api.schemas.inference import (
    AppliedCorruptionInfo,
    AppliedMitigationInfo,
    CategoricalMaskPayload,
    GeometryMetadata,
    ImagePreviewPayload,
    InferenceMetadata,
    OverlayPayload,
    SegmentationMetrics,
    SegmentRequest,
    SegmentResponse,
)
from api.services.model_service import ModelService
from api.services.patient_service import PatientService
from api.utils.image_encoder import (
    compute_mask_pixel_counts,
    encode_categorical_mask_png_base64,
    encode_mri_preview_png_base64,
    encode_rgba_overlay_png_base64,
    restore_mask_to_display_space,
)
from api.utils.validation import (
    ApiError,
    extract_corruption_registry_from_config,
    resolve_corruption_spec,
    resolve_mitigation_spec,
    resolve_safe_path_within,
)
from src.corruptions.engine import apply_corruption
from src.data.slice_extractor import SLICE_DEPTH_AXIS, load_raw_2d_slice_pair
from src.evaluation.metrics import compute_multiclass_metrics_numpy
from src.postprocessing.morphology import postprocess_cardiac_segmentation
from src.preprocessing.geometry import pad_or_crop_to_target
from src.preprocessing.normalization import normalize_mri_slice
from src.preprocessing.restoration import apply_clahe, apply_median_denoising


EXECUTION_ORDER: List[str] = [
    "load_raw_2d_ed_slice",
    "pad_or_crop_to_256x256",
    "normalize_mri_slice_clip_and_zscore",
    "optional_corruption_in_normalized_tensor_space",
    "optional_preprocessing_restoration_clahe_or_median",
    "lightweight_unet2d_forward_inference",
    "argmax_class_prediction",
    "optional_morphological_postprocessing",
    "compute_slice_metrics_if_ground_truth_available",
    "restore_display_space_mask_and_serialize",
]


class InferenceService:
    """Executes synchronous single-slice inference strictly following the research pipeline."""

    def __init__(self, model_service: ModelService, patient_service: PatientService) -> None:
        """Binds model session manager and test-cohort patient lookup service."""
        self.model_service = model_service
        self.patient_service = patient_service

    def run_segmentation(self, request: SegmentRequest) -> SegmentResponse:
        """Runs end-to-end synchronous segmentation, optional corruption/mitigation, and metrics."""
        cfg = self.model_service.get_config()
        corruption_cfg = resolve_corruption_spec(request.corruption, cfg)
        mitigation_cfg, experiment_arm = resolve_mitigation_spec(
            request.mitigation,
            corruption_family=corruption_cfg["family"],
        )

        # Verify test slice membership and get deterministic test-loader index
        slice_rec, test_loader_idx, raw_dir = self.patient_service.get_test_slice_context(
            patient_id=request.patient_id,
            slice_index=request.slice_index,
        )
        session = self.model_service.get_or_load_session(request.checkpoint_id)

        # Resolve NIfTI paths safely inside dataset root
        img_path = resolve_safe_path_within(raw_dir, str(slice_rec["image_rel_path"]))
        if not img_path.is_file():
            raise ApiError(
                status_code=503,
                code="IMAGE_FILE_MISSING",
                message=f"MRI volume for patient '{slice_rec['patient_id']}' is unavailable.",
            )

        mask_rel = str(slice_rec.get("mask_rel_path", ""))
        mask_path: Optional[Path] = resolve_safe_path_within(raw_dir, mask_rel) if mask_rel else None
        gt_available = mask_path is not None and mask_path.is_file()

        raw_img, raw_mask = self._load_slice_arrays(
            img_path=img_path,
            mask_path=mask_path if gt_available else None,
            slice_idx=int(slice_rec["slice_idx"]),
        )

        # Step 1-2: Geometry standardization + baseline normalization matching ACDCSliceDataset
        target_size = tuple(int(v) for v in cfg.get("dataset", {}).get("target_spatial_size", [256, 256]))
        preproc_base = cfg.get("preprocessing", {}).get("baseline", {})
        norm_method = str(preproc_base.get("normalization", "z_score"))
        clip_percentiles = tuple(float(v) for v in preproc_base.get("clip_percentiles", [0.5, 99.5]))

        std_img, std_mask = self._standardize_geometry(raw_img, raw_mask, target_size)
        norm_img = normalize_mri_slice(
            image=std_img,
            method=norm_method,
            clip_percentiles=clip_percentiles,
        )

        # Convert to (1, 1, H, W) float32 tensor on active model device
        clean_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(session.device)

        # Step 3: Optional corruption in normalized tensor space
        effective_seed = self._resolve_seed(corruption_cfg, test_loader_idx, int(slice_rec["slice_idx"]))
        corrupt_tensor = self._apply_optional_corruption(
            image_tensor=clean_tensor,
            corruption_cfg=corruption_cfg,
            seed=effective_seed,
            cfg=cfg,
        )

        # Step 4: Optional preprocessing restoration (CLAHE or Median)
        model_input_tensor = self._apply_optional_restoration(corrupt_tensor, mitigation_cfg)

        # Step 5: Synchronous U-Net forward pass + argmax with CUDA-aware timing
        pred_model_space, duration_ms = self._run_model_forward(
            model=session.model,
            model_input=model_input_tensor,
            device=session.device,
        )

        # Step 6: Optional morphological post-processing
        if mitigation_cfg["morphology"]:
            pred_model_space = postprocess_cardiac_segmentation(
                pred_model_space,
                enable_lcc=True,
                enable_hole_filling=True,
                min_component_sizes=dict(LOCKED_MORPHOLOGY_MIN_SIZES),
                max_myo_hole_size=LOCKED_MORPHOLOGY_MAX_MYO_HOLE_SIZE,
            ).astype(np.uint8)

        # Step 7: Ground-truth metrics (strictly on standardized model-space when GT exists)
        metrics_payload: Optional[SegmentationMetrics] = None
        gt_mask_payload: Optional[CategoricalMaskPayload] = None
        gt_overlay_payload: Optional[OverlayPayload] = None

        if gt_available and std_mask is not None:
            std_mask_u8 = std_mask.astype(np.uint8)
            metrics_payload = self._build_metrics(
                pred_mask=pred_model_space,
                gt_mask=std_mask_u8,
                patient_id=str(slice_rec["patient_id"]),
                slice_idx=int(slice_rec["slice_idx"]),
            )
            gt_mask_payload = self._build_mask_payload(std_mask_u8, space="model_space")
            gt_overlay_payload = self._build_overlay_payload(std_mask_u8, space="model_space")

        # Step 8: Display-space restoration and serialization
        display_mask, geom_transform = restore_mask_to_display_space(
            model_mask=pred_model_space,
            original_shape=(raw_img.shape[0], raw_img.shape[1]),
            target_size=(target_size[0], target_size[1]),
        )

        spacing = [float(x) for x in slice_rec["voxel_spacing"]] if slice_rec.get("voxel_spacing") else None
        orientation_note = (
            "Display space preserves original 2D slice dimensions and voxel spacing from the ACDC manifest; "
            "NIfTI affine matrices are not modified or synthesized."
        )
        if geom_transform["was_cropped"]:
            orientation_note += (
                " Input slice exceeded 256x256 and was center-cropped for model inference; "
                "peripheral cropped regions cannot be recovered and remain Background (0) in display space."
            )

        geometry = GeometryMetadata(
            original_height=geom_transform["original_height"],
            original_width=geom_transform["original_width"],
            model_height=geom_transform["model_height"],
            model_width=geom_transform["model_width"],
            voxel_spacing=spacing,
            pad_widths=geom_transform["pad_widths"],
            crop_bounds=geom_transform["crop_bounds"],
            was_padded=geom_transform["was_padded"],
            was_cropped=geom_transform["was_cropped"],
            cropped_regions_unrecoverable=geom_transform["cropped_regions_unrecoverable"],
            affine_preserved=False,
            orientation_note=orientation_note,
        )

        model_input_np = model_input_tensor.detach().cpu().numpy()[0, 0]
        metadata = InferenceMetadata(
            patient_id=str(slice_rec["patient_id"]),
            slice_index=int(slice_rec["slice_idx"]),
            ed_frame=int(slice_rec["ed_frame"]),
            pathology_group=str(slice_rec.get("group", "UNKNOWN")),
            checkpoint_id=session.checkpoint_id,
            checkpoint_sha256=session.checkpoint_sha256,
            device=str(session.device),
            inference_duration_ms=round(duration_ms, 3),
            applied_corruption=AppliedCorruptionInfo(
                family=corruption_cfg["family"],
                severity=corruption_cfg["severity"],
                parameter_value=corruption_cfg["parameter_value"],
                units=corruption_cfg["units"],
                seed=effective_seed if corruption_cfg["family"] == "gaussian_noise" else None,
            ),
            applied_mitigation=AppliedMitigationInfo(**mitigation_cfg),
            registered_experiment_arm=experiment_arm,
            execution_order=list(EXECUTION_ORDER),
        )

        return SegmentResponse(
            metadata=metadata,
            geometry=geometry,
            input_preview_model_space=ImagePreviewPayload(
                space="model_space",
                height=int(model_input_np.shape[0]),
                width=int(model_input_np.shape[1]),
                format="png",
                png_base64=encode_mri_preview_png_base64(model_input_np),
            ),
            input_preview_display_space=ImagePreviewPayload(
                space="display_space",
                height=int(raw_img.shape[0]),
                width=int(raw_img.shape[1]),
                format="png",
                png_base64=encode_mri_preview_png_base64(raw_img),
            ),
            model_space_mask=self._build_mask_payload(pred_model_space, space="model_space"),
            display_space_mask=self._build_mask_payload(display_mask, space="display_space"),
            model_space_overlay=self._build_overlay_payload(pred_model_space, space="model_space"),
            display_space_overlay=self._build_overlay_payload(display_mask, space="display_space"),
            ground_truth_available=gt_available,
            ground_truth_model_space_mask=gt_mask_payload,
            ground_truth_model_space_overlay=gt_overlay_payload,
            metrics=metrics_payload,
        )

    @staticmethod
    def _load_slice_arrays(
        img_path: Path,
        mask_path: Optional[Path],
        slice_idx: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Loads 2D MRI slice and optional ground-truth mask without fabricating missing masks."""
        if mask_path is not None:
            return load_raw_2d_slice_pair(img_path, mask_path, slice_idx)

        img_nii = nib.load(str(img_path))
        img_data = img_nii.get_fdata()
        total_slices = img_data.shape[SLICE_DEPTH_AXIS]
        if slice_idx < 0 or slice_idx >= total_slices:
            raise ApiError(
                status_code=404,
                code="SLICE_NOT_FOUND",
                message=f"Slice index {slice_idx} out of bounds for volume depth {total_slices}.",
            )
        return img_data[:, :, slice_idx].astype(np.float32), None

    @staticmethod
    def _standardize_geometry(
        raw_img: np.ndarray,
        raw_mask: Optional[np.ndarray],
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Applies research pad_or_crop_to_target with explicit error handling."""
        try:
            return pad_or_crop_to_target(
                image=raw_img,
                mask=raw_mask,
                target_size=target_size,
                image_pad_value=0.0,
                mask_pad_value=0,
            )
        except ValueError as exc:
            raise ApiError(
                status_code=422,
                code="INVALID_SLICE_GEOMETRY",
                message=f"Geometry standardization failed: {exc}",
            ) from exc

    @staticmethod
    def _resolve_seed(
        corruption_cfg: Dict[str, Any],
        test_loader_idx: int,
        slice_idx: int,
    ) -> Optional[int]:
        """Determines deterministic noise seed matching evaluate_robustness.py / evaluate_mitigation_test.py."""
        if corruption_cfg["family"] != "gaussian_noise":
            return None
        if corruption_cfg.get("seed") is not None:
            return int(corruption_cfg["seed"])
        return RESEARCH_BASE_SEED + (test_loader_idx * RESEARCH_BATCH_SEED_STRIDE) + int(slice_idx)

    @staticmethod
    def _apply_optional_corruption(
        image_tensor: torch.Tensor,
        corruption_cfg: Dict[str, Any],
        seed: Optional[int],
        cfg: Dict[str, Any],
    ) -> torch.Tensor:
        """Applies research corruption engine in normalized model-input space."""
        if corruption_cfg["family"] == "clean":
            return image_tensor
        registry = extract_corruption_registry_from_config(cfg)
        corrupted, _ = apply_corruption(
            image=image_tensor,
            corruption_name=corruption_cfg["family"],
            severity=corruption_cfg["severity"],
            seed=seed,
            custom_registry=registry,
        )
        return corrupted

    @staticmethod
    def _apply_optional_restoration(
        image_tensor: torch.Tensor,
        mitigation_cfg: Dict[str, Any],
    ) -> torch.Tensor:
        """Applies locked Phase 5 CLAHE or median filtering prior to U-Net forward pass."""
        mode = mitigation_cfg["preprocessing_mode"]
        if mode == "clahe":
            return apply_clahe(
                image_tensor,
                clip_limit=LOCKED_CLAHE_CLIP_LIMIT,
                kernel_size=LOCKED_CLAHE_KERNEL_SIZE,
                nbins=LOCKED_CLAHE_NBINS,
                re_standardize=LOCKED_CLAHE_RE_STANDARDIZE,
            )
        if mode == "median":
            return apply_median_denoising(
                image_tensor,
                size=LOCKED_MEDIAN_FILTER_SIZE,
                re_standardize=LOCKED_MEDIAN_RE_STANDARDIZE,
            )
        return image_tensor

    @staticmethod
    def _run_model_forward(
        model: torch.nn.Module,
        model_input: torch.Tensor,
        device: torch.device,
    ) -> Tuple[np.ndarray, float]:
        """Executes forward pass under torch.inference_mode() with CUDA-synchronized timing."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()

        with torch.inference_mode():
            logits = model(model_input)
            pred_mask = torch.argmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.uint8)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        return pred_mask, duration_ms

    @staticmethod
    def _build_metrics(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        patient_id: str,
        slice_idx: int,
    ) -> SegmentationMetrics:
        """Computes per-class and foreground Dice/IoU using src.evaluation.metrics."""
        raw_metrics = compute_multiclass_metrics_numpy(pred_mask, gt_mask, num_classes=len(CLASS_MAPPING))
        return SegmentationMetrics(
            aggregation_level="single_2d_slice",
            sample_description=f"{patient_id}_slice{slice_idx:02d}",
            evaluation_space="standardized_model_space_256x256",
            dice_per_class={
                "background": float(raw_metrics["dice_background"]),
                "rv": float(raw_metrics["dice_rv"]),
                "myo": float(raw_metrics["dice_myo"]),
                "lv": float(raw_metrics["dice_lv"]),
            },
            iou_per_class={
                "background": float(raw_metrics["iou_background"]),
                "rv": float(raw_metrics["iou_rv"]),
                "myo": float(raw_metrics["iou_myo"]),
                "lv": float(raw_metrics["iou_lv"]),
            },
            mean_dice_foreground=float(raw_metrics["mean_dice_foreground"]),
            mean_iou_foreground=float(raw_metrics["mean_iou_foreground"]),
            mean_dice_all=float(raw_metrics["mean_dice_all"]),
            mean_iou_all=float(raw_metrics["mean_iou_all"]),
        )

    @staticmethod
    def _build_mask_payload(mask: np.ndarray, space: str) -> CategoricalMaskPayload:
        """Serializes a 2D categorical mask {0, 1, 2, 3} with explicit dimensions and pixel counts."""
        return CategoricalMaskPayload(
            space=space,
            height=int(mask.shape[0]),
            width=int(mask.shape[1]),
            encoding="base64_png_uint8_labels",
            data_base64=encode_categorical_mask_png_base64(mask),
            valid_labels=list(VALID_MASK_LABELS),
            pixel_counts_per_class=compute_mask_pixel_counts(mask),
        )

    @staticmethod
    def _build_overlay_payload(mask: np.ndarray, space: str) -> OverlayPayload:
        """Serializes a transparent RGBA PNG overlay matching mask dimensions."""
        return OverlayPayload(
            space=space,
            height=int(mask.shape[0]),
            width=int(mask.shape[1]),
            format="png_rgba",
            png_base64=encode_rgba_overlay_png_base64(mask),
            color_legend=OVERLAY_COLOR_LEGEND,
        )
