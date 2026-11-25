"""Utility module exports for checkpoint loading, image encoding, and validation."""

from api.utils.checkpoint import get_checkpoint_spec, load_verified_model, verify_checkpoint_integrity
from api.utils.image_encoder import (
    compute_geometry_transform,
    compute_mask_pixel_counts,
    encode_categorical_mask_png_base64,
    encode_mri_preview_png_base64,
    encode_rgba_overlay_png_base64,
    restore_mask_to_display_space,
)
from api.utils.validation import (
    ApiError,
    resolve_corruption_spec,
    resolve_mitigation_spec,
    resolve_safe_path_within,
    validate_patient_id_format,
)

__all__ = [
    "ApiError",
    "compute_geometry_transform",
    "compute_mask_pixel_counts",
    "encode_categorical_mask_png_base64",
    "encode_mri_preview_png_base64",
    "encode_rgba_overlay_png_base64",
    "get_checkpoint_spec",
    "load_verified_model",
    "resolve_corruption_spec",
    "resolve_mitigation_spec",
    "resolve_safe_path_within",
    "restore_mask_to_display_space",
    "validate_patient_id_format",
    "verify_checkpoint_integrity",
]
