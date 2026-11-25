"""Compact lossless PNG encoding and geometry restoration helpers for masks, overlays, and previews."""

import base64
from io import BytesIO
from typing import Any, Dict, Tuple
import numpy as np
from PIL import Image

from api.config import OVERLAY_RGBA_PALETTE, VALID_MASK_LABELS
from src.preprocessing.geometry import (
    DEFAULT_TARGET_SPATIAL_SIZE,
    calculate_center_crop_bounds,
    calculate_symmetric_pad_widths,
)
from src.preprocessing.restoration import to_unit_interval


def validate_categorical_mask(mask: np.ndarray) -> np.ndarray:
    """Validates that a 2D segmentation mask contains only integer class labels {0, 1, 2, 3}."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D segmentation mask, got shape {mask.shape}")
    unique_vals = set(np.unique(mask).tolist())
    invalid = unique_vals - set(VALID_MASK_LABELS)
    if invalid:
        raise ValueError(f"Mask contains invalid class labels: {sorted(invalid)}")
    return mask.astype(np.uint8)


def compute_mask_pixel_counts(mask: np.ndarray) -> Dict[str, int]:
    """Computes pixel counts per anatomical class index {0, 1, 2, 3}."""
    return {str(c): int(np.sum(mask == c)) for c in VALID_MASK_LABELS}


def encode_categorical_mask_png_base64(mask: np.ndarray) -> str:
    """Encodes a 2D integer label mask {0, 1, 2, 3} as a lossless 8-bit grayscale PNG base64 string."""
    mask_u8 = validate_categorical_mask(mask)
    img = Image.fromarray(mask_u8, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_rgba_overlay_png_base64(mask: np.ndarray) -> str:
    """Renders a 2D integer label mask into a transparent RGBA PNG base64 string."""
    mask_u8 = validate_categorical_mask(mask)
    height, width = mask_u8.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for class_idx, color in OVERLAY_RGBA_PALETTE.items():
        if class_idx == 0:
            continue
        rgba[mask_u8 == class_idx] = color

    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_mri_preview_png_base64(image_2d: np.ndarray) -> str:
    """Scales a 2D MRI slice to [0, 255] via research to_unit_interval and encodes as PNG base64."""
    arr = np.asarray(image_2d, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D slice for preview, got shape {arr.shape}")
    scaled_01, _, _ = to_unit_interval(arr)
    u8 = np.clip(np.round(scaled_01 * 255.0), 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(u8, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_geometry_transform(
    original_shape: Tuple[int, int],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SPATIAL_SIZE,
) -> Dict[str, Any]:
    """Computes exact padding widths and center-crop bounds matching src.preprocessing.geometry."""
    orig_h, orig_w = int(original_shape[0]), int(original_shape[1])
    target_h, target_w = int(target_size[0]), int(target_size[1])

    pad_top, pad_bottom = (0, 0)
    pad_left, pad_right = (0, 0)
    crop_row_bounds: Tuple[int, int] | None = None
    crop_col_bounds: Tuple[int, int] | None = None

    if orig_h > target_h:
        crop_row_bounds = calculate_center_crop_bounds(orig_h, target_h)
    elif orig_h < target_h:
        pad_top, pad_bottom = calculate_symmetric_pad_widths(orig_h, target_h)

    if orig_w > target_w:
        crop_col_bounds = calculate_center_crop_bounds(orig_w, target_w)
    elif orig_w < target_w:
        pad_left, pad_right = calculate_symmetric_pad_widths(orig_w, target_w)

    was_padded = any(v > 0 for v in (pad_top, pad_bottom, pad_left, pad_right))
    was_cropped = crop_row_bounds is not None or crop_col_bounds is not None

    crop_bounds_dict: Dict[str, int] | None = None
    if was_cropped:
        r_start, r_end = crop_row_bounds if crop_row_bounds is not None else (0, min(orig_h, target_h))
        c_start, c_end = crop_col_bounds if crop_col_bounds is not None else (0, min(orig_w, target_w))
        crop_bounds_dict = {
            "row_start": int(r_start),
            "row_end": int(r_end),
            "col_start": int(c_start),
            "col_end": int(c_end),
        }

    return {
        "original_height": orig_h,
        "original_width": orig_w,
        "model_height": target_h,
        "model_width": target_w,
        "pad_widths": {
            "top": int(pad_top),
            "bottom": int(pad_bottom),
            "left": int(pad_left),
            "right": int(pad_right),
        },
        "crop_bounds": crop_bounds_dict,
        "was_padded": was_padded,
        "was_cropped": was_cropped,
        "cropped_regions_unrecoverable": was_cropped,
    }


def restore_mask_to_display_space(
    model_mask: np.ndarray,
    original_shape: Tuple[int, int],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SPATIAL_SIZE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Maps a standardized (256, 256) mask back to original display dimensions without inventing cropped borders."""
    mask_u8 = validate_categorical_mask(model_mask)
    transform = compute_geometry_transform(original_shape, target_size)

    orig_h = transform["original_height"]
    orig_w = transform["original_width"]
    target_h = transform["model_height"]
    target_w = transform["model_width"]

    if mask_u8.shape != (target_h, target_w):
        raise ValueError(f"Expected model-space mask shape {(target_h, target_w)}, got {mask_u8.shape}")

    pads = transform["pad_widths"]
    # Step 1: Strip any symmetric padding added along H or W
    h_slice_start = pads["top"]
    h_slice_end = target_h - pads["bottom"]
    w_slice_start = pads["left"]
    w_slice_end = target_w - pads["right"]
    unpadded = mask_u8[h_slice_start:h_slice_end, w_slice_start:w_slice_end]

    # Step 2: Embed back into original dimensions (cropped peripheral margins remain 0 / Background)
    display_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    crop_bounds = transform["crop_bounds"]
    if crop_bounds is not None:
        r_start = crop_bounds["row_start"]
        r_end = crop_bounds["row_end"]
        c_start = crop_bounds["col_start"]
        c_end = crop_bounds["col_end"]
    else:
        r_start, r_end = 0, unpadded.shape[0]
        c_start, c_end = 0, unpadded.shape[1]

    display_mask[r_start:r_end, c_start:c_end] = unpadded
    return display_mask, transform
