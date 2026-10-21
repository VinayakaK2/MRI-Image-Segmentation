"""Spatial geometry standardization, symmetric padding, and non-destructive cropping for 2D slices."""

from typing import Optional, Tuple
import numpy as np


DEFAULT_TARGET_SPATIAL_SIZE: Tuple[int, int] = (256, 256)


def calculate_symmetric_pad_widths(current_dim: int, target_dim: int) -> Tuple[int, int]:
    """Calculates before and after padding widths to center-pad a dimension."""
    deficit = target_dim - current_dim
    pad_before = deficit // 2
    pad_after = deficit - pad_before
    return pad_before, pad_after


def calculate_center_crop_bounds(current_dim: int, target_dim: int) -> Tuple[int, int]:
    """Calculates start and end indices for symmetric center cropping."""
    excess = current_dim - target_dim
    start_idx = excess // 2
    end_idx = start_idx + target_dim
    return start_idx, end_idx


def verify_foreground_preservation(
    mask: np.ndarray,
    crop_h: Optional[Tuple[int, int]] = None,
    crop_w: Optional[Tuple[int, int]] = None,
) -> None:
    """Verifies that center cropping does not clip or sever any non-zero anatomical foreground labels."""
    h_dim, w_dim = mask.shape[:2]

    # Check top / bottom margins
    if crop_h is not None:
        start_h, end_h = crop_h
        if np.any(mask[:start_h, :] > 0) or np.any(mask[end_h:, :] > 0):
            raise ValueError(
                f"Cropping rows to [{start_h}:{end_h}] would clip anatomical structures (image height={h_dim})"
            )

    # Check left / right margins
    if crop_w is not None:
        start_w, end_w = crop_w
        if np.any(mask[:, :start_w] > 0) or np.any(mask[:, end_w:] > 0):
            raise ValueError(
                f"Cropping columns to [{start_w}:{end_w}] would clip anatomical structures (image width={w_dim})"
            )


def pad_or_crop_to_target(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SPATIAL_SIZE,
    image_pad_value: float = 0.0,
    mask_pad_value: int = 0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Standardizes a 2D slice to a target spatial size using geometry-preserving padding and verified cropping."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image (H, W), received ndim={image.ndim} with shape {image.shape}")

    if mask is not None and mask.shape != image.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match mask shape {mask.shape}"
        )

    target_h, target_w = target_size
    curr_h, curr_w = image.shape

    # 1. Handle Height Dimension
    crop_h: Optional[Tuple[int, int]] = None
    pad_h: Optional[Tuple[int, int]] = None

    if curr_h > target_h:
        crop_h = calculate_center_crop_bounds(curr_h, target_h)
    elif curr_h < target_h:
        pad_h = calculate_symmetric_pad_widths(curr_h, target_h)

    # 2. Handle Width Dimension
    crop_w: Optional[Tuple[int, int]] = None
    pad_w: Optional[Tuple[int, int]] = None

    if curr_w > target_w:
        crop_w = calculate_center_crop_bounds(curr_w, target_w)
    elif curr_w < target_w:
        pad_w = calculate_symmetric_pad_widths(curr_w, target_w)

    # 3. Defensive check: Ensure no foreground labels are severed
    if mask is not None and (crop_h is not None or crop_w is not None):
        verify_foreground_preservation(mask, crop_h=crop_h, crop_w=crop_w)

    # 4. Perform Crop Operations
    out_img = image
    out_mask = mask

    if crop_h is not None:
        start_h, end_h = crop_h
        out_img = out_img[start_h:end_h, :]
        if out_mask is not None:
            out_mask = out_mask[start_h:end_h, :]

    if crop_w is not None:
        start_w, end_w = crop_w
        out_img = out_img[:, start_w:end_w]
        if out_mask is not None:
            out_mask = out_mask[:, start_w:end_w]

    # 5. Perform Padding Operations
    if pad_h is not None or pad_w is not None:
        p_top, p_bot = pad_h if pad_h is not None else (0, 0)
        p_left, p_right = pad_w if pad_w is not None else (0, 0)
        pad_spec = ((p_top, p_bot), (p_left, p_right))

        out_img = np.pad(
            out_img,
            pad_spec,
            mode="constant",
            constant_values=image_pad_value,
        )
        if out_mask is not None:
            out_mask = np.pad(
                out_mask,
                pad_spec,
                mode="constant",
                constant_values=mask_pad_value,
            )

    return out_img, out_mask
