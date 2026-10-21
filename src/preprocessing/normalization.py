"""Intensity normalization and percentile clipping utilities for 2D MRI slices."""

from typing import Optional, Tuple
import numpy as np


# Default numerical safety epsilon to prevent division by zero in homogeneous regions
DEFAULT_EPSILON: float = 1e-8
DEFAULT_CLIP_PERCENTILES: Tuple[float, float] = (0.5, 99.5)


def clip_intensities_by_percentiles(
    image: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Clips extreme intensity outliers based on configurable percentiles."""
    if lower_percentile < 0.0 or upper_percentile > 100.0 or lower_percentile >= upper_percentile:
        raise ValueError(
            f"Invalid percentile bounds: [{lower_percentile}, {upper_percentile}]. "
            "Expected 0 <= lower < upper <= 100."
        )

    p_low, p_high = np.percentile(image, [lower_percentile, upper_percentile])
    return np.clip(image, p_low, p_high)


def z_score_standardize(image: np.ndarray, eps: float = DEFAULT_EPSILON) -> np.ndarray:
    """Standardizes slice intensities to zero-mean and unit variance with safe constant handling."""
    mean_val = float(np.mean(image))
    std_val = float(np.std(image))

    # Defensive check: Constant-valued or nearly flat slices produce zero deviation
    if std_val < eps:
        return np.zeros_like(image, dtype=np.float32)

    standardized = (image - mean_val) / (std_val + eps)
    return standardized.astype(np.float32)


def min_max_scale(image: np.ndarray, eps: float = DEFAULT_EPSILON) -> np.ndarray:
    """Scales intensities to range [0.0, 1.0] with safe handling of uniform inputs."""
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    val_range = max_val - min_val

    if val_range < eps:
        return np.zeros_like(image, dtype=np.float32)

    scaled = (image - min_val) / (val_range + eps)
    return scaled.astype(np.float32)


def normalize_mri_slice(
    image: np.ndarray,
    method: str = "z_score",
    clip_percentiles: Optional[Tuple[float, float]] = DEFAULT_CLIP_PERCENTILES,
    eps: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Applies deterministic, per-slice intensity normalization to a 2D MRI slice."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

    # Ensure float32 representation and sanitize non-finite elements
    processed = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Optional percentile clipping
    if clip_percentiles is not None:
        processed = clip_intensities_by_percentiles(
            processed,
            lower_percentile=clip_percentiles[0],
            upper_percentile=clip_percentiles[1],
        )

    norm_method = method.lower().strip()
    if norm_method == "z_score":
        return z_score_standardize(processed, eps=eps)
    elif norm_method == "min_max":
        return min_max_scale(processed, eps=eps)
    elif norm_method == "none":
        return processed
    else:
        raise ValueError(
            f"Unsupported normalization method '{method}'. Allowed: ['z_score', 'min_max', 'none']"
        )
