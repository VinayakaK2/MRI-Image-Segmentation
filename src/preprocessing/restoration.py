"""Image restoration and contrast enhancement preprocessing interventions for MRI slices.

Provides inference-time preprocessing filters including CLAHE, global histogram equalization,
Gaussian smoothing, and median denoising with mathematically principled intensity scaling.
"""

from typing import Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndimage
from skimage import exposure
import torch

from src.preprocessing.normalization import z_score_standardize


# Default intensity epsilon to avoid division by zero in homogeneous regions
DEFAULT_EPSILON: float = 1e-8


def to_unit_interval(
    image: np.ndarray,
    eps: float = DEFAULT_EPSILON,
) -> Tuple[np.ndarray, float, float]:
    """Scales an image's intensities linearly to the unit interval [0.0, 1.0].

    Returns:
        scaled_image: Array scaled to [0.0, 1.0].
        min_val: Original minimum intensity.
        max_val: Original maximum intensity.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

    min_val = float(np.min(image))
    max_val = float(np.max(image))
    intensity_range = max_val - min_val

    # Defensive check: Constant-valued or nearly uniform inputs produce zero range
    if intensity_range < eps:
        return np.zeros_like(image, dtype=np.float32), min_val, max_val

    scaled = (image - min_val) / (intensity_range + eps)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32), min_val, max_val


def apply_clahe(
    image: Union[np.ndarray, torch.Tensor],
    clip_limit: float = 0.01,
    kernel_size: Optional[Tuple[int, int]] = (16, 16),
    nbins: int = 256,
    re_standardize: bool = True,
    eps: float = DEFAULT_EPSILON,
) -> Union[np.ndarray, torch.Tensor]:
    """Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    Pipeline:
    1. Rescales input to unit interval [0.0, 1.0].
    2. Applies scikit-image CLAHE with contextual tile grids and clipping.
    3. Re-standardizes to zero mean and unit variance (z-score) so the
       activation distribution matches the model's trained weight expectations.

    Args:
        image: 2D array or PyTorch tensor (1, H, W) or (H, W).
        clip_limit: Normalized clipping limit (higher = more contrast amplification).
        kernel_size: Contextual window tile dimensions (height, width).
        nbins: Number of histogram bins.
        re_standardize: If True, re-normalizes output to z-score (zero-mean, unit-variance).
        eps: Numerical stability constant.

    Returns:
        Enhanced image with same type and shape as input.
    """
    is_tensor = isinstance(image, torch.Tensor)
    device = image.device if is_tensor else None

    # Convert to 2D numpy array
    if is_tensor:
        np_img = image.detach().cpu().numpy().squeeze().astype(np.float32)
    else:
        np_img = np.asarray(image, dtype=np.float32).squeeze()

    if np_img.ndim != 2:
        raise ValueError(f"Expected 2D image slice, got shape {np_img.shape}")

    # Step 1: Scale to [0, 1] for histogram processing
    img_01, min_val, max_val = to_unit_interval(np_img, eps=eps)
    if max_val - min_val < eps:
        enhanced = np.zeros_like(np_img, dtype=np.float32)
    else:
        # Step 2: Apply adaptive histogram equalization
        # kernel_size in skimage: tuple of ints indicating window size in pixels
        enhanced = exposure.equalize_adapthist(
            img_01,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        ).astype(np.float32)

    # Step 3: Re-standardize back to z-score distribution for U-Net compatibility
    if re_standardize:
        enhanced = z_score_standardize(enhanced, eps=eps)

    # Convert back to original container type and shape
    if is_tensor:
        out_tensor = torch.from_numpy(enhanced).to(device=device, dtype=image.dtype)
        if image.ndim == 3 and out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0)
        elif image.ndim == 4 and out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0).unsqueeze(0)
        return out_tensor
    else:
        if np.asarray(image).ndim > 2:
            return enhanced.reshape(np.asarray(image).shape)
        return enhanced


def apply_histogram_equalization(
    image: Union[np.ndarray, torch.Tensor],
    nbins: int = 256,
    re_standardize: bool = True,
    eps: float = DEFAULT_EPSILON,
) -> Union[np.ndarray, torch.Tensor]:
    """Applies global histogram equalization across the entire slice.

    Pipeline:
    1. Rescales input to unit interval [0.0, 1.0].
    2. Computes cumulative histogram and flattens probability density.
    3. Re-standardizes to zero-mean and unit variance (z-score).
    """
    is_tensor = isinstance(image, torch.Tensor)
    device = image.device if is_tensor else None

    if is_tensor:
        np_img = image.detach().cpu().numpy().squeeze().astype(np.float32)
    else:
        np_img = np.asarray(image, dtype=np.float32).squeeze()

    if np_img.ndim != 2:
        raise ValueError(f"Expected 2D image slice, got shape {np_img.shape}")

    img_01, min_val, max_val = to_unit_interval(np_img, eps=eps)
    if max_val - min_val < eps:
        enhanced = np.zeros_like(np_img, dtype=np.float32)
    else:
        enhanced = exposure.equalize_hist(img_01, nbins=nbins).astype(np.float32)

    if re_standardize:
        enhanced = z_score_standardize(enhanced, eps=eps)

    if is_tensor:
        out_tensor = torch.from_numpy(enhanced).to(device=device, dtype=image.dtype)
        if image.ndim == 3 and out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0)
        elif image.ndim == 4 and out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0).unsqueeze(0)
        return out_tensor
    else:
        if np.asarray(image).ndim > 2:
            return enhanced.reshape(np.asarray(image).shape)
        return enhanced


def apply_gaussian_denoising(
    image: Union[np.ndarray, torch.Tensor],
    sigma: float = 1.0,
    re_standardize: bool = False,
    eps: float = DEFAULT_EPSILON,
) -> Union[np.ndarray, torch.Tensor]:
    """Applies isotropic Gaussian spatial smoothing to reduce high-frequency additive noise."""
    is_tensor = isinstance(image, torch.Tensor)
    device = image.device if is_tensor else None

    if is_tensor:
        np_img = image.detach().cpu().numpy().squeeze().astype(np.float32)
    else:
        np_img = np.asarray(image, dtype=np.float32).squeeze()

    if np_img.ndim != 2:
        raise ValueError(f"Expected 2D image slice, got shape {np_img.shape}")

    smoothed = ndimage.gaussian_filter(np_img, sigma=sigma, mode="reflect").astype(np.float32)

    if re_standardize:
        smoothed = z_score_standardize(smoothed, eps=eps)

    if is_tensor:
        out_tensor = torch.from_numpy(smoothed).to(device=device, dtype=image.dtype)
        if image.ndim == 3 and out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0)
        elif image.ndim == 4 and out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0).unsqueeze(0)
        return out_tensor
    else:
        if np.asarray(image).ndim > 2:
            return smoothed.reshape(np.asarray(image).shape)
        return smoothed


def apply_median_denoising(
    image: Union[np.ndarray, torch.Tensor],
    size: int = 3,
    re_standardize: bool = False,
    eps: float = DEFAULT_EPSILON,
) -> Union[np.ndarray, torch.Tensor]:
    """Applies 2D median filtering for edge-preserving impulsive and Gaussian noise suppression."""
    is_tensor = isinstance(image, torch.Tensor)
    device = image.device if is_tensor else None

    if is_tensor:
        np_img = image.detach().cpu().numpy().squeeze().astype(np.float32)
    else:
        np_img = np.asarray(image, dtype=np.float32).squeeze()

    if np_img.ndim != 2:
        raise ValueError(f"Expected 2D image slice, got shape {np_img.shape}")

    denoised = ndimage.median_filter(np_img, size=size, mode="reflect").astype(np.float32)

    if re_standardize:
        denoised = z_score_standardize(denoised, eps=eps)

    if is_tensor:
        out_tensor = torch.from_numpy(denoised).to(device=device, dtype=image.dtype)
        if image.ndim == 3 and out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0)
        elif image.ndim == 4 and out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0).unsqueeze(0)
        return out_tensor
    else:
        if np.asarray(image).ndim > 2:
            return denoised.reshape(np.asarray(image).shape)
        return denoised


def apply_restoration_pipeline(
    image: Union[np.ndarray, torch.Tensor],
    intervention_type: str,
    **kwargs,
) -> Union[np.ndarray, torch.Tensor]:
    """Unified dispatcher for all inference-time preprocessing interventions."""
    inv = intervention_type.lower().strip()
    if inv in ("none", "baseline"):
        return image
    elif inv in ("clahe", "contrast_limited_adaptive_he"):
        return apply_clahe(image, **kwargs)
    elif inv in ("histogram_equalization", "he", "global_he"):
        return apply_histogram_equalization(image, **kwargs)
    elif inv in ("gaussian", "gaussian_filter", "gaussian_denoising"):
        return apply_gaussian_denoising(image, **kwargs)
    elif inv in ("median", "median_filter", "median_denoising"):
        return apply_median_denoising(image, **kwargs)
    else:
        raise ValueError(
            f"Unknown restoration intervention '{intervention_type}'. "
            "Supported: ['none', 'clahe', 'histogram_equalization', 'gaussian', 'median']"
        )
