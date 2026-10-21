"""Spatial Gaussian blur corruption for normalized cardiac MRI tensors."""

import math
from typing import Tuple
import torch
import torch.nn.functional as F


def _create_gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generates a normalized 1D Gaussian filter kernel with 3-sigma radius."""
    radius = int(math.ceil(3.0 * sigma))
    kernel_size = 2 * radius + 1
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    variance = sigma * sigma
    kernel = torch.exp(-0.5 * (coords * coords) / variance)
    kernel = kernel / torch.sum(kernel)
    return kernel


def apply_gaussian_blur(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Applies isotropic spatial Gaussian blur to an image tensor using separable 2D filtering.

    Args:
        image: Normalized image tensor of shape (B, C, H, W) or (C, H, W).
        sigma: Standard deviation of the Gaussian filter kernel in pixel units.

    Returns:
        Corrupted image tensor with identical spatial dimensions, channel count, and dtype.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")
    if sigma < 0.0:
        raise ValueError(f"Blur standard deviation (sigma) must be non-negative, got {sigma}")

    # Identity condition: zero blur leaves input unchanged
    if sigma == 0.0:
        return image.clone()

    orig_dim = image.dim()
    if orig_dim == 2:
        # (H, W) -> (1, 1, H, W)
        x = image.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 3:
        # (C, H, W) -> (1, C, H, W)
        x = image.unsqueeze(0)
    elif orig_dim == 4:
        # (B, C, H, W)
        x = image
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {image.shape}")

    batch_size, channels, height, width = x.shape
    kernel_1d = _create_gaussian_kernel_1d(sigma, device=x.device, dtype=x.dtype)
    radius = kernel_1d.numel() // 2

    # 1. Horizontal 1D convolution (kernel shape: 1, 1, 1, K)
    kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    # Replicate padding along width
    x_pad_w = F.pad(x, [radius, radius, 0, 0], mode="replicate")
    x_conv_w = F.conv2d(x_pad_w, kernel_h, groups=channels)

    # 2. Vertical 1D convolution (kernel shape: 1, 1, K, 1)
    kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    # Replicate padding along height
    x_pad_h = F.pad(x_conv_w, [0, 0, radius, radius], mode="replicate")
    blurred = F.conv2d(x_pad_h, kernel_v, groups=channels)

    if not torch.isfinite(blurred).all():
        raise RuntimeError("Gaussian blur corruption produced non-finite (NaN/Inf) tensor values.")

    # Restore original input dimensionality
    if orig_dim == 2:
        return blurred.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        return blurred.squeeze(0)
    return blurred
