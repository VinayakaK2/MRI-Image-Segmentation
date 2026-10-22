"""Contrast reduction corruption for normalized cardiac MRI tensors."""

import torch


def apply_contrast_reduction(
    image: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Scales intensity dynamic range around the per-slice spatial mean.

    Args:
        image: Normalized image tensor of shape (B, C, H, W) or (C, H, W) or (H, W).
        factor: Dimensionless contrast factor alpha.
            factor = 1.0 preserves identical contrast (identity).
            factor < 1.0 compresses intensities towards the slice mean.
            factor = 0.0 yields a uniform mean-intensity slice.

    Returns:
        Corrupted image tensor with identical shape and dtype, with unconstrained z-score range.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")
    if factor < 0.0:
        raise ValueError(f"Contrast reduction factor must be non-negative, got {factor}")

    # Identity condition: factor = 1.0 leaves input unchanged
    if factor == 1.0:
        return image.clone()

    orig_dim = image.dim()
    if orig_dim == 2:
        x = image.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 3:
        x = image.unsqueeze(0)
    elif orig_dim == 4:
        x = image
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {image.shape}")

    # Compute spatial mean per slice and channel: (B, C, 1, 1)
    mean = x.mean(dim=(-2, -1), keepdim=True)

    # Scale deviation from the mean by the contrast factor
    # Note: No clamping to [0, 1] is applied because input is in normalized z-score units
    corrupted = mean + factor * (x - mean)

    if not torch.isfinite(corrupted).all():
        raise RuntimeError("Contrast reduction corruption produced non-finite (NaN/Inf) tensor values.")

    # Restore original input dimensionality
    if orig_dim == 2:
        return corrupted.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        return corrupted.squeeze(0)
    return corrupted
