"""Additive Gaussian noise corruption for normalized cardiac MRI tensors."""

from typing import Optional
import torch


def apply_gaussian_noise(
    image: torch.Tensor,
    std: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Applies zero-mean additive Gaussian noise to a normalized image tensor.

    Args:
        image: Normalized image tensor of shape (B, C, H, W) or (C, H, W) or (H, W).
        std: Standard deviation of the zero-mean Gaussian distribution in normalized-input units.
        seed: Optional integer seed for deterministic noise reproduction.

    Returns:
        Corrupted image tensor with identical shape and dtype, with guaranteed finite values.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")
    if std < 0.0:
        raise ValueError(f"Noise standard deviation must be non-negative, got {std}")

    # Identity condition: zero noise leaves input unchanged
    if std == 0.0:
        return image.clone()

    device = image.device
    gen_device = "cpu" if device.type not in ("cpu", "cuda") else device.type
    generator = None
    if seed is not None:
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(seed)

    # Sample zero-mean Gaussian noise with configured standard deviation
    noise = torch.randn(
        image.shape,
        dtype=image.dtype,
        device=image.device,
        generator=generator,
    ) * std

    corrupted = image + noise

    if not torch.isfinite(corrupted).all():
        raise RuntimeError("Gaussian noise corruption produced non-finite (NaN/Inf) tensor values.")

    return corrupted
