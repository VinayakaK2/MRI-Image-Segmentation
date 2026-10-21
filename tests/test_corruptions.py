"""Unit tests for controlled image corruption engines."""

import pytest
import torch

from src.corruptions import (
    apply_contrast_reduction,
    apply_corruption,
    apply_gaussian_blur,
    apply_gaussian_noise,
    get_corruption_parameter,
)


def test_gaussian_noise_identity_at_zero() -> None:
    """Verifies that std=0.0 leaves the input tensor bitwise unchanged."""
    x = torch.randn(2, 1, 64, 64)
    corrupted = apply_gaussian_noise(x, std=0.0)
    assert torch.equal(x, corrupted)


def test_gaussian_noise_deterministic_seed() -> None:
    """Verifies that identical seeds produce bitwise identical noise perturbations."""
    x = torch.randn(1, 1, 32, 32)
    c1 = apply_gaussian_noise(x, std=0.1, seed=42)
    c2 = apply_gaussian_noise(x, std=0.1, seed=42)
    c3 = apply_gaussian_noise(x, std=0.1, seed=123)

    assert torch.equal(c1, c2)
    assert not torch.equal(c1, c3)


def test_gaussian_noise_severity_scaling() -> None:
    """Verifies that higher noise standard deviation produces greater variance from clean."""
    x = torch.zeros(1, 1, 128, 128)
    c_mild = apply_gaussian_noise(x, std=0.02, seed=42)
    c_severe = apply_gaussian_noise(x, std=0.1, seed=42)

    diff_mild = torch.mean((c_mild - x) ** 2).item()
    diff_severe = torch.mean((c_severe - x) ** 2).item()
    assert diff_mild < diff_severe


def test_gaussian_blur_identity_at_zero() -> None:
    """Verifies that sigma=0.0 leaves the input tensor unchanged."""
    x = torch.randn(1, 1, 64, 64)
    corrupted = apply_gaussian_blur(x, sigma=0.0)
    assert torch.equal(x, corrupted)


def test_gaussian_blur_smoothing_reduces_gradient_energy() -> None:
    """Verifies that spatial blur smooths sharp edges, decreasing spatial gradient variance."""
    # Synthetic 2D impulse / step edge
    x = torch.zeros(1, 1, 64, 64)
    x[:, :, :, 32:] = 1.0

    b_mild = apply_gaussian_blur(x, sigma=0.75)
    b_severe = apply_gaussian_blur(x, sigma=2.5)

    grad_x = torch.diff(x, dim=-1).abs().sum().item()
    grad_mild = torch.diff(b_mild, dim=-1).abs().sum().item()
    grad_severe = torch.diff(b_severe, dim=-1).abs().sum().item()

    # Blurring spreads the step transition, reducing maximum peak edge gradient
    max_grad_x = torch.diff(x, dim=-1).abs().max().item()
    max_grad_mild = torch.diff(b_mild, dim=-1).abs().max().item()
    max_grad_severe = torch.diff(b_severe, dim=-1).abs().max().item()

    assert max_grad_mild < max_grad_x
    assert max_grad_severe < max_grad_mild


def test_contrast_reduction_identity_at_one() -> None:
    """Verifies that factor=1.0 preserves contrast identically."""
    x = torch.randn(1, 1, 64, 64)
    corrupted = apply_contrast_reduction(x, factor=1.0)
    assert torch.equal(x, corrupted)


def test_contrast_reduction_decreases_standard_deviation() -> None:
    """Verifies that lower contrast factor compresses intensity dynamic range towards the mean."""
    x = torch.randn(1, 1, 64, 64) * 2.0 + 1.0
    c_mild = apply_contrast_reduction(x, factor=0.75)
    c_severe = apply_contrast_reduction(x, factor=0.25)

    std_orig = x.std().item()
    std_mild = c_mild.std().item()
    std_severe = c_severe.std().item()

    assert std_mild < std_orig
    assert std_severe < std_mild
    # Mean should be preserved
    assert pytest.approx(x.mean().item(), rel=1e-3) == c_severe.mean().item()


def test_corruptions_preserve_shape_and_finite() -> None:
    """Verifies that all corruptions preserve tensor dimensions and produce finite numbers."""
    x = torch.randn(2, 1, 128, 128)

    for sev in ("mild", "moderate", "severe"):
        for name in ("gaussian_noise", "gaussian_blur", "contrast_reduction"):
            c, meta = apply_corruption(x, name, sev, seed=42)
            assert c.shape == x.shape
            assert c.dtype == x.dtype
            assert torch.isfinite(c).all()
            assert meta["corruption_name"] == name
            assert meta["severity"] == sev


def test_clean_corruption_passthrough() -> None:
    """Verifies that 'clean' or 'identity' returns an exact clone."""
    x = torch.randn(1, 1, 32, 32)
    c, meta = apply_corruption(x, "clean", "none")
    assert torch.equal(x, c)
    assert meta["corruption_name"] == "clean"


def test_invalid_parameters_raise_error() -> None:
    """Verifies defensive error handling for invalid names and parameter values."""
    x = torch.randn(1, 1, 32, 32)
    with pytest.raises(ValueError):
        apply_gaussian_noise(x, std=-0.5)
    with pytest.raises(ValueError):
        apply_gaussian_blur(x, sigma=-1.0)
    with pytest.raises(ValueError):
        apply_contrast_reduction(x, factor=-0.1)
    with pytest.raises(ValueError):
        get_corruption_parameter("non_existent_corruption", "mild")
    with pytest.raises(ValueError):
        get_corruption_parameter("gaussian_noise", "ultra_extreme")
