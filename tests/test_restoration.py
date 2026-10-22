"""Unit tests for inference-time restoration and preprocessing interventions."""

import numpy as np
import pytest
import torch

from src.preprocessing.restoration import (
    apply_clahe,
    apply_gaussian_denoising,
    apply_histogram_equalization,
    apply_median_denoising,
    apply_restoration_pipeline,
    to_unit_interval,
)


def test_to_unit_interval_standard():
    """Verifies linear scaling to [0, 1] range."""
    img = np.array([[-10.0, 0.0], [10.0, 20.0]], dtype=np.float32)
    scaled, min_v, max_v = to_unit_interval(img)
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)
    assert min_v == pytest.approx(-10.0)
    assert max_v == pytest.approx(20.0)


def test_to_unit_interval_constant():
    """Verifies safe handling of constant/flat images without division by zero."""
    img = np.full((16, 16), 5.0, dtype=np.float32)
    scaled, min_v, max_v = to_unit_interval(img)
    assert np.all(scaled == 0.0)
    assert min_v == pytest.approx(5.0)
    assert max_v == pytest.approx(5.0)


def test_apply_clahe_numpy():
    """Verifies CLAHE on 2D numpy array and z-score output standardization."""
    np.random.seed(42)
    img = np.random.randn(64, 64).astype(np.float32)
    enhanced = apply_clahe(img, clip_limit=0.02, kernel_size=(8, 8), re_standardize=True)

    assert enhanced.shape == (64, 64)
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.mean() == pytest.approx(0.0, abs=1e-2)
    assert enhanced.std() == pytest.approx(1.0, abs=1e-2)


def test_apply_clahe_torch_tensor():
    """Verifies CLAHE works transparently on PyTorch tensors with shape preservation."""
    tensor = torch.randn(1, 64, 64, dtype=torch.float32)
    enhanced = apply_clahe(tensor, clip_limit=0.01, kernel_size=(8, 8))

    assert isinstance(enhanced, torch.Tensor)
    assert enhanced.shape == (1, 64, 64)
    assert enhanced.dtype == torch.float32


def test_apply_histogram_equalization():
    """Verifies global histogram equalization on numpy array."""
    np.random.seed(42)
    img = np.random.randn(64, 64).astype(np.float32)
    enhanced = apply_histogram_equalization(img, nbins=64, re_standardize=True)

    assert enhanced.shape == (64, 64)
    assert enhanced.mean() == pytest.approx(0.0, abs=1e-2)
    assert enhanced.std() == pytest.approx(1.0, abs=1e-2)


def test_apply_gaussian_denoising():
    """Verifies Gaussian smoothing reduces high-frequency variance."""
    np.random.seed(42)
    img = np.random.randn(64, 64).astype(np.float32)
    smoothed = apply_gaussian_denoising(img, sigma=1.5, re_standardize=False)

    assert smoothed.shape == (64, 64)
    # Smoothing strictly reduces variance of i.i.d. Gaussian noise
    assert np.var(smoothed) < np.var(img)


def test_apply_median_denoising():
    """Verifies median filter on impulse/salt-and-pepper noise."""
    img = np.ones((32, 32), dtype=np.float32)
    img[10, 10] = 100.0  # Impulse noise spike
    denoised = apply_median_denoising(img, size=3)

    assert denoised.shape == (32, 32)
    assert denoised[10, 10] == pytest.approx(1.0)


def test_apply_restoration_pipeline_dispatcher():
    """Verifies unified pipeline dispatcher handles valid and invalid methods."""
    img = np.random.randn(32, 32).astype(np.float32)
    out_none = apply_restoration_pipeline(img, "none")
    assert np.array_equal(out_none, img)

    out_clahe = apply_restoration_pipeline(img, "clahe", clip_limit=0.01)
    assert out_clahe.shape == (32, 32)

    with pytest.raises(ValueError):
        apply_restoration_pipeline(img, "unsupported_magic_filter")
