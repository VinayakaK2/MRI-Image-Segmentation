"""Unit tests for slice normalization, clipping, and spatial geometry adjustments."""

import pytest
import numpy as np
from src.preprocessing.normalization import (
    clip_intensities_by_percentiles,
    min_max_scale,
    normalize_mri_slice,
    z_score_standardize,
)
from src.preprocessing.geometry import (
    calculate_center_crop_bounds,
    calculate_symmetric_pad_widths,
    pad_or_crop_to_target,
    verify_foreground_preservation,
)


def test_z_score_standardize_normal() -> None:
    """Verifies that normal non-constant array is standardized to zero-mean and unit variance."""
    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    stdized = z_score_standardize(arr)

    assert np.isclose(np.mean(stdized), 0.0, atol=1e-6)
    assert np.isclose(np.std(stdized), 1.0, atol=1e-5)
    assert stdized.dtype == np.float32


def test_z_score_standardize_constant_slice_safe() -> None:
    """Verifies that constant-valued inputs return zero arrays without crashing or NaNs."""
    constant_arr = np.full((10, 10), 42.0, dtype=np.float32)
    stdized = z_score_standardize(constant_arr)

    assert not np.isnan(stdized).any()
    assert not np.isinf(stdized).any()
    assert np.all(stdized == 0.0)


def test_min_max_scale_normal_and_constant() -> None:
    """Verifies min-max scaling bounds and safe handling of zero-range inputs."""
    arr = np.array([10.0, 50.0, 90.0], dtype=np.float32)
    scaled = min_max_scale(arr)
    assert np.isclose(np.min(scaled), 0.0)
    assert np.isclose(np.max(scaled), 1.0)

    const_arr = np.full((5,), 100.0, dtype=np.float32)
    scaled_const = min_max_scale(const_arr)
    assert np.all(scaled_const == 0.0)


def test_clip_percentiles_invalid_bounds_raise() -> None:
    """Verifies that invalid percentile intervals raise ValueError."""
    arr = np.arange(100, dtype=np.float32)
    with pytest.raises(ValueError):
        clip_intensities_by_percentiles(arr, lower_percentile=90.0, upper_percentile=10.0)
    with pytest.raises(ValueError):
        clip_intensities_by_percentiles(arr, lower_percentile=-1.0, upper_percentile=99.0)


def test_normalize_mri_slice_invalid_method_raises() -> None:
    """Verifies that an unknown normalization strategy raises ValueError."""
    arr = np.ones((5, 5), dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        normalize_mri_slice(arr, method="unsupported_technique")
    assert "Unsupported normalization method" in str(exc_info.value)


def test_pad_or_crop_padding_smaller_slice() -> None:
    """Verifies that slices smaller than target are center-padded symmetrically."""
    small_img = np.ones((200, 240), dtype=np.float32)
    small_mask = np.ones((200, 240), dtype=np.int64)

    out_img, out_mask = pad_or_crop_to_target(small_img, small_mask, target_size=(256, 256))
    assert out_img.shape == (256, 256)
    assert out_mask.shape == (256, 256)
    # Check padding was symmetric: (256 - 200)//2 = 28 top, (256 - 240)//2 = 8 left
    assert out_img[0, 0] == 0.0  # padded margin
    assert out_img[28, 8] == 1.0  # original content start


def test_pad_or_crop_safe_center_cropping() -> None:
    """Verifies that slices larger than target are center-cropped when foreground is centered."""
    large_img = np.ones((300, 300), dtype=np.float32)
    large_mask = np.zeros((300, 300), dtype=np.int64)
    # Place foreground centered at [140:160, 140:160]
    large_mask[140:160, 140:160] = 2

    out_img, out_mask = pad_or_crop_to_target(large_img, large_mask, target_size=(256, 256))
    assert out_img.shape == (256, 256)
    assert out_mask.shape == (256, 256)
    assert np.sum(out_mask == 2) == 20 * 20


def test_pad_or_crop_cropping_severing_foreground_raises() -> None:
    """Verifies that cropping raises ValueError if non-zero foreground labels would be cut off."""
    large_img = np.ones((300, 300), dtype=np.float32)
    large_mask = np.zeros((300, 300), dtype=np.int64)
    # Place foreground in peripheral margin that would be cropped out (rows 0-10)
    large_mask[0:10, 0:10] = 3

    with pytest.raises(ValueError) as exc_info:
        pad_or_crop_to_target(large_img, large_mask, target_size=(256, 256))
    assert "would clip anatomical structures" in str(exc_info.value)
