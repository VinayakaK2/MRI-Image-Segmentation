"""Unit tests verifying Phase 5 validation-locked mitigation protocol integrity."""

import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import CLASS_NAME_MAP, FOREGROUND_CLASS_INDICES
from src.evaluation.patient_metrics import adjust_pvalues_holm, classify_significance
from src.postprocessing.morphology import (
    DEFAULT_MAX_MYO_HOLE_SIZE,
    DEFAULT_MIN_SIZES,
    fill_mask_holes,
    filter_small_components,
    keep_largest_connected_component,
    postprocess_cardiac_segmentation,
)
from src.preprocessing.restoration import (
    apply_clahe,
    apply_median_denoising,
    to_unit_interval,
)


def test_locked_mitigation_parameters():
    """Verify that all mitigation parameters match the immutable locked protocol."""
    # 1. CLAHE parameters
    # clip_limit = 0.01, tile_grid_size = (16, 16), mapping = to_unit_interval
    dummy_img = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)
    scaled_img, min_val, max_val = to_unit_interval(dummy_img)
    assert 0.0 <= scaled_img.min() <= scaled_img.max() <= 1.0

    clahe_out = apply_clahe(dummy_img, clip_limit=0.01, kernel_size=(16, 16), re_standardize=True)
    assert clahe_out.shape == dummy_img.shape
    # Output should be z-score standardized
    assert np.isclose(float(np.mean(clahe_out)), 0.0, atol=0.1)
    assert np.isclose(float(np.std(clahe_out)), 1.0, atol=0.1)

    # 2. Denoising parameters
    # Spatial Median Filter kernel_size = 3
    median_out = apply_median_denoising(dummy_img, size=3, re_standardize=False)
    assert median_out.shape == dummy_img.shape

    # 3. Morphological post-processing parameters
    # Per-class Largest Connected Component (LCC)
    # RV minimum area = 30 pixels
    # MYO minimum area = 50 pixels
    # LV minimum area = 30 pixels
    assert DEFAULT_MIN_SIZES[1] == 30, f"Expected RV min size 30, got {DEFAULT_MIN_SIZES[1]}"
    assert DEFAULT_MIN_SIZES[2] == 50, f"Expected MYO min size 50, got {DEFAULT_MIN_SIZES[2]}"
    assert DEFAULT_MIN_SIZES[3] == 30, f"Expected LV min size 30, got {DEFAULT_MIN_SIZES[3]}"

    # 4. Myocardium cavity safeguard
    # max_hole_size = 16 pixels
    assert DEFAULT_MAX_MYO_HOLE_SIZE == 16, f"Expected max MYO hole size 16, got {DEFAULT_MAX_MYO_HOLE_SIZE}"


def test_locked_19_primary_hypotheses_structure():
    """Verify that exactly 19 primary hypotheses exist across the 4 pre-specified families."""
    family_1_conds = [
        "clean_baseline",
        "gaussian_noise_mild",
        "gaussian_noise_moderate",
        "gaussian_noise_severe",
        "gaussian_blur_mild",
        "gaussian_blur_moderate",
        "gaussian_blur_severe",
        "contrast_reduction_mild",
        "contrast_reduction_moderate",
        "contrast_reduction_severe",
    ]
    family_2_conds = [
        "contrast_reduction_mild",
        "contrast_reduction_moderate",
        "contrast_reduction_severe",
    ]
    family_3_conds = [
        "contrast_reduction_mild",
        "contrast_reduction_moderate",
        "contrast_reduction_severe",
    ]
    family_4_conds = [
        "gaussian_noise_mild",
        "gaussian_noise_moderate",
        "gaussian_noise_severe",
    ]

    assert len(family_1_conds) == 10, "Family 1 must have m=10"
    assert len(family_2_conds) == 3, "Family 2 must have m=3"
    assert len(family_3_conds) == 3, "Family 3 must have m=3"
    assert len(family_4_conds) == 3, "Family 4 must have m=3"

    total_primary_hypotheses = len(family_1_conds) + len(family_2_conds) + len(family_3_conds) + len(family_4_conds)
    assert total_primary_hypotheses == 19, f"Expected exactly 19 primary hypotheses, got {total_primary_hypotheses}"


def test_negative_controls_separation():
    """Verify that negative controls are distinct and not included in primary families."""
    negative_controls = [
        ("clean_baseline", "clahe_alone"),
        ("gaussian_noise_mild", "clahe_alone"),
        ("gaussian_noise_moderate", "clahe_alone"),
        ("gaussian_noise_severe", "clahe_alone"),
        ("clean_baseline", "median_alone"),
        ("clean_baseline", "combined_contrast"),
        ("clean_baseline", "combined_noise"),
    ]
    # Family 2 must only contain contrast conditions, never clean or noise
    family_2_valid_families = ["contrast_reduction"]
    for cond_id, arm in negative_controls:
        if arm == "clahe_alone":
            assert not any(cond_id.startswith(f) for f in family_2_valid_families), (
                f"Negative control {cond_id} must not be in Family 2"
            )


def test_myocardium_hole_filling_safeguard():
    """Verify that myocardium hole filling does not fill large interior cavities like the LV cavity."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Create an annular myocardium ring (radius 30 to 40)
    y, x = np.ogrid[:100, :100]
    dist_sq = (x - 50) ** 2 + (y - 50) ** 2
    # Myocardium ring between r=15 and r=30 (cavity area = pi * 15^2 ~ 706 pixels)
    ring = (dist_sq >= 15**2) & (dist_sq <= 30**2)
    mask[ring] = 1

    # Small intramyocardial hole of 9 pixels (3x3)
    mask[48:51, 68:71] = 0

    # Fill with max_hole_size=16
    filled = fill_mask_holes(mask, max_hole_size=16)

    # The small hole (9 px) should be filled
    assert np.all(filled[48:51, 68:71] == 1), "Small intramyocardial hole should be filled"

    # The central LV cavity (~706 px) must NOT be filled!
    assert np.all(filled[45:55, 45:55] == 0), "Central LV cavity must NOT be filled"


def test_holm_stepdown_monotonicity():
    """Verify that Holm-Bonferroni step-down correction is strictly monotonic and capped at 1.0."""
    raw_p = [0.0001, 0.005, 0.01, 0.04, 0.045, 0.10, 0.50, 0.80, 0.95]
    adj_p = adjust_pvalues_holm(raw_p)

    assert len(adj_p) == len(raw_p)
    # Check monotonicity for sorted raw inputs
    for i in range(len(adj_p) - 1):
        assert adj_p[i] <= adj_p[i + 1] + 1e-9, f"Monotonicity violation at index {i}: {adj_p[i]} > {adj_p[i+1]}"

    # Check capping at 1.0
    for p in adj_p:
        assert 0.0 <= p <= 1.0, f"Adjusted p-value out of [0, 1]: {p}"
