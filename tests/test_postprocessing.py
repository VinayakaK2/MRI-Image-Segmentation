"""Unit tests for morphological post-processing and anatomical safeguards."""

import numpy as np
import pytest

from src.postprocessing.morphology import (
    DEFAULT_MAX_MYO_HOLE_SIZE,
    fill_mask_holes,
    filter_small_components,
    keep_largest_connected_component,
    postprocess_cardiac_segmentation,
)


def test_keep_largest_connected_component():
    """Verifies that smaller disconnected islands are removed."""
    mask = np.zeros((30, 30), dtype=np.uint8)
    # Component 1: large (10x10 = 100 pixels)
    mask[5:15, 5:15] = 1
    # Component 2: small (2x2 = 4 pixels)
    mask[25:27, 25:27] = 1

    cleaned = keep_largest_connected_component(mask)
    assert cleaned.sum() == 100
    assert cleaned[5, 5] == 1
    assert cleaned[25, 25] == 0


def test_keep_largest_connected_component_empty():
    """Verifies that empty masks return all zeros gracefully."""
    empty = np.zeros((20, 20), dtype=np.uint8)
    res = keep_largest_connected_component(empty)
    assert np.all(res == 0)


def test_filter_small_components():
    """Verifies filtering of objects below size threshold."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    # Component A: 50 pixels
    mask[5:10, 5:15] = 1
    # Component B: 10 pixels
    mask[20:22, 20:25] = 1

    filtered = filter_small_components(mask, min_size=30)
    assert filtered[6, 6] == 1
    assert filtered[21, 21] == 0
    assert filtered.sum() == 50


def test_fill_mask_holes_doughnut_safeguard():
    """CRITICAL TEST: Verifies that myocardium hole-filling preserves the large central cavity.

    A circular/doughnut myocardium has a large central cavity (e.g. 100 pixels)
    and potentially small intra-myocardial micro-holes (e.g. 4 pixels).
    Hole filling with max_hole_size=16 must fill the 4-pixel micro-hole
    while strictly keeping the 100-pixel central LV blood pool hollow.
    """
    # Create doughnut ring (outer radius 15, inner radius 8)
    y, x = np.ogrid[:40, :40]
    center = (20, 20)
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2

    myo = ((dist_sq <= 15**2) & (dist_sq >= 8**2)).astype(np.uint8)
    central_cavity_size_before = int(np.sum(dist_sq < 8**2))
    assert central_cavity_size_before > 150  # Central cavity is large

    # Add a small 4-pixel defect/micro-hole inside the myocardium ring
    myo[20:22, 30:32] = 0

    # 1. Unconstrained hole filling WOULD catastrophically fill the entire central cavity
    unconstrained = fill_mask_holes(myo, max_hole_size=None)
    assert unconstrained[20, 20] == 1  # Center filled! Bad for myocardium!

    # 2. Constrained hole filling fills the 4-pixel defect but PROTECTS the central cavity!
    safe_filled = fill_mask_holes(myo, max_hole_size=16)
    assert safe_filled[20, 20] == 0  # Center remains HOLLOW (protected)!
    assert safe_filled[20, 30] == 1  # Small 4-pixel micro-hole is filled!


def test_postprocess_cardiac_segmentation_multiclass():
    """Verifies multi-class reassembly, priority, and spurious component removal."""
    pred = np.zeros((64, 64), dtype=np.int64)

    # True LV cavity at center (radius 8, class 3)
    y, x = np.ogrid[:64, :64]
    center = (32, 32)
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
    pred[dist_sq <= 8**2] = 3

    # True Myocardium ring around LV (radius 9 to 15, class 2)
    pred[(dist_sq > 8**2) & (dist_sq <= 15**2)] = 2

    # Spurious false-positive LV island in top-left corner (class 3, 4 pixels)
    pred[2:4, 2:4] = 3

    # Spurious false-positive MYO island in bottom-right corner (class 2, 6 pixels)
    pred[58:61, 58:60] = 2

    cleaned = postprocess_cardiac_segmentation(
        pred,
        enable_lcc=True,
        enable_hole_filling=True,
        max_myo_hole_size=DEFAULT_MAX_MYO_HOLE_SIZE,
    )

    # Spurious islands must be removed
    assert cleaned[2, 2] == 0
    assert cleaned[59, 59] == 0

    # Main heart structures must be preserved with correct classes
    assert cleaned[32, 32] == 3  # LV center
    assert cleaned[32, 44] == 2  # Myocardium
    assert cleaned[0, 0] == 0    # Background


def test_postprocess_single_undersized_component_removed_across_all_classes():
    """Verifies that single undersized components are removed across RV, MYO, and LV even under enable_lcc=True."""
    for class_idx, min_size in [(1, 30), (2, 50), (3, 30)]:
        pred = np.zeros((40, 40), dtype=np.int64)
        # Create a single small component of 12 pixels (< min_size)
        pred[5:8, 5:9] = class_idx
        assert np.sum(pred == class_idx) == 12

        cleaned = postprocess_cardiac_segmentation(pred, enable_lcc=True)
        # Under enable_lcc=True with size filtering, this component must be removed
        assert np.sum(cleaned == class_idx) == 0, f"Class {class_idx} small island of size 12 should be removed"


def test_postprocess_multiple_undersized_components_all_removed():
    """Verifies that when multiple components exist but the largest is still below min_size, all are removed."""
    pred = np.zeros((50, 50), dtype=np.int64)
    # Class 1 (RV, min_size=30): Component A (20 pixels), Component B (8 pixels)
    pred[5:9, 5:10] = 1   # 4x5 = 20 pixels (largest, but < 30)
    pred[25:27, 25:29] = 1 # 2x4 = 8 pixels
    assert np.sum(pred == 1) == 28

    cleaned = postprocess_cardiac_segmentation(pred, enable_lcc=True)
    assert np.sum(cleaned == 1) == 0


def test_postprocess_component_exactly_at_threshold_survives():
    """Verifies that a component with area exactly equal to the class threshold is retained."""
    # Test Class 1 (RV, threshold 30): 5x6 = 30 pixels
    pred1 = np.zeros((40, 40), dtype=np.int64)
    pred1[10:15, 10:16] = 1
    assert np.sum(pred1 == 1) == 30
    cleaned1 = postprocess_cardiac_segmentation(pred1, enable_lcc=True)
    assert np.sum(cleaned1 == 1) == 30

    # Test Class 2 (Myo, threshold 50): 5x10 = 50 pixels
    pred2 = np.zeros((40, 40), dtype=np.int64)
    pred2[10:15, 10:20] = 2
    assert np.sum(pred2 == 2) == 50
    cleaned2 = postprocess_cardiac_segmentation(pred2, enable_lcc=True)
    assert np.sum(cleaned2 == 2) == 50

    # Test Class 3 (LV, threshold 30): 5x6 = 30 pixels
    pred3 = np.zeros((40, 40), dtype=np.int64)
    pred3[10:15, 10:16] = 3
    assert np.sum(pred3 == 3) == 30
    cleaned3 = postprocess_cardiac_segmentation(pred3, enable_lcc=True)
    assert np.sum(cleaned3 == 3) == 30


def test_postprocess_valid_component_survives_while_smaller_islands_removed():
    """Verifies that a component exceeding min_size survives while small noise islands are removed."""
    pred = np.zeros((50, 50), dtype=np.int64)
    # Valid LV component: 10x10 = 100 pixels (> 30)
    pred[10:20, 10:20] = 3
    # Small noise island: 3x3 = 9 pixels (< 30)
    pred[35:38, 35:38] = 3
    assert np.sum(pred == 3) == 109

    cleaned = postprocess_cardiac_segmentation(pred, enable_lcc=True)
    assert np.sum(cleaned == 3) == 100
    assert np.all(cleaned[10:20, 10:20] == 3)
    assert np.all(cleaned[35:38, 35:38] == 0)


def test_postprocess_empty_mask():
    """Verifies that all-zero masks are handled gracefully without exceptions."""
    empty = np.zeros((32, 32), dtype=np.int64)
    cleaned = postprocess_cardiac_segmentation(empty, enable_lcc=True, enable_hole_filling=True)
    assert np.all(cleaned == 0)
    assert cleaned.shape == (32, 32)
    assert cleaned.dtype == np.int64


def test_postprocess_hole_filling_and_class_conflict():
    """Verifies hole filling interaction with blood pool priorities over myocardium."""
    pred = np.zeros((40, 40), dtype=np.int64)

    # Myocardium solid block 20x20
    pred[10:30, 10:30] = 2

    # LV cavity placed in center 8x8 overlapping Myocardium
    pred[16:24, 16:24] = 3

    # Add 2-pixel hole in LV cavity
    pred[18:20, 18:20] = 0

    cleaned = postprocess_cardiac_segmentation(pred, enable_lcc=True, enable_hole_filling=True)

    # LV should take priority and its hole should be filled
    assert np.all(cleaned[16:24, 16:24] == 3)
    # Outer region should remain Myocardium
    assert cleaned[12, 12] == 2
    assert cleaned[0, 0] == 0
