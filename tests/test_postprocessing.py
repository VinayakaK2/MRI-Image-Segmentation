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
