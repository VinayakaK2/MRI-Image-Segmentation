"""Morphological post-processing interventions with class-specific anatomical safeguards.

Provides connected-component analysis, small-object filtering, and topologically safe
hole-filling tailored for 4-class cardiac MRI semantic segmentation (ACDC dataset).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndimage


# Default minimum component sizes in pixels per foreground class
DEFAULT_MIN_SIZES: Dict[int, int] = {
    1: 30,  # Right Ventricle cavity
    2: 50,  # Myocardium muscular ring
    3: 30,  # Left Ventricle cavity
}

# Maximum hole size (in pixels) permitted for myocardium hole-filling
# Prevents catastrophic filling of the central Left Ventricular blood pool cavity
DEFAULT_MAX_MYO_HOLE_SIZE: int = 16


def keep_largest_connected_component(
    mask: np.ndarray,
    connectivity: int = 2,
) -> np.ndarray:
    """Extracts the single largest connected foreground component from a binary mask.

    Args:
        mask: 2D binary numpy array (boolean or integer 0/1).
        connectivity: Neighborhood connectivity (1 for 4-connected, 2 for 8-connected).

    Returns:
        Binary mask of identical shape containing only the largest connected component.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(mask)}")

    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros_like(mask, dtype=mask.dtype)

    # 8-connectivity structuring element for 2D
    struct = ndimage.generate_binary_structure(2, connectivity)
    labeled, num_features = ndimage.label(binary, structure=struct)

    if num_features <= 1:
        return mask.copy()

    # Calculate component sizes (ignoring background label 0)
    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(component_sizes)) + 1

    cleaned = (labeled == largest_label).astype(mask.dtype)
    return cleaned


def filter_small_components(
    mask: np.ndarray,
    min_size: int = 30,
    connectivity: int = 2,
) -> np.ndarray:
    """Removes disconnected foreground components with area smaller than min_size pixels.

    Args:
        mask: 2D binary numpy array.
        min_size: Minimum pixel count required to retain a connected component.
        connectivity: Neighborhood connectivity (1 or 2).

    Returns:
        Filtered binary mask containing only components with area >= min_size.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(mask)}")

    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary) or min_size <= 1:
        return mask.copy()

    struct = ndimage.generate_binary_structure(2, connectivity)
    labeled, num_features = ndimage.label(binary, structure=struct)

    if num_features == 0:
        return np.zeros_like(mask, dtype=mask.dtype)

    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    # Identify labels satisfying the minimum area criterion
    valid_labels = set(np.where(component_sizes >= min_size)[0] + 1)

    cleaned = np.isin(labeled, list(valid_labels)).astype(mask.dtype)
    return cleaned


def fill_mask_holes(
    mask: np.ndarray,
    max_hole_size: Optional[int] = None,
    connectivity: int = 1,
) -> np.ndarray:
    """Fills interior background holes within a binary foreground structure.

    If max_hole_size is specified, only holes with pixel count <= max_hole_size are filled.
    This provides an essential safeguard against filling large anatomical chambers
    (e.g., preserving the hollow center of the doughnut-shaped myocardium).

    Args:
        mask: 2D binary numpy array.
        max_hole_size: Maximum pixel count of holes to fill (None fills all enclosed holes).
        connectivity: Connectivity for hole components (1 for 4-connected, 2 for 8-connected).

    Returns:
        Binary mask with specified enclosed holes filled.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(mask)}")

    binary = (mask > 0).astype(bool)
    if not np.any(binary):
        return mask.copy()

    # Fill all enclosed holes
    all_filled = ndimage.binary_fill_holes(binary)

    # If no size constraint, return standard filled mask
    if max_hole_size is None:
        return all_filled.astype(mask.dtype)

    # Identify individual hole components
    holes = all_filled & (~binary)
    if not np.any(holes):
        return mask.copy()

    struct = ndimage.generate_binary_structure(2, connectivity)
    labeled_holes, num_holes = ndimage.label(holes, structure=struct)

    if num_holes == 0:
        return mask.copy()

    hole_sizes = ndimage.sum(holes, labeled_holes, range(1, num_holes + 1))
    # Select only holes satisfying size <= max_hole_size
    small_hole_labels = set(np.where(hole_sizes <= max_hole_size)[0] + 1)

    if not small_hole_labels:
        return mask.copy()

    small_holes_mask = np.isin(labeled_holes, list(small_hole_labels))
    filled_result = binary | small_holes_mask
    return filled_result.astype(mask.dtype)


def postprocess_cardiac_segmentation(
    pred_mask: np.ndarray,
    enable_lcc: bool = True,
    enable_hole_filling: bool = True,
    min_component_sizes: Optional[Dict[int, int]] = None,
    max_myo_hole_size: int = DEFAULT_MAX_MYO_HOLE_SIZE,
) -> np.ndarray:
    """Applies anatomical post-processing to a 4-class cardiac MRI segmentation mask.

    Class-Specific Anatomical Safeguards:
    - Class 3 (Left Ventricular Cavity):
      Applies Largest Connected Component (LCC) and complete hole filling to enforce
      a contiguous, void-free blood pool chamber.
    - Class 2 (Left Ventricular Myocardium):
      Applies LCC to preserve the main muscular ring and remove stray false-positive islands.
      CRITICAL SAFEGUARD: Hole-filling is constrained to max_myo_hole_size (default 16 px)
      so small intramyocardial voids are sealed without filling the central LV cavity!
    - Class 1 (Right Ventricular Cavity):
      Applies small-component filtering (min_size) and hole filling.
    - Conflict Resolution:
      Cavity blood pools (LV=3, RV=1) take priority over Myocardium (2) in overlap regions.
      Background (0) is assigned to all remaining unsegmented pixels.

    Args:
        pred_mask: 2D integer array of shape (H, W) with class labels {0, 1, 2, 3}.
        enable_lcc: If True, retains only the largest connected component for LV and MYO.
        enable_hole_filling: If True, fills internal voids with class-specific size limits.
        min_component_sizes: Dict mapping class_idx -> min_size_pixels.
        max_myo_hole_size: Maximum hole size (pixels) to fill in Myocardium.

    Returns:
        Post-processed integer mask of shape (H, W) with same dtype.
    """
    if not isinstance(pred_mask, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(pred_mask)}")

    if pred_mask.ndim != 2:
        raise ValueError(f"Expected 2D segmentation mask (H, W), got shape {pred_mask.shape}")

    min_sizes = DEFAULT_MIN_SIZES.copy()
    if min_component_sizes is not None:
        min_sizes.update(min_component_sizes)

    # 1. Process Class 3 (Left Ventricle Cavity)
    lv_mask = (pred_mask == 3).astype(np.uint8)
    if np.any(lv_mask):
        if enable_lcc:
            lv_mask = keep_largest_connected_component(lv_mask)
        else:
            lv_mask = filter_small_components(lv_mask, min_size=min_sizes[3])
        if enable_hole_filling:
            # Unrestricted hole filling is valid for solid LV cavity
            lv_mask = fill_mask_holes(lv_mask, max_hole_size=None)

    # 2. Process Class 2 (Myocardium Ring)
    myo_mask = (pred_mask == 2).astype(np.uint8)
    if np.any(myo_mask):
        if enable_lcc:
            myo_mask = keep_largest_connected_component(myo_mask)
        else:
            myo_mask = filter_small_components(myo_mask, min_size=min_sizes[2])
        if enable_hole_filling:
            # CRITICAL SAFEGUARD: Restrict hole-filling size to protect hollow LV interior
            myo_mask = fill_mask_holes(myo_mask, max_hole_size=max_myo_hole_size)

    # 3. Process Class 1 (Right Ventricle Cavity)
    rv_mask = (pred_mask == 1).astype(np.uint8)
    if np.any(rv_mask):
        # RV may have complex geometry near base; filter small spurious noise
        if enable_lcc:
            rv_mask = keep_largest_connected_component(rv_mask)
        else:
            rv_mask = filter_small_components(rv_mask, min_size=min_sizes[1])
        if enable_hole_filling:
            rv_mask = fill_mask_holes(rv_mask, max_hole_size=None)

    # 4. Conflict Resolution & Reassembly
    # Priority order: Background (0) -> Myocardium (2) -> RV (1) -> LV (3)
    out_mask = np.zeros_like(pred_mask, dtype=pred_mask.dtype)
    out_mask[myo_mask > 0] = 2
    out_mask[rv_mask > 0] = 1
    out_mask[lv_mask > 0] = 3

    return out_mask
