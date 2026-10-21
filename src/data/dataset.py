"""PyTorch Dataset implementation for 2D short-axis cardiac MRI slice segmentation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.slice_extractor import load_raw_2d_slice_pair
from src.preprocessing.geometry import pad_or_crop_to_target
from src.preprocessing.normalization import normalize_mri_slice
from src.utils.config_parser import get_project_root, resolve_dataset_raw_dir


class ACDCSliceDataset(Dataset):
    """PyTorch Dataset providing paired 2D MRI slices and segmentation masks with metadata."""

    def __init__(
        self,
        slice_records: Optional[List[Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None,
        target_spatial_size: Tuple[int, int] = (256, 256),
        normalization_method: str = "z_score",
        clip_percentiles: Optional[Tuple[float, float]] = (0.5, 99.5),
        foreground_only: bool = False,
        cache_data: bool = False,
    ) -> None:
        """Initializes the dataset from in-memory records or a persistent JSON manifest."""
        self.target_spatial_size = target_spatial_size
        self.normalization_method = normalization_method
        self.clip_percentiles = clip_percentiles
        self.foreground_only = foreground_only
        self.cache_data = cache_data
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]] = {}

        # Resolve base directory for relative path resolution
        if base_dir is not None:
            self.base_dir = Path(base_dir).resolve()
        else:
            try:
                self.base_dir = resolve_dataset_raw_dir().resolve()
            except Exception:
                self.base_dir = get_project_root().resolve()

        # Load records
        if slice_records is not None:
            records = list(slice_records)
        elif manifest_path is not None:
            records = self._load_records_from_manifest(Path(manifest_path), split)
        else:
            raise ValueError("Either slice_records or manifest_path must be provided")

        if self.foreground_only:
            records = [r for r in records if r.get("has_foreground", False)]

        if not records:
            raise ValueError(f"ACDCSliceDataset initialized with 0 slice records (split={split})")

        self.records: List[Dict[str, Any]] = records

    @staticmethod
    def _load_records_from_manifest(
        manifest_file: Path,
        target_split: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Loads and filters slice records from a structured manifest file."""
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_slices: List[Dict[str, Any]] = data.get("slices", [])
        if target_split is None:
            return all_slices

        split_clean = target_split.lower().strip()
        filtered = [s for s in all_slices if s.get("split", "").lower() == split_clean]
        return filtered

    def __len__(self) -> int:
        """Returns the total number of 2D slices in the dataset partition."""
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Loads, standardizes, normalizes, and returns a single (image, target, metadata) sample."""
        if index < 0 or index >= len(self.records):
            raise IndexError(f"Slice index {index} out of bounds for dataset of length {len(self.records)}")

        if self.cache_data and index in self._cache:
            return self._cache[index]

        rec = self.records[index]
        slice_idx = rec["slice_idx"]

        # Resolve paths against base directory
        img_path = self.base_dir / rec["image_rel_path"]
        mask_path = self.base_dir / rec["mask_rel_path"]

        # 1. Load raw 2D slice pair from NIfTI
        raw_img, raw_mask = load_raw_2d_slice_pair(img_path, mask_path, slice_idx)

        # 2. Geometry standardization (Symmetric padding or verified center cropping)
        std_img, std_mask = pad_or_crop_to_target(
            image=raw_img,
            mask=raw_mask,
            target_size=self.target_spatial_size,
            image_pad_value=0.0,
            mask_pad_value=0,
        )

        # 3. Deterministic per-slice intensity normalization (Image only)
        norm_img = normalize_mri_slice(
            image=std_img,
            method=self.normalization_method,
            clip_percentiles=self.clip_percentiles,
        )

        # 4. Convert to PyTorch tensors with strict type enforcement
        # Shape: (1, H, W) for single-channel grayscale cardiac MRI
        image_tensor = torch.from_numpy(norm_img).unsqueeze(0).float()
        # Shape: (H, W) categorical class index in range [0, 3]
        target_tensor = torch.from_numpy(std_mask).long()

        # Fixed-size multi-hot boolean tensor for label presence (classes 0, 1, 2, 3)
        # Prevents collation failures with variable-length lists in standard PyTorch DataLoader
        label_presence = torch.zeros(4, dtype=torch.bool)
        for lbl in rec.get("foreground_labels", []):
            if 0 <= lbl < 4:
                label_presence[lbl] = True
        if (std_mask == 0).any():
            label_presence[0] = True

        metadata = {
            "patient_id": rec["patient_id"],
            "group": rec["group"],
            "slice_idx": rec["slice_idx"],
            "total_slices": rec["total_slices"],
            "ed_frame": rec["ed_frame"],
            "has_foreground": rec["has_foreground"],
            "label_presence": label_presence,
        }

        item = (image_tensor, target_tensor, metadata)
        if self.cache_data:
            self._cache[index] = item

        return item


def acdc_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Custom collate function for DataLoader that stacks images and targets while preserving metadata list."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metadatas = [item[2] for item in batch]
    return images, targets, metadatas
