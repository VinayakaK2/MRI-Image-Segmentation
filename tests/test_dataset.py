"""Integration tests for ACDCSliceDataset, DataLoader batching, and leakage prevention."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ACDCSliceDataset, acdc_collate_fn
from src.utils.config_parser import get_project_root


@pytest.fixture
def manifest_path() -> Path:
    """Provides path to the generated slice manifest file."""
    root = get_project_root()
    p = root / "data" / "processed" / "acdc_slice_manifest.json"
    if not p.exists():
        pytest.skip(f"Slice manifest not found at {p}. Run build_acdc_pipeline_artifacts first.")
    return p


def test_dataset_split_partition_sizes(manifest_path: Path) -> None:
    """Verifies that ACDCSliceDataset partition sizes match the manifest summary."""
    train_ds = ACDCSliceDataset(manifest_path=manifest_path, split="train")
    val_ds = ACDCSliceDataset(manifest_path=manifest_path, split="val")
    test_ds = ACDCSliceDataset(manifest_path=manifest_path, split="test")

    assert len(train_ds) == 678
    assert len(val_ds) == 142
    assert len(test_ds) == 131


def test_dataset_item_shape_and_dtypes(manifest_path: Path) -> None:
    """Verifies that __getitem__ returns (1, 256, 256) float32 and (256, 256) int64."""
    train_ds = ACDCSliceDataset(manifest_path=manifest_path, split="train")
    img_tensor, tgt_tensor, metadata = train_ds[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(tgt_tensor, torch.Tensor)
    assert img_tensor.shape == (1, 256, 256)
    assert tgt_tensor.shape == (256, 256)
    assert img_tensor.dtype == torch.float32
    assert tgt_tensor.dtype == torch.int64

    # Target class values must be strictly within {0, 1, 2, 3}
    unique_classes = set(torch.unique(tgt_tensor).tolist())
    assert unique_classes.issubset({0, 1, 2, 3})


def test_no_patient_leakage_across_partitions(manifest_path: Path) -> None:
    """Rigorous verification: No patient ID from train can appear in val or test partitions."""
    train_ds = ACDCSliceDataset(manifest_path=manifest_path, split="train")
    val_ds = ACDCSliceDataset(manifest_path=manifest_path, split="val")
    test_ds = ACDCSliceDataset(manifest_path=manifest_path, split="test")

    train_patients = {r["patient_id"] for r in train_ds.records}
    val_patients = {r["patient_id"] for r in val_ds.records}
    test_patients = {r["patient_id"] for r in test_ds.records}

    assert len(train_patients) == 70
    assert len(val_patients) == 15
    assert len(test_patients) == 15

    # Check zero intersection
    assert train_patients.isdisjoint(val_patients), "Leakage between train and val cohorts"
    assert train_patients.isdisjoint(test_patients), "Leakage between train and test cohorts"
    assert val_patients.isdisjoint(test_patients), "Leakage between val and test cohorts"


def test_dataloader_batch_generation(manifest_path: Path) -> None:
    """Verifies that DataLoader generates batches with expected tensor shapes."""
    val_ds = ACDCSliceDataset(manifest_path=manifest_path, split="val")
    batch_size = 8
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=acdc_collate_fn)

    batch_imgs, batch_tgts, batch_metas = next(iter(loader))
    assert batch_imgs.shape == (batch_size, 1, 256, 256)
    assert batch_tgts.shape == (batch_size, 256, 256)
    assert len(batch_metas) == batch_size
