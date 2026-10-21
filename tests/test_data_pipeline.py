"""Unit tests for ACDC patient discovery, metadata parsing, and stratified splitting."""

import pytest
from pathlib import Path
from src.data.discovery import (
    parse_ed_frame_indices,
    parse_info_cfg,
    validate_nifti_ed_pair,
)
from src.data.patient_split import (
    create_stratified_patient_splits,
    load_patient_splits,
    save_patient_splits,
    verify_patient_splits_integrity,
)


def test_ed_frame_indices_conversion() -> None:
    """Verifies that 1-based string frame converts accurately to (1-based, 0-based) tuple."""
    frame_1based, frame_0based = parse_ed_frame_indices("1")
    assert frame_1based == 1
    assert frame_0based == 0

    frame_1based, frame_0based = parse_ed_frame_indices("14")
    assert frame_1based == 14
    assert frame_0based == 13


def test_ed_frame_indices_invalid_values_raise() -> None:
    """Verifies that non-integer or non-positive ED frame strings raise ValueError."""
    with pytest.raises(ValueError):
        parse_ed_frame_indices("0")
    with pytest.raises(ValueError):
        parse_ed_frame_indices("-3")
    with pytest.raises(ValueError):
        parse_ed_frame_indices("invalid")


def test_parse_info_cfg_success(tmp_path: Path) -> None:
    """Verifies that valid Info.cfg text is parsed into key-value metadata."""
    cfg_file = tmp_path / "Info.cfg"
    cfg_file.write_text("ED: 1\nES: 12\nGroup: DCM\nHeight: 184.0\nNbFrame: 30\nWeight: 95.0\n")

    metadata = parse_info_cfg(cfg_file)
    assert metadata["ED"] == "1"
    assert metadata["ES"] == "12"
    assert metadata["Group"] == "DCM"
    assert metadata["NbFrame"] == "30"


def test_parse_info_cfg_missing_fields_raise(tmp_path: Path) -> None:
    """Verifies that missing mandatory fields in Info.cfg raise KeyError."""
    cfg_file = tmp_path / "Info.cfg"
    cfg_file.write_text("Height: 180.0\nWeight: 80.0\n")

    with pytest.raises(KeyError) as exc_info:
        parse_info_cfg(cfg_file)
    assert "missing mandatory fields" in str(exc_info.value)


def test_patient_stratified_splitting_ratios() -> None:
    """Verifies exact 70/15/15 patient partitioning across balanced pathology groups."""
    # 20 patients per group for 5 groups = 100 patients
    dummy_groups = {}
    for g in ["DCM", "HCM", "MINF", "NOR", "RV"]:
        for i in range(1, 21):
            dummy_groups[f"p_{g}_{i:02d}"] = g

    splits = create_stratified_patient_splits(dummy_groups, seed=42)
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15


def test_patient_stratified_splitting_disjointness() -> None:
    """Verifies that train, val, and test partitions have zero patient overlap (no leakage)."""
    dummy_groups = {f"patient_{i:03d}": "NOR" for i in range(1, 101)}
    splits = create_stratified_patient_splits(dummy_groups, seed=42)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set & val_set) == 0, "Train and Val share patients"
    assert len(train_set & test_set) == 0, "Train and Test share patients"
    assert len(val_set & test_set) == 0, "Val and Test share patients"
    assert len(train_set | val_set | test_set) == 100


def test_patient_stratified_splitting_reproducibility() -> None:
    """Verifies that identical seed produces identical splits and different seeds differ."""
    dummy_groups = {f"p_{i:03d}": "DCM" for i in range(1, 101)}

    split_a = create_stratified_patient_splits(dummy_groups, seed=42)
    split_b = create_stratified_patient_splits(dummy_groups, seed=42)
    split_c = create_stratified_patient_splits(dummy_groups, seed=99)

    assert split_a == split_b, "Identical seed must yield identical partition"
    assert split_a != split_c, "Different seed should produce different partition"


def test_patient_splits_roundtrip_save_load(tmp_path: Path) -> None:
    """Verifies JSON serialization and deserialization of patient split dictionaries."""
    dummy_groups = {f"p_{i:03d}": "HCM" for i in range(1, 101)}
    original_splits = create_stratified_patient_splits(dummy_groups, seed=42)

    save_path = tmp_path / "splits.json"
    save_patient_splits(original_splits, save_path)
    loaded_splits = load_patient_splits(save_path)

    assert loaded_splits == original_splits
