"""Patient-level stratified dataset splitting utilities for ACDC cardiac MRI."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


# Constants for default partitioning and validation
DEFAULT_TRAIN_RATIO: float = 0.70
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15
DEFAULT_SPLIT_SEED: int = 42
EXPECTED_PRIMARY_COHORT_SIZE: int = 100
EXPECTED_PATHOLOGY_GROUPS: Tuple[str, ...] = ("DCM", "HCM", "MINF", "NOR", "RV")


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validates that partition ratios are non-negative and sum to 1.0 within floating tolerance."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r < 0.0 or r > 1.0 for r in ratios):
        raise ValueError(f"Ratios must be between 0.0 and 1.0, received: {ratios}")

    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Partition ratios must sum to 1.0, received sum: {ratio_sum:.6f}")


def partition_group_patients(
    patient_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    """Partitions a single pathology group list into train, val, and test subsets."""
    # Deterministic sort before shuffle guarantees reproducible permutation across platforms
    shuffled_ids = sorted(list(patient_ids))
    rng.shuffle(shuffled_ids)

    total_count = len(shuffled_ids)
    num_train = int(round(total_count * train_ratio))
    num_val = int(round(total_count * val_ratio))
    # Assign remainder to test to guarantee conservation of patient counts
    num_test = total_count - num_train - num_val

    train_ids = shuffled_ids[:num_train]
    val_ids = shuffled_ids[num_train : num_train + num_val]
    test_ids = shuffled_ids[num_train + num_val :]

    return train_ids, val_ids, test_ids


def create_stratified_patient_splits(
    patient_to_group: Dict[str, str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Dict[str, List[str]]:
    """Creates deterministic, pathology-stratified patient-level partitions (train, val, test)."""
    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    if not patient_to_group:
        raise ValueError("Cannot partition empty patient dictionary")

    # Group patients by pathology label
    grouped: Dict[str, List[str]] = defaultdict(list)
    for p_id, group_name in patient_to_group.items():
        grouped[group_name].append(p_id)

    rng = random.Random(seed)
    train_patients: List[str] = []
    val_patients: List[str] = []
    test_patients: List[str] = []

    for group_name in sorted(grouped.keys()):
        grp_ids = grouped[group_name]
        g_train, g_val, g_test = partition_group_patients(grp_ids, train_ratio, val_ratio, rng)
        train_patients.extend(g_train)
        val_patients.extend(g_val)
        test_patients.extend(g_test)

    splits = {
        "train": sorted(train_patients),
        "val": sorted(val_patients),
        "test": sorted(test_patients),
    }

    verify_patient_splits_integrity(splits, expected_total=len(patient_to_group))
    return splits


def verify_patient_splits_integrity(
    splits: Dict[str, List[str]],
    expected_total: int = EXPECTED_PRIMARY_COHORT_SIZE,
) -> bool:
    """Verifies that splits are strictly disjoint, non-empty, and sum to the expected patient total."""
    required_keys = {"train", "val", "test"}
    if not required_keys.issubset(splits.keys()):
        missing = required_keys - set(splits.keys())
        raise KeyError(f"Splits dictionary missing required partitions: {missing}")

    train_set: Set[str] = set(splits["train"])
    val_set: Set[str] = set(splits["val"])
    test_set: Set[str] = set(splits["test"])

    # Disjointness checks: No patient may leak between partitions
    train_val_overlap = train_set & val_set
    if train_val_overlap:
        raise ValueError(f"Patient leakage detected between train and val: {train_val_overlap}")

    train_test_overlap = train_set & test_set
    if train_test_overlap:
        raise ValueError(f"Patient leakage detected between train and test: {train_test_overlap}")

    val_test_overlap = val_set & test_set
    if val_test_overlap:
        raise ValueError(f"Patient leakage detected between val and test: {val_test_overlap}")

    total_unique = len(train_set | val_set | test_set)
    if total_unique != expected_total:
        raise ValueError(
            f"Expected {expected_total} unique patients across splits, but found {total_unique}"
        )

    return True


def save_patient_splits(
    splits: Dict[str, List[str]],
    output_path: Path,
    metadata: Dict[str, object] = None,
) -> None:
    """Saves patient partitions to a structured, versioned JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = {
        "version": "1.0",
        "description": "Deterministic pathology-stratified patient splits for ACDC primary cohort",
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    if metadata:
        data_to_save["metadata"] = metadata

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2)


def load_patient_splits(input_path: Path) -> Dict[str, List[str]]:
    """Loads and validates patient partitions from a JSON split file."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "splits" not in data:
        raise ValueError(f"Invalid split file schema at {input_path}, expected 'splits' key")

    splits = data["splits"]
    verify_patient_splits_integrity(splits, expected_total=sum(len(v) for v in splits.values()))
    return splits
