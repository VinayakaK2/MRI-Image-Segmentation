"""Tests for experiment configuration loading and schema validation."""

import pytest
from pathlib import Path
from src.utils.config_parser import (
    load_yaml_config,
    validate_config_schema,
    get_project_root,
)


def test_baseline_config_exists() -> None:
    """Verifies that the default baseline configuration file exists."""
    config_path = get_project_root() / "configs" / "baseline_unet.yaml"
    assert config_path.exists()
    assert config_path.is_file()


def test_baseline_config_loads_valid_dict() -> None:
    """Verifies that the baseline config parses into a valid dict with required keys."""
    cfg = load_yaml_config("configs/baseline_unet.yaml")
    assert isinstance(cfg, dict)
    assert "project" in cfg
    assert "dataset" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert "evaluation" in cfg
    assert "corruptions" in cfg


def test_config_dataset_classes() -> None:
    """Verifies that the dataset configuration defines four cardiac classes."""
    cfg = load_yaml_config("configs/baseline_unet.yaml")
    dataset_cfg = cfg["dataset"]
    assert dataset_cfg["num_classes"] == 4
    labels = dataset_cfg["class_labels"]
    assert len(labels) == 4
    assert labels[0] == "background"
    assert labels[1] == "right_ventricle"
    assert labels[2] == "myocardium"
    assert labels[3] == "left_ventricle"


def test_config_missing_file_raises_error() -> None:
    """Verifies that loading a non-existent configuration raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("configs/non_existent_config.yaml")


def test_config_schema_validation_failure() -> None:
    """Verifies that validate_config_schema raises KeyError when mandatory sections are missing."""
    incomplete_config = {"project": {"name": "test"}}
    with pytest.raises(KeyError) as exc_info:
        validate_config_schema(incomplete_config)
    assert "missing required section" in str(exc_info.value)
