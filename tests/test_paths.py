"""Tests for workspace paths, directory structure, and resolution utilities."""

from pathlib import Path
import pytest
from src.utils.config_parser import (
    get_project_root,
    resolve_project_path,
    resolve_dataset_raw_dir,
)


REQUIRED_PROJECT_DIRECTORIES = [
    "configs",
    "src/data",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/corruptions",
    "src/preprocessing",
    "src/postprocessing",
    "src/utils",
    "tests",
    "scripts",
    "checkpoints",
    "experiments",
    "reports",
    "data/raw",
    "data/processed",
]


def test_project_root_detection() -> None:
    """Verifies that get_project_root identifies the expected workspace root."""
    root = get_project_root()
    assert root.exists()
    assert (root / "README.md").exists()
    assert (root / "configs").exists()


def test_required_directories_exist() -> None:
    """Verifies that all specified project foundation directories exist on disk."""
    root = get_project_root()
    for rel_dir in REQUIRED_PROJECT_DIRECTORIES:
        dir_path = root / rel_dir
        assert dir_path.exists(), f"Required directory missing: {rel_dir}"
        assert dir_path.is_dir(), f"Expected directory but found file: {rel_dir}"


def test_resolve_project_path() -> None:
    """Verifies relative and absolute path resolution."""
    root = get_project_root()
    # Relative path resolution
    resolved_rel = resolve_project_path("configs/baseline_unet.yaml")
    assert resolved_rel == root / "configs" / "baseline_unet.yaml"

    # Absolute path identity
    abs_path = (root / "README.md").resolve()
    resolved_abs = resolve_project_path(str(abs_path))
    assert resolved_abs == abs_path


def test_resolve_dataset_raw_dir_success() -> None:
    """Verifies that the canonical dataset path resolves to an existing directory with training data."""
    raw_dir = resolve_dataset_raw_dir()
    assert raw_dir.exists(), f"Resolved dataset path does not exist: {raw_dir}"
    assert (raw_dir / "training").is_dir(), f"Training directory missing in {raw_dir}"
    assert (raw_dir / "training" / "patient001" / "Info.cfg").is_file(), "patient001/Info.cfg missing"


def test_resolve_dataset_raw_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that ACDC_RAW_DIR environment variable takes precedence when valid."""
    custom_dir = tmp_path / "custom_acdc"
    (custom_dir / "training").mkdir(parents=True)
    monkeypatch.setenv("ACDC_RAW_DIR", str(custom_dir))

    resolved = resolve_dataset_raw_dir()
    assert resolved == custom_dir


def test_resolve_dataset_raw_dir_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that resolve_dataset_raw_dir raises FileNotFoundError when no valid dataset exists."""
    empty_root = tmp_path / "empty_proj"
    empty_root.mkdir()
    monkeypatch.setattr("src.utils.config_parser.get_project_root", lambda: empty_root)
    monkeypatch.setattr("src.utils.config_parser.Path.exists", lambda p: False)
    monkeypatch.delenv("ACDC_RAW_DIR", raising=False)

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_dataset_raw_dir()
    assert "ACDC raw dataset directory could not be located" in str(exc_info.value)
