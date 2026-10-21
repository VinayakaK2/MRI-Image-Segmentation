"""Configuration loading, schema validation, and path resolution utilities."""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml


REQUIRED_CONFIG_SECTIONS = (
    "project",
    "dataset",
    "model",
    "training",
    "evaluation",
    "corruptions",
)


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # src/utils/config_parser.py -> parent (utils) -> parent (src) -> parent (root)
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads and parses a YAML configuration file into a dictionary."""
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Specified configuration path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML file at {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must contain a top-level mapping/dict")

    validate_config_schema(data)
    return data


def validate_config_schema(config: Dict[str, Any]) -> bool:
    """Validates that all essential top-level sections exist in the configuration."""
    missing_sections = [sec for sec in REQUIRED_CONFIG_SECTIONS if sec not in config]
    if missing_sections:
        raise KeyError(f"Configuration is missing required section(s): {', '.join(missing_sections)}")
    return True


def resolve_project_path(path_str: str) -> Path:
    """Resolves a relative path string against the project root into an absolute Path."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return get_project_root() / p


def resolve_dataset_raw_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    """Resolves the raw ACDC dataset directory across environments (Colab/Linux and local Windows).

    Resolution strategy:
    1. Respects ACDC_RAW_DIR environment variable if explicitly configured and directory exists.
    2. Resolves configured dataset.raw_dir relative to project root.
    3. Checks canonical Colab path (/content/data/raw/ACDC).
    4. Checks local Windows fallback path (H:/database).

    Raises:
        FileNotFoundError: If the raw ACDC dataset cannot be located in any known path.
    """
    # 1. Environment variable override
    env_path_str = os.environ.get("ACDC_RAW_DIR")
    if env_path_str:
        env_path = Path(env_path_str).resolve()
        if env_path.exists() and (env_path / "training").is_dir():
            return env_path

    # 2. Configured dataset raw_dir
    if config and "dataset" in config and "raw_dir" in config["dataset"]:
        cfg_raw = resolve_project_path(config["dataset"]["raw_dir"])
        if cfg_raw.exists() and (cfg_raw / "training").is_dir():
            return cfg_raw

    # 3. Default relative project path
    default_proj_path = resolve_project_path("data/raw/ACDC")
    if default_proj_path.exists() and (default_proj_path / "training").is_dir():
        return default_proj_path

    # 4. Canonical Colab path
    colab_canonical = Path("/content/data/raw/ACDC")
    if colab_canonical.exists() and (colab_canonical / "training").is_dir():
        return colab_canonical

    # 5. Local Windows fallback
    local_fallback = Path("H:/database")
    if local_fallback.exists() and (local_fallback / "training").is_dir():
        return local_fallback

    raise FileNotFoundError(
        "ACDC raw dataset directory could not be located. "
        "Ensure the dataset is staged at data/raw/ACDC, /content/data/raw/ACDC, or set ACDC_RAW_DIR."
    )
