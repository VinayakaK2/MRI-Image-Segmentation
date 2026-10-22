"""Unified corruption execution engine and parameter registry for cardiac MRI robustness."""

from typing import Any, Dict, Optional, Tuple
import torch

from src.corruptions.blur import apply_gaussian_blur
from src.corruptions.contrast import apply_contrast_reduction
from src.corruptions.noise import apply_gaussian_noise


# Default severity configurations matching configs/baseline_unet.yaml
DEFAULT_CORRUPTIONS_REGISTRY: Dict[str, Dict[str, float]] = {
    "gaussian_noise": {
        "mild": 0.02,
        "moderate": 0.05,
        "severe": 0.1,
    },
    "gaussian_blur": {
        "mild": 0.75,
        "moderate": 1.5,
        "severe": 2.5,
    },
    "contrast_reduction": {
        "mild": 0.75,
        "moderate": 0.5,
        "severe": 0.25,
    },
}

CORRUPTION_UNITS: Dict[str, str] = {
    "gaussian_noise": "normalized z-score std",
    "gaussian_blur": "pixels (filter sigma)",
    "contrast_reduction": "scaling factor alpha",
}


def get_corruption_parameter(
    corruption_name: str,
    severity: str,
    custom_registry: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Retrieves the numeric parameter value for a corruption family and severity level."""
    registry = custom_registry or DEFAULT_CORRUPTIONS_REGISTRY
    name_clean = corruption_name.lower().strip()
    sev_clean = severity.lower().strip()

    if name_clean not in registry:
        raise ValueError(f"Unknown corruption family '{corruption_name}'. Available: {list(registry.keys())}")
    if sev_clean not in registry[name_clean]:
        raise ValueError(
            f"Unknown severity level '{severity}' for '{corruption_name}'. Available: {list(registry[name_clean].keys())}"
        )

    return registry[name_clean][sev_clean]


def apply_corruption(
    image: torch.Tensor,
    corruption_name: str,
    severity: str,
    seed: Optional[int] = None,
    custom_registry: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Applies a specified corruption condition to an input tensor.

    Args:
        image: Normalized image tensor (B, C, H, W) or (C, H, W).
        corruption_name: Name of corruption family ('gaussian_noise', 'gaussian_blur', 'contrast_reduction').
            Or 'clean' / 'identity' for uncorrupted baseline.
        severity: Severity level ('mild', 'moderate', 'severe').
        seed: Random seed for stochastic corruptions (e.g. Gaussian noise).
        custom_registry: Optional override dictionary of severity parameter values.

    Returns:
        Tuple of (corrupted_tensor, metadata_dict).
    """
    name_clean = corruption_name.lower().strip()
    sev_clean = severity.lower().strip()

    if name_clean in ("clean", "identity", "none"):
        return image.clone(), {
            "corruption_name": "clean",
            "severity": "none",
            "parameter_value": 0.0,
            "units": "none",
        }

    param_val = get_corruption_parameter(name_clean, sev_clean, custom_registry)
    units = CORRUPTION_UNITS.get(name_clean, "dimensionless")

    if name_clean == "gaussian_noise":
        corrupted = apply_gaussian_noise(image, std=param_val, seed=seed)
    elif name_clean == "gaussian_blur":
        corrupted = apply_gaussian_blur(image, sigma=param_val)
    elif name_clean == "contrast_reduction":
        corrupted = apply_contrast_reduction(image, factor=param_val)
    else:
        raise ValueError(f"Unhandled corruption name: {corruption_name}")

    metadata = {
        "corruption_name": name_clean,
        "severity": sev_clean,
        "parameter_value": param_val,
        "units": units,
        "seed": seed,
    }
    return corrupted, metadata
