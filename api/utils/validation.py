"""Request validation, path traversal prevention, and research parameter resolution."""

import math
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

from api.config import (
    CORRUPTION_TYPE_ALIASES,
    LOCKED_CLAHE_CLIP_LIMIT,
    LOCKED_CLAHE_KERNEL_SIZE,
    LOCKED_CLAHE_NBINS,
    LOCKED_CLAHE_RE_STANDARDIZE,
    LOCKED_MEDIAN_FILTER_SIZE,
    LOCKED_MEDIAN_RE_STANDARDIZE,
    LOCKED_MORPHOLOGY_MAX_MYO_HOLE_SIZE,
    LOCKED_MORPHOLOGY_MIN_SIZES,
)
from api.schemas.inference import CorruptionSpec, MitigationSpec
from src.corruptions.engine import CORRUPTION_UNITS, DEFAULT_CORRUPTIONS_REGISTRY


PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ApiError(Exception):
    """Structured domain exception converted to clean HTTP JSON errors by FastAPI handlers."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes an API error with HTTP status code and machine-readable code."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def validate_patient_id_format(patient_id: str) -> str:
    """Validates a patient ID token to prevent path traversal or injection."""
    cleaned = patient_id.strip()
    if not PATIENT_ID_PATTERN.match(cleaned):
        raise ApiError(
            status_code=400,
            code="INVALID_PATIENT_ID",
            message="Patient ID contains invalid characters.",
        )
    return cleaned


def resolve_safe_path_within(base_dir: Path, rel_path: str) -> Path:
    """Resolves a relative path inside base_dir and blocks any traversal outside it."""
    base_resolved = base_dir.resolve()
    candidate = (base_resolved / rel_path).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ApiError(
            status_code=400,
            code="INVALID_RESOURCE_PATH",
            message="Requested resource path resolves outside the allowed dataset directory.",
        ) from exc
    return candidate


def extract_corruption_registry_from_config(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extracts the corruption parameter registry from baseline_unet.yaml or defaults."""
    registry: Dict[str, Dict[str, float]] = {
        k: dict(v) for k, v in DEFAULT_CORRUPTIONS_REGISTRY.items()
    }
    families = cfg.get("corruptions", {}).get("families", [])
    if isinstance(families, list):
        for fam in families:
            name = fam.get("name")
            levels = fam.get("levels")
            if name and isinstance(levels, dict):
                registry[name] = {k: float(v) for k, v in levels.items()}
    return registry


def resolve_corruption_spec(
    spec: Optional[CorruptionSpec],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Validates and normalizes a corruption request strictly against research-supported severities."""
    if spec is None:
        return {
            "family": "clean",
            "severity": "none",
            "parameter_value": 0.0,
            "units": "none",
            "seed": None,
        }

    raw_type = spec.type.strip().lower()
    family = CORRUPTION_TYPE_ALIASES.get(raw_type)
    if family is None:
        raise ApiError(
            status_code=422,
            code="UNSUPPORTED_CORRUPTION",
            message=f"Unsupported corruption type '{spec.type}'.",
            details={"supported_types": ["clean", "gaussian_noise", "gaussian_blur", "contrast_reduction"]},
        )

    if family == "clean":
        if spec.severity is not None and str(spec.severity).strip().lower() not in ("none", "0", "0.0"):
            raise ApiError(
                status_code=422,
                code="UNSUPPORTED_SEVERITY",
                message="Clean baseline must not specify a non-zero corruption severity.",
            )
        return {
            "family": "clean",
            "severity": "none",
            "parameter_value": 0.0,
            "units": "none",
            "seed": None,
        }

    registry = extract_corruption_registry_from_config(cfg)
    allowed_levels = registry.get(family, {})
    if spec.severity is None:
        raise ApiError(
            status_code=422,
            code="MISSING_SEVERITY",
            message=f"Severity is required for corruption '{family}'.",
            details={"allowed_levels": allowed_levels},
        )

    resolved_tier: Optional[str] = None
    if isinstance(spec.severity, str):
        sev_str = spec.severity.strip().lower()
        if sev_str in allowed_levels:
            resolved_tier = sev_str
        else:
            # Also support numeric string matching registered values
            try:
                numeric_val = float(sev_str)
            except ValueError:
                numeric_val = None
            if numeric_val is not None:
                resolved_tier = _match_numeric_severity(numeric_val, allowed_levels)
    else:
        resolved_tier = _match_numeric_severity(float(spec.severity), allowed_levels)

    if resolved_tier is None:
        raise ApiError(
            status_code=422,
            code="UNSUPPORTED_SEVERITY",
            message=(
                f"Severity '{spec.severity}' is not supported for corruption '{family}'. "
                "Only registered research severity tiers and parameter values are allowed."
            ),
            details={"allowed_levels": allowed_levels},
        )

    return {
        "family": family,
        "severity": resolved_tier,
        "parameter_value": float(allowed_levels[resolved_tier]),
        "units": CORRUPTION_UNITS.get(family, "dimensionless"),
        "seed": spec.seed,
    }


def _match_numeric_severity(value: float, allowed_levels: Dict[str, float]) -> Optional[str]:
    """Maps a numeric severity strictly to an exact registered tier without silent clamping."""
    if not math.isfinite(value):
        return None
    for tier_name, tier_val in allowed_levels.items():
        if math.isclose(value, float(tier_val), rel_tol=1e-6, abs_tol=1e-9):
            return tier_name
    return None


def resolve_mitigation_spec(
    spec: Optional[MitigationSpec],
    corruption_family: str,
) -> Tuple[Dict[str, Any], str]:
    """Validates mitigation flags against locked Phase 5 protocols and resolves the experiment arm."""
    clahe = bool(spec.clahe) if spec is not None else False
    median = bool(spec.median) if spec is not None else False
    morphology = bool(spec.morphology) if spec is not None else False

    # Research evaluate_test_pipeline treats preproc_mode ('none'|'clahe'|'median') as mutually exclusive
    if clahe and median:
        raise ApiError(
            status_code=422,
            code="UNSUPPORTED_MITIGATION_COMBINATION",
            message=(
                "Combining CLAHE and median filtering simultaneously is not supported by the "
                "research pipeline (preproc_mode must be 'none', 'clahe', or 'median')."
            ),
        )

    preproc_mode = "clahe" if clahe else ("median" if median else "none")
    params: Dict[str, Any] = {}
    if clahe:
        params["clahe"] = {
            "clip_limit": LOCKED_CLAHE_CLIP_LIMIT,
            "kernel_size": list(LOCKED_CLAHE_KERNEL_SIZE),
            "nbins": LOCKED_CLAHE_NBINS,
            "re_standardize": LOCKED_CLAHE_RE_STANDARDIZE,
        }
    if median:
        params["median"] = {
            "size": LOCKED_MEDIAN_FILTER_SIZE,
            "re_standardize": LOCKED_MEDIAN_RE_STANDARDIZE,
        }
    if morphology:
        params["morphology"] = {
            "enable_lcc": True,
            "enable_hole_filling": True,
            "min_component_sizes": dict(LOCKED_MORPHOLOGY_MIN_SIZES),
            "max_myo_hole_size": LOCKED_MORPHOLOGY_MAX_MYO_HOLE_SIZE,
        }

    arm = _classify_experiment_arm(
        corruption_family=corruption_family,
        preproc_mode=preproc_mode,
        morphology=morphology,
    )
    resolved = {
        "preprocessing_mode": preproc_mode,
        "clahe": clahe,
        "median": median,
        "morphology": morphology,
        "parameters": params,
    }
    return resolved, arm


def _classify_experiment_arm(
    corruption_family: str,
    preproc_mode: str,
    morphology: bool,
) -> str:
    """Identifies whether the configuration matches a registered Phase 5 arm or exploratory combo."""
    if preproc_mode == "none" and not morphology:
        return "unmitigated"
    if preproc_mode == "none" and morphology:
        return "postproc_alone"
    if preproc_mode == "clahe" and not morphology:
        return "clahe_alone"
    if preproc_mode == "median" and not morphology:
        if corruption_family in ("clean", "gaussian_noise"):
            return "median_alone"
        return "exploratory_median_off_target"
    if preproc_mode == "clahe" and morphology:
        if corruption_family in ("clean", "contrast_reduction"):
            return "combined_contrast"
        return "exploratory_clahe_postproc_off_target"
    if preproc_mode == "median" and morphology:
        if corruption_family in ("clean", "gaussian_noise"):
            return "combined_noise"
        return "exploratory_median_postproc_off_target"
    return "exploratory"
