"""Safe checkpoint verification and loading utilities for LightweightUNet2D."""

from pathlib import Path
from typing import Any, Dict, Tuple
import torch

from api.config import CHECKPOINT_REGISTRY, CheckpointSpec
from api.utils.validation import ApiError
from src.models.unet2d import LightweightUNet2D
from src.utils.provenance import compute_file_sha256


def get_checkpoint_spec(checkpoint_id: str) -> CheckpointSpec:
    """Returns the registered CheckpointSpec or raises a 400 ApiError for unknown IDs."""
    spec = CHECKPOINT_REGISTRY.get(checkpoint_id)
    if spec is None:
        raise ApiError(
            status_code=400,
            code="INVALID_CHECKPOINT_ID",
            message=f"Checkpoint '{checkpoint_id}' is not registered.",
            details={"allowed_checkpoint_ids": list(CHECKPOINT_REGISTRY.keys())},
        )
    return spec


def verify_checkpoint_integrity(spec: CheckpointSpec, project_root: Path) -> Tuple[Path, str]:
    """Resolves a registered checkpoint path safely and verifies its SHA-256 digest."""
    ckpt_path = (project_root / spec.relative_path).resolve()
    try:
        ckpt_path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ApiError(
            status_code=500,
            code="CHECKPOINT_PATH_ERROR",
            message="Registered checkpoint path failed boundary verification.",
        ) from exc

    if not ckpt_path.is_file():
        raise ApiError(
            status_code=503,
            code="MODEL_NOT_FOUND",
            message=f"Registered checkpoint '{spec.checkpoint_id}' is unavailable on disk.",
        )

    actual_sha256 = compute_file_sha256(ckpt_path)
    if actual_sha256 != spec.expected_sha256:
        raise ApiError(
            status_code=500,
            code="CHECKPOINT_INTEGRITY_ERROR",
            message=f"SHA-256 verification failed for checkpoint '{spec.checkpoint_id}'.",
            details={
                "expected_sha256": spec.expected_sha256,
                "actual_sha256": actual_sha256,
            },
        )
    return ckpt_path, actual_sha256


def load_verified_model(
    spec: CheckpointSpec,
    project_root: Path,
    model_config: Dict[str, Any],
    device: torch.device,
) -> Tuple[LightweightUNet2D, str, Dict[str, Any]]:
    """Verifies SHA-256 and loads a checkpoint strictly with weights_only=True (no unsafe fallback)."""
    ckpt_path, actual_sha256 = verify_checkpoint_integrity(spec, project_root)

    try:
        # Allowlist PyTorch's TorchVersion metadata class so weights_only=True loads safely
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ApiError(
            status_code=500,
            code="MODEL_LOAD_FAILED",
            message=(
                f"Safe weights_only=True loading failed for checkpoint '{spec.checkpoint_id}': "
                f"{type(exc).__name__}"
            ),
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ApiError(
            status_code=500,
            code="MODEL_LOAD_FAILED",
            message=f"Checkpoint '{spec.checkpoint_id}' has unexpected payload format.",
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ApiError(
            status_code=500,
            code="MODEL_LOAD_FAILED",
            message=f"Checkpoint '{spec.checkpoint_id}' is missing a valid model_state_dict.",
        )

    model = LightweightUNet2D.from_config(model_config).to(device)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ApiError(
            status_code=500,
            code="MODEL_LOAD_FAILED",
            message=f"Checkpoint '{spec.checkpoint_id}' weights do not match LightweightUNet2D.",
        )

    model.eval()
    metadata = {
        "epoch": checkpoint.get("epoch", spec.best_epoch),
        "best_epoch": checkpoint.get("best_epoch", spec.best_epoch),
        "best_val_dice": checkpoint.get("best_val_dice"),
        "total_completed_epochs": checkpoint.get("total_completed_epochs", spec.epochs_completed),
        "architecture": checkpoint.get("architecture", "LightweightUNet2D"),
    }
    return model, actual_sha256, metadata
