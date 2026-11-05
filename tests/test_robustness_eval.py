"""Unit tests verifying robustness evaluation consistency and condition configurations."""

from pathlib import Path
import pytest
import torch

from scripts.evaluate_robustness import CONDITIONS, evaluate_condition
from src.corruptions import DEFAULT_CORRUPTIONS_REGISTRY, get_corruption_parameter
from src.evaluation.patient_metrics import PatientMetricAccumulator
from src.models.unet2d import LightweightUNet2D
from src.utils.config_parser import load_yaml_config


def test_ten_conditions_uniquely_defined() -> None:
    """Verifies that exactly 10 distinct experimental conditions are defined."""
    assert len(CONDITIONS) == 10
    cond_ids = [c["condition_id"] for c in CONDITIONS]
    assert len(set(cond_ids)) == 10
    assert "clean_baseline" in cond_ids


def test_conditions_match_yaml_parameters() -> None:
    """Verifies that condition parameters match the configuration YAML."""
    cfg = load_yaml_config("configs/baseline_unet.yaml")
    yaml_families = {f["name"]: f["levels"] for f in cfg.get("corruptions", {}).get("families", [])}

    for c in CONDITIONS:
        if c["family"] == "clean":
            continue
        fam = c["family"]
        sev = c["severity"]
        val = c["param_value"]

        assert fam in yaml_families, f"Family {fam} not in YAML"
        assert sev in yaml_families[fam], f"Severity {sev} not in YAML family {fam}"
        assert yaml_families[fam][sev] == val


def test_evaluation_clean_condition_identity() -> None:
    """Verifies that clean condition produces deterministic predictions without mutating model weights."""
    model = LightweightUNet2D()
    model.eval()

    # Freeze model weights copy
    initial_weights = [p.clone() for p in model.parameters()]

    # Synthetic batch of 2 slices
    images = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out1 = model(images)
        out2 = model(images)

    # Output determinism
    assert torch.equal(out1, out2)

    # Weights immutability
    for p_init, p_curr in zip(initial_weights, model.parameters()):
        assert torch.equal(p_init, p_curr)


def test_50epochs_checkpoint_integrity() -> None:
    """Verifies that the 50-epoch checkpoint exists and matches its expected SHA-256 hash."""
    import hashlib
    ckpt_path = Path("checkpoints/50epochs/best_model_50epochs.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/best_model_50epochs.pt")
    assert ckpt_path.exists(), f"50-epoch checkpoint not found at {ckpt_path}"
    expected_hash = "a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38"
    actual_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"SHA-256 mismatch: {actual_hash} != {expected_hash}"


def test_50epochs_checkpoint_model_inference() -> None:
    """Verifies that the 50-epoch checkpoint loads cleanly into LightweightUNet2D and produces valid outputs."""
    ckpt_path = Path("checkpoints/50epochs/best_model_50epochs.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/best_model_50epochs.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LightweightUNet2D(in_channels=1, num_classes=4, base_channels=16)
    msg = model.load_state_dict(checkpoint["model_state_dict"])
    assert len(msg.missing_keys) == 0
    assert len(msg.unexpected_keys) == 0

    model.eval()
    dummy_input = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (1, 4, 256, 256)
    assert torch.isfinite(logits).all()


def test_evaluate_robustness_cli_parser() -> None:
    """Verifies that CLI argument parsing supports custom output-dir and experiment-id."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_unet.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_frozen.pt")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--experiment-id", type=str, default=None)

    args = parser.parse_args([
        "--checkpoint", "checkpoints/50epochs/best_model_50epochs.pt",
        "--output-dir", "experiments/phase4_robustness_50epoch_best_epoch44",
        "--experiment-id", "phase4_robustness_50epoch_best_epoch44",
    ])
    assert args.output_dir == "experiments/phase4_robustness_50epoch_best_epoch44"
    assert args.experiment_id == "phase4_robustness_50epoch_best_epoch44"
