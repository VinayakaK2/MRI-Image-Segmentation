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
