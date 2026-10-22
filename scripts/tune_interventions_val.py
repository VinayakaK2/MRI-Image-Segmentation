"""Validation cohort hyperparameter tuning for inference-time interventions.

Evaluates candidates on the 15-patient validation split (142 slices) to select
optimal preprocessing filters and post-processing safeguards before testing.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import apply_corruption
from src.data.dataset import ACDCSliceDataset, acdc_collate_fn
from src.evaluation.metrics import compute_multiclass_metrics_numpy
from src.evaluation.patient_metrics import PatientMetricAccumulator
from src.models.unet2d import LightweightUNet2D
from src.postprocessing.morphology import postprocess_cardiac_segmentation
from src.preprocessing.restoration import (
    apply_clahe,
    apply_gaussian_denoising,
    apply_histogram_equalization,
    apply_median_denoising,
)
from src.utils.config_parser import load_yaml_config
from src.utils.reproducibility import get_device, set_seed


def evaluate_val_candidate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corruption_family: str,
    corruption_severity: str,
    preprocessing_fn=None,
    preprocessing_kwargs: Dict[str, Any] = None,
    apply_postprocessing: bool = False,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates a single intervention configuration on the validation cohort."""
    if preprocessing_kwargs is None:
        preprocessing_kwargs = {}

    patient_acc = PatientMetricAccumulator(num_classes=4)
    slice_dices = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets, metadatas) in enumerate(loader):
            image = images.to(device)
            target_np = targets[0].cpu().numpy()
            meta = metadatas[0]
            patient_id = meta["patient_id"]
            slice_idx = meta["slice_idx"]

            # Deterministic per-slice corruption seed
            seed = base_seed + (batch_idx * 17) + int(slice_idx)

            # 1. Apply corruption
            if corruption_family != "clean":
                corrupt_img, _ = apply_corruption(
                    image=image,
                    corruption_name=corruption_family,
                    severity=corruption_severity,
                    seed=seed,
                )
            else:
                corrupt_img = image

            # 2. Apply preprocessing intervention (if specified)
            if preprocessing_fn is not None:
                proc_img = preprocessing_fn(corrupt_img, **preprocessing_kwargs)
            else:
                proc_img = corrupt_img

            # 3. Model forward pass
            logits = model(proc_img)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            # 4. Apply post-processing intervention (if specified)
            if apply_postprocessing:
                pred = postprocess_cardiac_segmentation(pred)

            # Compute slice metrics
            metrics = compute_multiclass_metrics_numpy(pred, target_np, num_classes=4)
            slice_dices.append(metrics["mean_dice_foreground"])
            patient_acc.update(patient_id, pred, target_np)

    patient_summary = patient_acc.compute_summary_statistics()
    return {
        "slice_fg_dice": float(np.mean(slice_dices)),
        "patient_3d_fg_dice_mean": float(patient_summary["mean_dice_foreground"]["mean"]),
        "patient_3d_fg_dice_std": float(patient_summary["mean_dice_foreground"]["std"]),
        "patient_3d_rv_dice": float(patient_summary.get("dice_rv", {}).get("mean", 0.0)),
        "patient_3d_myo_dice": float(patient_summary.get("dice_myo", {}).get("mean", 0.0)),
        "patient_3d_lv_dice": float(patient_summary.get("dice_lv", {}).get("mean", 0.0)),
    }


def main():
    set_seed(42)
    device = get_device("auto")
    print(f"Executing Validation Interventions Tuning on device: {device}")

    # Load frozen baseline checkpoint
    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model_frozen.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"

    config_path = PROJECT_ROOT / "configs" / "baseline_unet.yaml"
    cfg = load_yaml_config(config_path)

    model = LightweightUNet2D(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
        dropout=cfg["model"]["dropout"],
        use_batchnorm=cfg["model"]["use_batchnorm"],
        bilinear_upsample=cfg["model"]["bilinear_upsample"],
    ).to(device)

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Load validation split
    manifest_path = PROJECT_ROOT / "data" / "processed" / "acdc_slice_manifest.json"
    val_dataset = ACDCSliceDataset(
        manifest_path=manifest_path,
        split="val",
        target_spatial_size=tuple(cfg["dataset"]["target_spatial_size"]),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=acdc_collate_fn)
    print(f"Loaded validation split: {len(val_dataset)} slices across 15 patients.")

    results: Dict[str, Any] = {}

    print("\n" + "=" * 75)
    print(" 1. CONTRAST RESTORATION TUNING (Under Severe Contrast: alpha=0.25)")
    print("=" * 75)

    contrast_candidates = [
        ("Unmitigated Corrupted", None, {}),
        ("Global HE (nbins=256)", apply_histogram_equalization, {"nbins": 256}),
        ("Global HE (nbins=128)", apply_histogram_equalization, {"nbins": 128}),
        ("CLAHE (clip=0.01, grid=16x16)", apply_clahe, {"clip_limit": 0.01, "kernel_size": (16, 16)}),
        ("CLAHE (clip=0.02, grid=16x16)", apply_clahe, {"clip_limit": 0.02, "kernel_size": (16, 16)}),
        ("CLAHE (clip=0.04, grid=16x16)", apply_clahe, {"clip_limit": 0.04, "kernel_size": (16, 16)}),
        ("CLAHE (clip=0.02, grid=8x8)", apply_clahe, {"clip_limit": 0.02, "kernel_size": (8, 8)}),
    ]

    contrast_results = {}
    for name, fn, kwargs in contrast_candidates:
        t0 = time.time()
        res = evaluate_val_candidate(
            model=model,
            loader=val_loader,
            device=device,
            corruption_family="contrast_reduction",
            corruption_severity="severe",
            preprocessing_fn=fn,
            preprocessing_kwargs=kwargs,
        )
        dt = time.time() - t0
        contrast_results[name] = res
        print(f"  [{name:<32}] -> Slice FG Dice: {res['slice_fg_dice']:.4f} | Patient 3D Dice: {res['patient_3d_fg_dice_mean']:.4f} +/- {res['patient_3d_fg_dice_std']:.4f} (RV: {res['patient_3d_rv_dice']:.3f}, MYO: {res['patient_3d_myo_dice']:.3f}, LV: {res['patient_3d_lv_dice']:.3f}) [{dt:.1f}s]", flush=True)

    results["contrast_tuning_severe"] = contrast_results

    print("\n" + "=" * 75)
    print(" 2. DENOISING TUNING (Under Severe Gaussian Noise: std=0.10)")
    print("=" * 75)

    noise_candidates = [
        ("Unmitigated Corrupted", None, {}),
        ("Gaussian Filter (sigma=0.5)", apply_gaussian_denoising, {"sigma": 0.5}),
        ("Gaussian Filter (sigma=1.0)", apply_gaussian_denoising, {"sigma": 1.0}),
        ("Gaussian Filter (sigma=1.5)", apply_gaussian_denoising, {"sigma": 1.5}),
        ("Median Filter (size=3)", apply_median_denoising, {"size": 3}),
        ("Median Filter (size=5)", apply_median_denoising, {"size": 5}),
    ]

    noise_results = {}
    for name, fn, kwargs in noise_candidates:
        t0 = time.time()
        res = evaluate_val_candidate(
            model=model,
            loader=val_loader,
            device=device,
            corruption_family="gaussian_noise",
            corruption_severity="severe",
            preprocessing_fn=fn,
            preprocessing_kwargs=kwargs,
        )
        dt = time.time() - t0
        noise_results[name] = res
        print(f"  [{name:<32}] -> Slice FG Dice: {res['slice_fg_dice']:.4f} | Patient 3D Dice: {res['patient_3d_fg_dice_mean']:.4f} +/- {res['patient_3d_fg_dice_std']:.4f} (RV: {res['patient_3d_rv_dice']:.3f}, MYO: {res['patient_3d_myo_dice']:.3f}, LV: {res['patient_3d_lv_dice']:.3f}) [{dt:.1f}s]", flush=True)

    results["noise_tuning_severe"] = noise_results

    print("\n" + "=" * 75)
    print(" 3. POST-PROCESSING EVALUATION (On Clean Baseline & Corrupted Conditions)")
    print("=" * 75)

    postproc_scenarios = [
        ("Clean Baseline", "clean", "none"),
        ("Gaussian Noise (Severe)", "gaussian_noise", "severe"),
        ("Gaussian Blur (Severe)", "gaussian_blur", "severe"),
        ("Contrast Reduction (Mod)", "contrast_reduction", "moderate"),
    ]

    postproc_results = {}
    for sc_name, fam, sev in postproc_scenarios:
        res_raw = evaluate_val_candidate(
            model=model, loader=val_loader, device=device, corruption_family=fam, corruption_severity=sev, apply_postprocessing=False
        )
        res_post = evaluate_val_candidate(
            model=model, loader=val_loader, device=device, corruption_family=fam, corruption_severity=sev, apply_postprocessing=True
        )
        diff = res_post["patient_3d_fg_dice_mean"] - res_raw["patient_3d_fg_dice_mean"]
        postproc_results[sc_name] = {"raw": res_raw, "postprocessed": res_post, "delta_patient_dice": diff}
        print(f"  [{sc_name:<26}] Raw: {res_raw['patient_3d_fg_dice_mean']:.4f} -> Postprocessed: {res_post['patient_3d_fg_dice_mean']:.4f} (Delta: {diff:+.4f})", flush=True)

    results["postprocessing_val"] = postproc_results

    print("\n" + "=" * 75)
    print(" 4. SIDE-EFFECT CHECK: PREPROCESSING APPLIED TO CLEAN BASELINE")
    print("=" * 75)

    clean_side_effects = [
        ("Clean Baseline (No Preproc)", None, {}),
        ("Clean + CLAHE (clip=0.01)", apply_clahe, {"clip_limit": 0.01, "kernel_size": (16, 16)}),
        ("Clean + CLAHE (clip=0.02)", apply_clahe, {"clip_limit": 0.02, "kernel_size": (16, 16)}),
        ("Clean + Global HE", apply_histogram_equalization, {"nbins": 256}),
        ("Clean + Gaussian Filter (sigma=0.5)", apply_gaussian_denoising, {"sigma": 0.5}),
        ("Clean + Median Filter (size=3)", apply_median_denoising, {"size": 3}),
    ]

    side_effect_results = {}
    for name, fn, kwargs in clean_side_effects:
        res = evaluate_val_candidate(
            model=model, loader=val_loader, device=device, corruption_family="clean", corruption_severity="none", preprocessing_fn=fn, preprocessing_kwargs=kwargs
        )
        side_effect_results[name] = res
        print(f"  [{name:<36}] -> Patient 3D Dice: {res['patient_3d_fg_dice_mean']:.4f} (RV: {res['patient_3d_rv_dice']:.3f}, MYO: {res['patient_3d_myo_dice']:.3f}, LV: {res['patient_3d_lv_dice']:.3f})")

    results["clean_side_effects"] = side_effect_results

    # Save validation tuning results
    val_out_path = PROJECT_ROOT / "reports" / "validation_tuning_results.json"
    with open(val_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation tuning results to: {val_out_path}")


if __name__ == "__main__":
    main()
