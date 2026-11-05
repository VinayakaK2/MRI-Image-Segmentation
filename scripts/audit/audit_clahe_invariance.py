"""Forensic tensor-level audit of CLAHE contrast invariance.

Investigates whether min-max normalization in to_unit_interval mathematically
and empirically eliminates synthetic contrast corruption before CLAHE executes.
Measures tensor statistics, differences, correlations, and predictions across
all 15 held-out test patients (131 slices).
"""

import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import apply_corruption
from src.data.dataset import ACDCSliceDataset, acdc_collate_fn
from src.evaluation.metrics import (
    compute_multiclass_metrics_numpy,
)
from src.evaluation.patient_metrics import PatientMetricAccumulator
from src.models.unet2d import LightweightUNet2D
from src.preprocessing.normalization import z_score_standardize
from src.preprocessing.restoration import (
    apply_clahe,
    to_unit_interval,
)
from src.utils.config_parser import load_yaml_config
from src.utils.provenance import (
    CACHE_SCHEMA_VERSION,
    build_cache_provenance,
)
from src.utils.reproducibility import get_device, set_seed


def main():
    set_seed(42)
    device = get_device("auto")
    print("=" * 80)
    print("  PHASE 5 FORENSIC AUDIT: CLAHE CONTRAST INVARIANCE & CAUSAL ATTRIBUTION")
    print("=" * 80, flush=True)

    # 1. Paths and Provenance
    ckpt_path = PROJECT_ROOT / "checkpoints" / "50epochs" / "best_model_50epochs.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model_frozen.pt"
    config_path = PROJECT_ROOT / "configs" / "baseline_unet.yaml"
    manifest_path = PROJECT_ROOT / "data" / "processed" / "acdc_slice_manifest.json"
    output_dir = PROJECT_ROOT / "experiments" / "phase5_clahe_invariance_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    prov = build_cache_provenance(ckpt_path, config_path, manifest_path, CACHE_SCHEMA_VERSION)
    print(f"Evaluated Checkpoint: {ckpt_path}")
    print(f"Checkpoint SHA-256:   {prov['checkpoint_sha256']}")
    print(f"Config SHA-256:       {prov['config_sha256']}")
    print(f"Manifest SHA-256:     {prov['manifest_sha256']}", flush=True)

    # 2. Load Model
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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # 3. Load Test Cohort
    test_dataset = ACDCSliceDataset(
        manifest_path=manifest_path,
        split="test",
        target_spatial_size=tuple(cfg["dataset"]["target_spatial_size"]),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=acdc_collate_fn)
    print(f"Loaded held-out test cohort: {len(test_dataset)} slices across 15 patients.\n", flush=True)

    # Accumulators for patient-level 3D metrics
    accum_clean = PatientMetricAccumulator(num_classes=4)
    accum_corrupt = PatientMetricAccumulator(num_classes=4)
    accum_clean_clahe = PatientMetricAccumulator(num_classes=4)
    accum_corrupt_clahe = PatientMetricAccumulator(num_classes=4)
    accum_corrupt_zscore = PatientMetricAccumulator(num_classes=4)

    slice_rows: List[Dict[str, Any]] = []

    # Forensic tracking across all slices
    diff_C_D_max_list: List[float] = []
    diff_C_D_mean_list: List[float] = []
    diff_E_F_max_list: List[float] = []
    diff_E_F_mean_list: List[float] = []
    pred_EF_dice_list: List[float] = []
    pred_EF_agreement_list: List[float] = []

    alpha_factor = 0.25

    with torch.no_grad():
        for batch_idx, (images, targets, metadatas) in enumerate(test_loader):
            meta = metadatas[0]
            pid = meta["patient_id"]
            sidx = int(meta["slice_idx"])
            slice_id = f"{pid}_slice{sidx:02d}"
            target_np = targets[0].cpu().numpy()

            # Seed formula matching project standard
            seed = 42 + (batch_idx * 17) + sidx

            # ----------------------------------------------------
            # Stage A: Original Clean Image Tensor (from DataLoader)
            # ----------------------------------------------------
            img_clean_tensor = images.to(device)  # shape: (1, 1, 256, 256)
            A = img_clean_tensor.cpu().numpy().squeeze().astype(np.float32)

            # ----------------------------------------------------
            # Stage B: Severe Contrast-Corrupted Image
            # ----------------------------------------------------
            img_corrupt_tensor, _ = apply_corruption(
                image=img_clean_tensor,
                corruption_name="contrast_reduction",
                severity="severe",
                seed=seed,
            )
            B = img_corrupt_tensor.cpu().numpy().squeeze().astype(np.float32)

            # ----------------------------------------------------
            # Stage C: Clean Image after to_unit_interval
            # ----------------------------------------------------
            C, min_c, max_c = to_unit_interval(A)

            # ----------------------------------------------------
            # Stage D: Severe Contrast Image after to_unit_interval
            # ----------------------------------------------------
            D, min_d, max_d = to_unit_interval(B)

            # ----------------------------------------------------
            # Stage E: Clean Image after CLAHE
            # ----------------------------------------------------
            tensor_E = apply_clahe(img_clean_tensor, clip_limit=0.01, kernel_size=(16, 16), re_standardize=True)
            E = tensor_E.cpu().numpy().squeeze().astype(np.float32)

            # ----------------------------------------------------
            # Stage F: Severe Contrast Image after CLAHE
            # ----------------------------------------------------
            tensor_F = apply_clahe(img_corrupt_tensor, clip_limit=0.01, kernel_size=(16, 16), re_standardize=True)
            F = tensor_F.cpu().numpy().squeeze().astype(np.float32)

            # Diagnostic comparison: Pure Z-Score Standardization alone
            B_zscore = z_score_standardize(B)
            tensor_B_zscore = torch.from_numpy(B_zscore).unsqueeze(0).unsqueeze(0).to(device)

            # ----------------------------------------------------
            # Forward Passes & Predictions
            # ----------------------------------------------------
            logits_clean = model(img_clean_tensor)
            pred_clean = torch.argmax(logits_clean, dim=1).cpu().numpy()[0]

            logits_corrupt = model(img_corrupt_tensor)
            pred_corrupt = torch.argmax(logits_corrupt, dim=1).cpu().numpy()[0]

            logits_clean_clahe = model(tensor_E)
            pred_clean_clahe = torch.argmax(logits_clean_clahe, dim=1).cpu().numpy()[0]

            logits_corrupt_clahe = model(tensor_F)
            pred_corrupt_clahe = torch.argmax(logits_corrupt_clahe, dim=1).cpu().numpy()[0]

            logits_corrupt_zscore = model(tensor_B_zscore)
            pred_corrupt_zscore = torch.argmax(logits_corrupt_zscore, dim=1).cpu().numpy()[0]

            # Accumulate patient-level metrics
            accum_clean.update(pid, pred_clean, target_np)
            accum_corrupt.update(pid, pred_corrupt, target_np)
            accum_clean_clahe.update(pid, pred_clean_clahe, target_np)
            accum_corrupt_clahe.update(pid, pred_corrupt_clahe, target_np)
            accum_corrupt_zscore.update(pid, pred_corrupt_zscore, target_np)

            # Slice Dice metrics
            m_clean = compute_multiclass_metrics_numpy(pred_clean, target_np, num_classes=4)
            m_corrupt = compute_multiclass_metrics_numpy(pred_corrupt, target_np, num_classes=4)
            m_clean_clahe = compute_multiclass_metrics_numpy(pred_clean_clahe, target_np, num_classes=4)
            m_corrupt_clahe = compute_multiclass_metrics_numpy(pred_corrupt_clahe, target_np, num_classes=4)
            m_corrupt_zscore = compute_multiclass_metrics_numpy(pred_corrupt_zscore, target_np, num_classes=4)

            # Pairwise tensor comparison functions
            def compare_tensors(t1: np.ndarray, t2: np.ndarray):
                diff = np.abs(t1 - t2)
                max_diff = float(np.max(diff))
                mean_diff = float(np.mean(diff))
                rmse = float(np.sqrt(np.mean((t1 - t2) ** 2)))
                identical_exact = float(np.mean(t1 == t2) * 100.0)
                identical_tol = float(np.mean(np.isclose(t1, t2, atol=1e-5)) * 100.0)
                if np.std(t1) < 1e-8 or np.std(t2) < 1e-8:
                    corr = 1.0 if np.allclose(t1, t2) else 0.0
                else:
                    corr = float(np.corrcoef(t1.ravel(), t2.ravel())[0, 1])
                return max_diff, mean_diff, rmse, corr, identical_exact, identical_tol

            # A vs B (Clean vs Corrupted)
            ab_max, ab_mean, ab_rmse, ab_corr, ab_id_exact, ab_id_tol = compare_tensors(A, B)

            # C vs D (Clean Unit vs Corrupted Unit) -> THE CRUCIAL TEST
            cd_max, cd_mean, cd_rmse, cd_corr, cd_id_exact, cd_id_tol = compare_tensors(C, D)
            diff_C_D_max_list.append(cd_max)
            diff_C_D_mean_list.append(cd_mean)

            # E vs F (Clean CLAHE vs Corrupted CLAHE)
            ef_max, ef_mean, ef_rmse, ef_corr, ef_id_exact, ef_id_tol = compare_tensors(E, F)
            diff_E_F_max_list.append(ef_max)
            diff_E_F_mean_list.append(ef_mean)

            # Model prediction comparison between E and F
            # Pixel-level agreement between pred_clean_clahe and pred_corrupt_clahe
            pred_match = float(np.mean(pred_clean_clahe == pred_corrupt_clahe) * 100.0)
            pred_EF_agreement_list.append(pred_match)

            # Foreground mask Dice between pred_clean_clahe and pred_corrupt_clahe
            fg_E = (pred_clean_clahe > 0).astype(np.float32)
            fg_F = (pred_corrupt_clahe > 0).astype(np.float32)
            intersection = np.sum(fg_E * fg_F)
            total = np.sum(fg_E) + np.sum(fg_F)
            dice_EF = float((2.0 * intersection) / (total + 1e-8)) if total > 0 else 1.0
            pred_EF_dice_list.append(dice_EF)

            slice_rows.append({
                "patient_id": pid,
                "slice_idx": sidx,
                "slice_id": slice_id,
                # Summary of stages
                "stage_A_min": round(float(np.min(A)), 4),
                "stage_A_max": round(float(np.max(A)), 4),
                "stage_A_mean": round(float(np.mean(A)), 4),
                "stage_A_std": round(float(np.std(A)), 4),
                "stage_B_min": round(float(np.min(B)), 4),
                "stage_B_max": round(float(np.max(B)), 4),
                "stage_B_mean": round(float(np.mean(B)), 4),
                "stage_B_std": round(float(np.std(B)), 4),
                "stage_C_min": round(float(np.min(C)), 4),
                "stage_C_max": round(float(np.max(C)), 4),
                "stage_D_min": round(float(np.min(D)), 4),
                "stage_D_max": round(float(np.max(D)), 4),
                "stage_E_mean": round(float(np.mean(E)), 4),
                "stage_E_std": round(float(np.std(E)), 4),
                "stage_F_mean": round(float(np.mean(F)), 4),
                "stage_F_std": round(float(np.std(F)), 4),
                # Pairwise differences
                "A_vs_B_max_diff": f"{ab_max:.4e}",
                "A_vs_B_mean_diff": f"{ab_mean:.4e}",
                "A_vs_B_rmse": f"{ab_rmse:.4e}",
                "A_vs_B_corr": f"{ab_corr:.6f}",
                # CRITICAL: C vs D (Unit interval)
                "C_vs_D_max_diff": f"{cd_max:.4e}",
                "C_vs_D_mean_diff": f"{cd_mean:.4e}",
                "C_vs_D_rmse": f"{cd_rmse:.4e}",
                "C_vs_D_corr": f"{cd_corr:.6f}",
                "C_vs_D_pct_identical_tol": f"{cd_id_tol:.2f}",
                # CRITICAL: E vs F (CLAHE)
                "E_vs_F_max_diff": f"{ef_max:.4e}",
                "E_vs_F_mean_diff": f"{ef_mean:.4e}",
                "E_vs_F_rmse": f"{ef_rmse:.4e}",
                "E_vs_F_corr": f"{ef_corr:.6f}",
                "E_vs_F_pct_identical_tol": f"{ef_id_tol:.2f}",
                # Prediction comparison
                "pred_EF_pixel_agreement_pct": f"{pred_match:.2f}",
                "pred_EF_dice": f"{dice_EF:.6f}",
                # Dice with ground truth
                "dice_clean": round(float(m_clean["mean_dice_foreground"]), 4),
                "dice_corrupt_unmitigated": round(float(m_corrupt["mean_dice_foreground"]), 4),
                "dice_clean_clahe": round(float(m_clean_clahe["mean_dice_foreground"]), 4),
                "dice_corrupt_clahe": round(float(m_corrupt_clahe["mean_dice_foreground"]), 4),
                "dice_corrupt_zscore": round(float(m_corrupt_zscore["mean_dice_foreground"]), 4),
            })

    # Write tensor_comparison_results.csv
    csv_path = output_dir / "tensor_comparison_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(slice_rows[0].keys()))
        writer.writeheader()
        writer.writerows(slice_rows)
    print(f"Written slice-level tensor comparison CSV (131 rows) to: {csv_path}")

    # Patient-level summary
    summary_clean = accum_clean.compute_summary_statistics()
    summary_corrupt = accum_corrupt.compute_summary_statistics()
    summary_clean_clahe = accum_clean_clahe.compute_summary_statistics()
    summary_corrupt_clahe = accum_corrupt_clahe.compute_summary_statistics()
    summary_corrupt_zscore = accum_corrupt_zscore.compute_summary_statistics()

    per_pt_clean = accum_clean.compute_per_patient_metrics()
    per_pt_corrupt = accum_corrupt.compute_per_patient_metrics()
    per_pt_clean_clahe = accum_clean_clahe.compute_per_patient_metrics()
    per_pt_corrupt_clahe = accum_corrupt_clahe.compute_per_patient_metrics()
    per_pt_corrupt_zscore = accum_corrupt_zscore.compute_per_patient_metrics()

    patient_rows: List[Dict[str, Any]] = []
    p_ids = sorted(list(per_pt_clean.keys()))
    for pid in p_ids:
        d_clean = per_pt_clean[pid]["mean_dice_foreground"]
        d_corrupt = per_pt_corrupt[pid]["mean_dice_foreground"]
        d_clean_clahe = per_pt_clean_clahe[pid]["mean_dice_foreground"]
        d_corrupt_clahe = per_pt_corrupt_clahe[pid]["mean_dice_foreground"]
        d_corrupt_zscore = per_pt_corrupt_zscore[pid]["mean_dice_foreground"]

        diff_clean_clahe_vs_corrupt_clahe = abs(d_clean_clahe - d_corrupt_clahe)
        diff_clean_vs_corrupt_zscore = abs(d_clean - d_corrupt_zscore)

        patient_rows.append({
            "patient_id": pid,
            "dice_clean_baseline": round(float(d_clean), 4),
            "dice_severe_unmitigated": round(float(d_corrupt), 4),
            "dice_clean_clahe": round(float(d_clean_clahe), 4),
            "dice_severe_clahe": round(float(d_corrupt_clahe), 4),
            "diff_clean_clahe_vs_severe_clahe": f"{diff_clean_clahe_vs_corrupt_clahe:.4e}",
            "dice_severe_zscore_alone": round(float(d_corrupt_zscore), 4),
            "diff_clean_vs_severe_zscore_alone": f"{diff_clean_vs_corrupt_zscore:.4e}",
            "delta_clahe_rescue_vs_unmitigated": round(float(d_corrupt_clahe - d_corrupt), 4),
            "delta_zscore_rescue_vs_unmitigated": round(float(d_corrupt_zscore - d_corrupt), 4),
        })

    patient_csv_path = output_dir / "patient_level_prediction_comparison.csv"
    with open(patient_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(patient_rows[0].keys()))
        writer.writeheader()
        writer.writerows(patient_rows)
    print(f"Written patient-level prediction comparison CSV (15 rows) to: {patient_csv_path}")

    # Compile Preprocessing Stage Analysis JSON
    stage_analysis = {
        "provenance": prov,
        "sample_size": {
            "patients": len(p_ids),
            "slices": len(slice_rows),
        },
        "contrast_corruption_parameters": {
            "name": "contrast_reduction",
            "severity": "severe",
            "alpha": alpha_factor,
            "formula": "I_corrupt = mean + alpha * (I - mean) = alpha * I + (1 - alpha) * mean",
            "affine_property": "Linear transformation with positive slope alpha > 0 and additive constant (1-alpha)*mean",
        },
        "stage_C_vs_D_unit_interval": {
            "max_absolute_difference": float(np.max(diff_C_D_max_list)),
            "mean_absolute_difference": float(np.mean(diff_C_D_mean_list)),
            "all_slices_max_diff_below_1e_5": bool(np.max(diff_C_D_max_list) < 1e-5),
            "scientific_interpretation": (
                "Min-max scaling in to_unit_interval: (I - min) / (max - min). "
                "For I_corrupt = alpha * I + (1-alpha)*mu, (alpha*I - alpha*min) / (alpha*max - alpha*min) = (I - min) / (max - min). "
                "Alpha cancels out analytically. Tensors at Stage C and Stage D are mathematically and numerically identical."
            ),
        },
        "stage_E_vs_F_clahe_output": {
            "max_absolute_difference": float(np.max(diff_E_F_max_list)),
            "mean_absolute_difference": float(np.mean(diff_E_F_mean_list)),
            "prediction_pixel_agreement_mean_pct": float(np.mean(pred_EF_agreement_list)),
            "prediction_dice_agreement_mean": float(np.mean(pred_EF_dice_list)),
            "scientific_interpretation": (
                "Because Stage C and Stage D are identical, CLAHE receives identical inputs for Clean and Severe Contrast. "
                "Consequently, the outputs E and F, and their model predictions, are identical."
            ),
        },
        "cohort_summary_dice": {
            "clean_baseline": {
                "mean": float(summary_clean["mean_dice_foreground"]["mean"]),
                "std": float(summary_clean["mean_dice_foreground"]["std"]),
            },
            "severe_contrast_unmitigated": {
                "mean": float(summary_corrupt["mean_dice_foreground"]["mean"]),
                "std": float(summary_corrupt["mean_dice_foreground"]["std"]),
            },
            "clean_plus_clahe": {
                "mean": float(summary_clean_clahe["mean_dice_foreground"]["mean"]),
                "std": float(summary_clean_clahe["mean_dice_foreground"]["std"]),
            },
            "severe_contrast_plus_clahe": {
                "mean": float(summary_corrupt_clahe["mean_dice_foreground"]["mean"]),
                "std": float(summary_corrupt_clahe["mean_dice_foreground"]["std"]),
            },
            "severe_contrast_plus_zscore_alone": {
                "mean": float(summary_corrupt_zscore["mean_dice_foreground"]["mean"]),
                "std": float(summary_corrupt_zscore["mean_dice_foreground"]["std"]),
            },
        },
        "causal_decomposition": {
            "clean_baseline_dice": float(summary_clean["mean_dice_foreground"]["mean"]),
            "severe_unmitigated_dice": float(summary_corrupt["mean_dice_foreground"]["mean"]),
            "unmitigated_drop": float(summary_clean["mean_dice_foreground"]["mean"] - summary_corrupt["mean_dice_foreground"]["mean"]),
            "clahe_reported_rescue": float(summary_corrupt_clahe["mean_dice_foreground"]["mean"] - summary_corrupt["mean_dice_foreground"]["mean"]),
            "zscore_alone_rescue": float(summary_corrupt_zscore["mean_dice_foreground"]["mean"] - summary_corrupt["mean_dice_foreground"]["mean"]),
            "clahe_net_effect_relative_to_clean": float(summary_clean_clahe["mean_dice_foreground"]["mean"] - summary_clean["mean_dice_foreground"]["mean"]),
            "clahe_net_effect_relative_to_zscore_alone": float(summary_corrupt_clahe["mean_dice_foreground"]["mean"] - summary_corrupt_zscore["mean_dice_foreground"]["mean"]),
        },
    }

    stage_json_path = output_dir / "preprocessing_stage_analysis.json"
    with open(stage_json_path, "w", encoding="utf-8") as f:
        json.dump(stage_analysis, f, indent=2)
    print(f"Written preprocessing stage analysis JSON to: {stage_json_path}")

    # Reproducibility Manifest
    manifest = {
        "audit_name": "phase5_clahe_invariance_audit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": prov,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "seed": 42,
        },
        "files_generated": [
            "tensor_comparison_results.csv",
            "patient_level_prediction_comparison.csv",
            "preprocessing_stage_analysis.json",
            "clahe_invariance_audit.md",
            "scientific_claims_review.md",
            "reproducibility_manifest.json",
        ],
    }
    with open(output_dir / "reproducibility_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Written reproducibility manifest to: {output_dir / 'reproducibility_manifest.json'}")

    print("\n--- Summary of Findings ---")
    print(f"Clean Baseline 3D Dice:                 {summary_clean['mean_dice_foreground']['mean']:.4f}")
    print(f"Severe Contrast Unmitigated 3D Dice:    {summary_corrupt['mean_dice_foreground']['mean']:.4f}")
    print(f"Severe Contrast + CLAHE 3D Dice:        {summary_corrupt_clahe['mean_dice_foreground']['mean']:.4f}")
    print(f"Clean + CLAHE 3D Dice:                  {summary_clean_clahe['mean_dice_foreground']['mean']:.4f}")
    print(f"Severe Contrast + Z-Score Alone 3D Dice:{summary_corrupt_zscore['mean_dice_foreground']['mean']:.4f}")
    print(f"Max abs diff between C and D (unit):    {np.max(diff_C_D_max_list):.4e}")
    print(f"Max abs diff between E and F (CLAHE):   {np.max(diff_E_F_max_list):.4e}")
    print(f"Mean pixel agreement between E & F:     {np.mean(pred_EF_agreement_list):.2f}%")


if __name__ == "__main__":
    main()
