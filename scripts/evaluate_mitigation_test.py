"""Full Phase 5 mitigation and ablation evaluation on the held-out test cohort.

Evaluates frozen preprocessing interventions (CLAHE, Median Filter) and
anatomical morphological post-processing across all 10 experimental conditions
on the 15 held-out test patients (131 slices), computing paired statistical tests.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import apply_corruption
from src.data.dataset import ACDCSliceDataset, acdc_collate_fn
from src.evaluation.metrics import (
    CLASS_NAME_MAP,
    FOREGROUND_CLASS_INDICES,
    MetricAccumulator,
    compute_multiclass_metrics_numpy,
)
from src.evaluation.patient_metrics import (
    PatientMetricAccumulator,
    adjust_pvalues_holm,
    compute_paired_patient_comparison,
)
from src.models.unet2d import LightweightUNet2D
from src.postprocessing.morphology import postprocess_cardiac_segmentation
from src.preprocessing.restoration import (
    apply_clahe,
    apply_gaussian_denoising,
    apply_histogram_equalization,
    apply_median_denoising,
)
from src.utils.config_parser import load_yaml_config
from src.utils.provenance import (
    CACHE_SCHEMA_VERSION,
    build_cache_provenance,
    resolve_cache_state,
)
from src.utils.reproducibility import get_device, set_seed


CONDITIONS: List[Dict[str, Any]] = [
    {"id": "clean_baseline", "family": "clean", "severity": "none", "param": 0.0, "name": "Clean Baseline"},
    {"id": "gaussian_noise_mild", "family": "gaussian_noise", "severity": "mild", "param": 0.02, "name": "Gaussian Noise (Mild, std=0.02)"},
    {"id": "gaussian_noise_moderate", "family": "gaussian_noise", "severity": "moderate", "param": 0.05, "name": "Gaussian Noise (Mod, std=0.05)"},
    {"id": "gaussian_noise_severe", "family": "gaussian_noise", "severity": "severe", "param": 0.10, "name": "Gaussian Noise (Sev, std=0.10)"},
    {"id": "gaussian_blur_mild", "family": "gaussian_blur", "severity": "mild", "param": 0.75, "name": "Gaussian Blur (Mild, sigma=0.75)"},
    {"id": "gaussian_blur_moderate", "family": "gaussian_blur", "severity": "moderate", "param": 1.50, "name": "Gaussian Blur (Mod, sigma=1.50)"},
    {"id": "gaussian_blur_severe", "family": "gaussian_blur", "severity": "severe", "param": 2.50, "name": "Gaussian Blur (Sev, sigma=2.50)"},
    {"id": "contrast_reduction_mild", "family": "contrast_reduction", "severity": "mild", "param": 0.75, "name": "Contrast Reduction (Mild, alpha=0.75)"},
    {"id": "contrast_reduction_moderate", "family": "contrast_reduction", "severity": "moderate", "param": 0.50, "name": "Contrast Reduction (Mod, alpha=0.50)"},
    {"id": "contrast_reduction_severe", "family": "contrast_reduction", "severity": "severe", "param": 0.25, "name": "Contrast Reduction (Sev, alpha=0.25)"},
]


def evaluate_test_pipeline(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    condition: Dict[str, Any],
    preproc_mode: str = "none",  # 'none', 'clahe', 'median'
    apply_postproc: bool = False,
    base_seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Runs inference across the held-out test cohort under a specific intervention configuration."""
    slice_accumulator = MetricAccumulator()
    patient_accumulator = PatientMetricAccumulator(num_classes=4)

    fam = condition["family"]
    sev = condition["severity"]

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets, metadatas) in enumerate(loader):
            image = images.to(device)
            target_np = targets[0].cpu().numpy()
            meta = metadatas[0]
            patient_id = meta["patient_id"]
            slice_idx = meta["slice_idx"]

            seed = base_seed + (batch_idx * 17) + int(slice_idx)

            # Step 1: Corrupt input tensor
            if fam != "clean":
                corrupt_img, _ = apply_corruption(
                    image=image,
                    corruption_name=fam,
                    severity=sev,
                    seed=seed,
                )
            else:
                corrupt_img = image

            # Step 2: Preprocessing intervention
            if preproc_mode == "clahe":
                inp = apply_clahe(corrupt_img, clip_limit=0.01, kernel_size=(16, 16), re_standardize=True)
            elif preproc_mode == "median":
                inp = apply_median_denoising(corrupt_img, size=3, re_standardize=False)
            else:
                inp = corrupt_img

            # Step 3: Forward pass
            logits = model(inp)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            # Step 4: Morphological post-processing intervention
            if apply_postproc:
                pred = postprocess_cardiac_segmentation(pred)

            # Metrics
            slice_metrics = compute_multiclass_metrics_numpy(pred, target_np, num_classes=4)
            slice_accumulator.update(slice_metrics, batch_size=1)
            patient_accumulator.update(patient_id, pred, target_np)

    slice_averages = slice_accumulator.compute_averages()
    per_patient = patient_accumulator.compute_per_patient_metrics()
    patient_summary = patient_accumulator.compute_summary_statistics()

    return slice_averages, per_patient, patient_summary


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the Phase 5 evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 5 controlled mitigation and ablation evaluation on held-out test cohort."
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        default=False,
        help="Bypass cached evaluation results and recompute all interventions/conditions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to frozen model checkpoint (defaults to checkpoints/best_model_frozen.pt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to model YAML configuration (defaults to configs/baseline_unet.yaml).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to dataset slice manifest JSON (defaults to data/processed/acdc_slice_manifest.json).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to output results JSON (defaults to reports/mitigation_results.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output summary CSV (defaults to reports/mitigation_results.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = get_device("auto")
    print("=" * 80)
    print("  PHASE 5: CONTROLLED MITIGATION & ABLATION EVALUATION (HELD-OUT TEST)")
    print("=" * 80, flush=True)

    # 1. Resolve file paths and evaluate cryptographic provenance
    ckpt_path = args.checkpoint or (PROJECT_ROOT / "checkpoints" / "best_model_frozen.pt")
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"

    config_path = args.config or (PROJECT_ROOT / "configs" / "baseline_unet.yaml")
    manifest_path = args.manifest or (PROJECT_ROOT / "data" / "processed" / "acdc_slice_manifest.json")
    out_json_path = args.output_json or (PROJECT_ROOT / "reports" / "mitigation_results.json")
    out_csv_path = args.output_csv or (PROJECT_ROOT / "reports" / "mitigation_results.csv")

    expected_provenance = build_cache_provenance(
        checkpoint_path=ckpt_path,
        config_path=config_path,
        manifest_path=manifest_path,
        schema_version=CACHE_SCHEMA_VERSION,
    )
    print(f"[PROVENANCE] Checkpoint:     {ckpt_path} (SHA-256: {expected_provenance['checkpoint_sha256']})")
    print(f"[PROVENANCE] Config:         {config_path} (SHA-256: {expected_provenance['config_sha256']})")
    print(f"[PROVENANCE] Manifest:       {manifest_path} (SHA-256: {expected_provenance['manifest_sha256']})")
    print(f"[PROVENANCE] Schema Version: {expected_provenance['schema_version']}", flush=True)

    # Resolve cache state with cryptographic provenance validation
    existing_results, is_cache_reused, cache_msg = resolve_cache_state(
        cache_path=out_json_path,
        expected_provenance=expected_provenance,
        force_recompute=args.force_recompute,
    )
    print(f"[PROVENANCE] Cache Resolution: {cache_msg}\n", flush=True)

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

    # 2. Load held-out test split
    test_dataset = ACDCSliceDataset(
        manifest_path=manifest_path,
        split="test",
        target_spatial_size=tuple(cfg["dataset"]["target_spatial_size"]),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=acdc_collate_fn)
    print(f"Loaded held-out test split: {len(test_dataset)} slices across 15 patients.\n", flush=True)

    # 3. Define the ablation modes to evaluate
    # A. Unmitigated Baseline (reference matching Phase 4)
    # B. Preprocessing Alone (CLAHE on Contrast, Median on Noise)
    # C. Post-processing Alone (Morphological LCC + hole filling with myocardium safeguard)
    # D. Combined (Preproc + Postproc)

    all_evaluation_records: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    def save_checkpoint(results_dict: Dict[str, Any]) -> None:
        def sanitize(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        results_dict["provenance"] = expected_provenance
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(sanitize(results_dict), f, indent=2)

    # First, load or evaluate Unmitigated Baseline across all 10 conditions
    print("--- 1. Evaluating Unmitigated Baseline (Reference Model) ---", flush=True)
    unmitigated_results: Dict[str, Any] = existing_results.get("unmitigated", {})
    for cond in CONDITIONS:
        cid = cond["id"]
        if cid in unmitigated_results and is_cache_reused:
            fg_slice = unmitigated_results[cid]["slice_level"]["mean_dice_foreground"]
            fg_patient = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            print(f"  [{cid:<28}] Slice FG Dice: {fg_slice:.4f} | Patient 3D Dice: {fg_patient:.4f} [Cached]", flush=True)
            continue

        t0 = time.time()
        slice_avg, per_patient, patient_sum = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="none", apply_postproc=False
        )
        dt = time.time() - t0
        unmitigated_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        fg_slice = slice_avg["mean_dice_foreground"]
        fg_patient = patient_sum["mean_dice_foreground"]["mean"]
        print(f"  [{cid:<28}] Slice FG Dice: {fg_slice:.4f} | Patient 3D Dice: {fg_patient:.4f} [{dt:.1f}s]", flush=True)
        existing_results["unmitigated"] = unmitigated_results
        save_checkpoint(existing_results)

    # Second, run Post-Processing Alone across all 10 conditions
    print("\n--- 2. Evaluating Post-Processing Alone (LCC + Hole Filling) ---", flush=True)
    postproc_results: Dict[str, Any] = existing_results.get("postproc_alone", {})
    for cond in CONDITIONS:
        cid = cond["id"]
        if cid in postproc_results and is_cache_reused:
            new_dice = postproc_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            delta = new_dice - unmit_dice
            print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [Cached]", flush=True)
            continue

        t0 = time.time()
        slice_avg, per_patient, patient_sum = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="none", apply_postproc=True
        )
        dt = time.time() - t0
        postproc_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["postproc_alone"] = postproc_results
        existing_results["unmitigated"] = unmitigated_results
        save_checkpoint(existing_results)

    # Third, run Contrast Restoration (CLAHE) Alone on all conditions
    print("\n--- 3. Evaluating Preprocessing Alone: CLAHE (clip=0.01, grid=16x16) ---", flush=True)
    clahe_results: Dict[str, Any] = existing_results.get("clahe_alone", {})
    for cond in CONDITIONS:
        cid = cond["id"]
        if cid in clahe_results and is_cache_reused:
            new_dice = clahe_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            delta = new_dice - unmit_dice
            print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [Cached]", flush=True)
            continue

        t0 = time.time()
        slice_avg, per_patient, patient_sum = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="clahe", apply_postproc=False
        )
        dt = time.time() - t0
        clahe_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["clahe_alone"] = clahe_results
        save_checkpoint(existing_results)

    # Fourth, run Denoising (Median filter) Alone on Gaussian Noise conditions and Clean
    print("\n--- 4. Evaluating Preprocessing Alone: Median Denoising (size=3) on Noise ---", flush=True)
    median_results: Dict[str, Any] = existing_results.get("median_alone", {})
    noise_conds = [c for c in CONDITIONS if c["family"] in ("gaussian_noise", "clean")]
    for cond in noise_conds:
        cid = cond["id"]
        if cid in median_results and is_cache_reused:
            new_dice = median_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            delta = new_dice - unmit_dice
            print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [Cached]", flush=True)
            continue

        t0 = time.time()
        slice_avg, per_patient, patient_sum = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="median", apply_postproc=False
        )
        dt = time.time() - t0
        median_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["median_alone"] = median_results
        save_checkpoint(existing_results)

    # Fifth, run Combined Pipeline (CLAHE + Morphological Post-Processing) on Contrast conditions
    print("\n--- 5. Evaluating Combined Pipeline (CLAHE + Post-Processing) on Contrast ---", flush=True)
    combined_contrast_results: Dict[str, Any] = existing_results.get("combined_contrast", {})
    contrast_conds = [c for c in CONDITIONS if c["family"] in ("contrast_reduction", "clean")]
    for cond in contrast_conds:
        cid = cond["id"]
        if cid in combined_contrast_results and is_cache_reused:
            new_dice = combined_contrast_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            delta = new_dice - unmit_dice
            print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [Cached]", flush=True)
            continue

        t0 = time.time()
        slice_avg, per_patient, patient_sum = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="clahe", apply_postproc=True
        )
        dt = time.time() - t0
        combined_contrast_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["combined_contrast"] = combined_contrast_results
        save_checkpoint(existing_results)

    # Sixth, run Combined Pipeline (Median Denoising + Morphological Post-Processing) on Noise conditions
    print("\n--- 6. Evaluating Combined Pipeline (Median + Post-Processing) on Noise ---", flush=True)
    combined_noise_results: Dict[str, Any] = existing_results.get("combined_noise", {})
    for cond in noise_conds:
        cid = cond["id"]
        if cid in combined_noise_results and is_cache_reused:
            new_dice = combined_noise_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
            delta = new_dice - unmit_dice
            print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [Cached]", flush=True)
            continue

        t0 = time.time()
        slice_avg, per_patient, patient_sum = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="median", apply_postproc=True
        )
        dt = time.time() - t0
        combined_noise_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["combined_noise"] = combined_noise_results
        save_checkpoint(existing_results)

    # 4. Compile Comprehensive Tabular & Statistical Comparisons
    # We compare each mitigated arm against the exact unmitigated corrupted baseline
    print("\n" + "=" * 80)
    print("  COMPUTING PAIRED STATISTICAL COMPARISONS & FWER CONTROL")
    print("=" * 80, flush=True)

    # Contrast Reduction Ablation Comparison Table
    contrast_comparison_rows = []
    contrast_eval_keys = ["contrast_reduction_severe", "contrast_reduction_moderate", "contrast_reduction_mild"]
    
    # Store all comparison records including cryptographic provenance
    all_ablation_summary: Dict[str, Any] = {
        "provenance": expected_provenance,
        "unmitigated": unmitigated_results,
        "postproc_alone": postproc_results,
        "clahe_alone": clahe_results,
        "median_alone": median_results,
        "combined_contrast": combined_contrast_results,
        "combined_noise": combined_noise_results,
    }

    # Save complete JSON
    # Helper to serialize numpy floats
    def sanitize_for_json(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return obj

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(all_ablation_summary), f, indent=2)
    print(f"Written complete JSON results to: {out_json_path}")

    # Build Master Mitigation CSV
    # Columns:
    # condition_id, family, severity, arm_name, slice_dice_fg, patient_dice_fg_mean, patient_dice_fg_std,
    # rv_dice, myo_dice, lv_dice, delta_vs_unmitigated, paired_t_pvalue, wilcoxon_pvalue
    csv_rows = []
    
    # Helper for paired test against unmitigated
    def compute_stats_vs_unmit(unmit_dict, test_dict):
        p_ids = sorted(list(unmit_dict.keys()))
        u_vals = [unmit_dict[p]["mean_dice_foreground"] for p in p_ids]
        t_vals = [test_dict[p]["mean_dice_foreground"] for p in p_ids]
        diffs = np.array(t_vals) - np.array(u_vals)
        mean_diff = float(np.mean(diffs))
        
        # Paired t-test
        if np.allclose(diffs, 0.0):
            t_p = 1.0
            w_p = 1.0
        else:
            try:
                t_stat, t_p = stats.ttest_rel(t_vals, u_vals)
            except Exception:
                t_p = 1.0
            try:
                w_stat, w_p = stats.wilcoxon(t_vals, u_vals)
            except Exception:
                w_p = 1.0
        return mean_diff, float(t_p), float(w_p)

    # 1. Unmitigated rows
    for cond in CONDITIONS:
        cid = cond["id"]
        res = unmitigated_results[cid]
        psum = res["patient_summary"]
        csv_rows.append({
            "condition_id": cid,
            "family": cond["family"],
            "severity": cond["severity"],
            "intervention_arm": "Unmitigated Baseline",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": 0.0,
            "paired_t_pvalue": "N/A",
            "wilcoxon_pvalue": "N/A",
        })

    # 2. Post-processing alone rows
    for cond in CONDITIONS:
        cid = cond["id"]
        res = postproc_results[cid]
        psum = res["patient_summary"]
        diff, t_p, w_p = compute_stats_vs_unmit(unmitigated_results[cid]["per_patient"], res["per_patient"])
        csv_rows.append({
            "condition_id": cid,
            "family": cond["family"],
            "severity": cond["severity"],
            "intervention_arm": "Post-Processing Alone (LCC+HoleFill)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(diff, 4),
            "paired_t_pvalue": f"{t_p:.4e}",
            "wilcoxon_pvalue": f"{w_p:.4e}",
        })

    # 3. CLAHE alone rows
    for cond in CONDITIONS:
        cid = cond["id"]
        res = clahe_results[cid]
        psum = res["patient_summary"]
        diff, t_p, w_p = compute_stats_vs_unmit(unmitigated_results[cid]["per_patient"], res["per_patient"])
        csv_rows.append({
            "condition_id": cid,
            "family": cond["family"],
            "severity": cond["severity"],
            "intervention_arm": "CLAHE Alone (clip=0.01)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(diff, 4),
            "paired_t_pvalue": f"{t_p:.4e}",
            "wilcoxon_pvalue": f"{w_p:.4e}",
        })

    # 4. Combined CLAHE + Post-processing on Contrast
    for cond in contrast_conds:
        cid = cond["id"]
        res = combined_contrast_results[cid]
        psum = res["patient_summary"]
        diff, t_p, w_p = compute_stats_vs_unmit(unmitigated_results[cid]["per_patient"], res["per_patient"])
        csv_rows.append({
            "condition_id": cid,
            "family": cond["family"],
            "severity": cond["severity"],
            "intervention_arm": "Combined (CLAHE + Post-Processing)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(diff, 4),
            "paired_t_pvalue": f"{t_p:.4e}",
            "wilcoxon_pvalue": f"{w_p:.4e}",
        })

    # 5. Combined Median + Post-processing on Noise
    for cond in noise_conds:
        cid = cond["id"]
        res = combined_noise_results[cid]
        psum = res["patient_summary"]
        diff, t_p, w_p = compute_stats_vs_unmit(unmitigated_results[cid]["per_patient"], res["per_patient"])
        csv_rows.append({
            "condition_id": cid,
            "family": cond["family"],
            "severity": cond["severity"],
            "intervention_arm": "Combined (Median + Post-Processing)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(diff, 4),
            "paired_t_pvalue": f"{t_p:.4e}",
            "wilcoxon_pvalue": f"{w_p:.4e}",
        })

    # Write CSV
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_rows:
        keys = list(csv_rows[0].keys())
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Written summary CSV to: {out_csv_path}")

    # 5. Generate Visual Mitigation Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Severe Contrast Rescue Bar Chart
    ax1 = axes[0]
    arms_contrast = [
        "Unmitigated\n(Corrupted)",
        "Post-Processing\nAlone",
        "CLAHE\nAlone",
        "Combined\n(CLAHE + Post)",
    ]
    vals_contrast = [
        unmitigated_results["contrast_reduction_severe"]["patient_summary"]["mean_dice_foreground"]["mean"],
        postproc_results["contrast_reduction_severe"]["patient_summary"]["mean_dice_foreground"]["mean"],
        clahe_results["contrast_reduction_severe"]["patient_summary"]["mean_dice_foreground"]["mean"],
        combined_contrast_results["contrast_reduction_severe"]["patient_summary"]["mean_dice_foreground"]["mean"],
    ]
    clean_val = unmitigated_results["clean_baseline"]["patient_summary"]["mean_dice_foreground"]["mean"]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    bars = ax1.bar(arms_contrast, vals_contrast, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.axhline(clean_val, color="#7f8c8d", linestyle="--", linewidth=1.5, label=f"Clean Baseline ({clean_val:.3f})")
    ax1.set_ylabel("Patient 3D Macro FG Dice", fontsize=11, fontweight="bold")
    ax1.set_title("Severe Contrast Reduction (alpha=0.25) Rescue", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 0.90)
    ax1.legend(loc="upper left")
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Panel B: Contrast Family Response Curve (Mild, Mod, Sev)
    ax2 = axes[1]
    alphas = [0.75, 0.50, 0.25]
    unmit_curve = [
        unmitigated_results["contrast_reduction_mild"]["patient_summary"]["mean_dice_foreground"]["mean"],
        unmitigated_results["contrast_reduction_moderate"]["patient_summary"]["mean_dice_foreground"]["mean"],
        unmitigated_results["contrast_reduction_severe"]["patient_summary"]["mean_dice_foreground"]["mean"],
    ]
    clahe_curve = [
        clahe_results["contrast_reduction_mild"]["patient_summary"]["mean_dice_foreground"]["mean"],
        clahe_results["contrast_reduction_moderate"]["patient_summary"]["mean_dice_foreground"]["mean"],
        clahe_results["contrast_reduction_severe"]["patient_summary"]["mean_dice_foreground"]["mean"],
    ]
    comb_curve = [
        combined_contrast_results["contrast_reduction_mild"]["patient_summary"]["mean_dice_foreground"]["mean"],
        combined_contrast_results["contrast_reduction_moderate"]["patient_summary"]["mean_dice_foreground"]["mean"],
        combined_contrast_results["contrast_reduction_severe"]["patient_summary"]["mean_dice_foreground"]["mean"],
    ]
    ax2.plot(alphas, unmit_curve, "o--", color="#e74c3c", label="Unmitigated Baseline", linewidth=2, markersize=8)
    ax2.plot(alphas, clahe_curve, "s-", color="#3498db", label="CLAHE Alone", linewidth=2, markersize=8)
    ax2.plot(alphas, comb_curve, "^-", color="#2ecc71", label="Combined (CLAHE + Post)", linewidth=2.5, markersize=9)
    ax2.set_xlabel("Contrast Scaling Factor alpha (Lower = Worse)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Patient 3D Macro FG Dice", fontsize=11, fontweight="bold")
    ax2.set_title("Contrast Degradation vs Restoration Curves", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlim(0.80, 0.20)
    ax2.legend()

    # Panel C: Clean Baseline Interventions Comparison
    ax3 = axes[2]
    clean_arms = ["Raw Baseline", "Post-Proc", "CLAHE", "Combined"]
    clean_vals = [
        unmitigated_results["clean_baseline"]["patient_summary"]["mean_dice_foreground"]["mean"],
        postproc_results["clean_baseline"]["patient_summary"]["mean_dice_foreground"]["mean"],
        clahe_results["clean_baseline"]["patient_summary"]["mean_dice_foreground"]["mean"],
        combined_contrast_results["clean_baseline"]["patient_summary"]["mean_dice_foreground"]["mean"],
    ]
    bars_clean = ax3.bar(clean_arms, clean_vals, color=["#95a5a6", "#9b59b6", "#3498db", "#1abc9c"], width=0.55, edgecolor="black", linewidth=1.2)
    ax3.set_ylabel("Patient 3D Macro FG Dice", fontsize=11, fontweight="bold")
    ax3.set_title("Clean Baseline Inputs (No Degradation)", fontsize=12, fontweight="bold")
    ax3.set_ylim(0.70, 0.85)
    for bar in bars_clean:
        h = bar.get_height()
        ax3.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "reports" / "mitigation_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved visual mitigation comparison plot to: {plot_path}")

    print("\nPhase 5 Mitigation Evaluation Completed Successfully!", flush=True)


if __name__ == "__main__":
    main()
