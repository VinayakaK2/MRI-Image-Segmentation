"""Full Phase 5 mitigation and ablation evaluation on the held-out test cohort.

Evaluates frozen preprocessing interventions (CLAHE, Median Filter) and
anatomical morphological post-processing across all 10 experimental conditions
on the 15 held-out test patients (131 slices), computing paired statistical tests.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
    classify_significance,
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

# Fixed batch indices in test loader chosen for consistent representative overlay tracking across all conditions
REPRESENTATIVE_BATCH_INDICES = [4, 15, 45]

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


def step_down_holm_monotonic(pvals: List[float]) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Applies Holm-Bonferroni step-down with monotonicity enforcement."""
    m = len(pvals)
    if m == 0:
        return [], []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    running_max = 0.0
    steps_record = []
    final_adj = [0.0] * m

    for rank_idx, (orig_idx, p) in enumerate(indexed):
        mult = m - rank_idx  # 1-based rank i gives mult = m - i + 1
        raw_adj = p * mult
        running_max = max(running_max, raw_adj)
        capped = min(1.0, float(running_max))
        final_adj[orig_idx] = capped
        steps_record.append({
            "rank": rank_idx + 1,
            "orig_idx": orig_idx,
            "raw_p": p,
            "multiplier": mult,
            "raw_times_mult": raw_adj,
            "running_max": running_max,
            "capped_adj_p": capped,
        })
    return final_adj, steps_record


def evaluate_test_pipeline(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    condition: Dict[str, Any],
    preproc_mode: str = "none",  # 'none', 'clahe', 'median'
    apply_postproc: bool = False,
    base_seed: int = 42,
    rep_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Runs inference across the held-out test cohort under a specific intervention configuration."""
    slice_accumulator = MetricAccumulator()
    patient_accumulator = PatientMetricAccumulator(num_classes=4)
    slice_records: List[Dict[str, Any]] = []
    representative_samples: List[Dict[str, Any]] = []
    target_rep_indices = set(rep_indices if rep_indices is not None else REPRESENTATIVE_BATCH_INDICES)

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

            slice_records.append({
                "patient_id": patient_id,
                "slice_idx": int(slice_idx),
                "slice_id": f"{patient_id}_slice{int(slice_idx):02d}",
                "dice_foreground": float(slice_metrics["mean_dice_foreground"]),
                "dice_rv": float(slice_metrics["dice_rv"]),
                "dice_myo": float(slice_metrics["dice_myo"]),
                "dice_lv": float(slice_metrics["dice_lv"]),
            })

            if batch_idx in target_rep_indices:
                representative_samples.append({
                    "batch_idx": batch_idx,
                    "patient_id": patient_id,
                    "slice_idx": int(slice_idx),
                    "slice_id": f"{patient_id}_slice{int(slice_idx):02d}",
                    "corrupt_img": inp.cpu().numpy()[0, 0],
                    "target_mask": target_np,
                    "pred_mask": pred,
                })

    slice_averages = slice_accumulator.compute_averages()
    per_patient = patient_accumulator.compute_per_patient_metrics()
    patient_summary = patient_accumulator.compute_summary_statistics()

    return slice_averages, per_patient, patient_summary, slice_records, representative_samples


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the Phase 5 evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 5 controlled mitigation and ablation evaluation on held-out test cohort."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save all Phase 5 evaluation deliverables.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="phase5_validation_locked_mitigation",
        help="Experiment identifier string.",
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
        help="Path to output results JSON (defaults to reports/mitigation_results.json or output-dir/mitigation_results.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output summary CSV (defaults to reports/mitigation_results.csv or output-dir/mitigation_results.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = get_device("auto")
    print("=" * 80)
    print("  PHASE 5: CONTROLLED MITIGATION & ABLATION EVALUATION (HELD-OUT TEST)")
    print("=" * 80, flush=True)

    # 1. Resolve output directory and target file paths
    output_dir = args.output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json_path = args.output_json or (output_dir / "mitigation_results.json")
        out_csv_path = args.output_csv or (output_dir / "mitigation_results.csv")
        plot_path = output_dir / "mitigation_comparison.png"
    else:
        out_json_path = args.output_json or (PROJECT_ROOT / "reports" / "mitigation_results.json")
        out_csv_path = args.output_csv or (PROJECT_ROOT / "reports" / "mitigation_results.csv")
        plot_path = PROJECT_ROOT / "reports" / "mitigation_comparison.png"

    # Resolve model checkpoint
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Explicitly specified checkpoint not found at: {ckpt_path}")
    else:
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model_frozen.pt"
        if not ckpt_path.exists():
            if (PROJECT_ROOT / "checkpoints" / "50epochs" / "best_model_50epochs.pt").exists():
                ckpt_path = PROJECT_ROOT / "checkpoints" / "50epochs" / "best_model_50epochs.pt"
            elif (PROJECT_ROOT / "checkpoints" / "best_model.pt").exists():
                ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
            else:
                raise FileNotFoundError(f"Default checkpoint not found at: {ckpt_path}")

    config_path = args.config or (PROJECT_ROOT / "configs" / "baseline_unet.yaml")
    manifest_path = args.manifest or (PROJECT_ROOT / "data" / "processed" / "acdc_slice_manifest.json")

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
    target_size = tuple(cfg.get("dataset", {}).get("target_spatial_size", [256, 256]))
    norm_method = cfg.get("preprocessing", {}).get("baseline", {}).get("normalization", "z_score")
    clip_percentiles = tuple(cfg.get("preprocessing", {}).get("baseline", {}).get("clip_percentiles", [0.5, 99.5]))

    test_dataset = ACDCSliceDataset(
        manifest_path=manifest_path,
        split="test",
        target_spatial_size=target_size,
        normalization_method=norm_method,
        clip_percentiles=clip_percentiles,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=acdc_collate_fn)
    print(f"Loaded held-out test split: {len(test_dataset)} slices across 15 patients.\n", flush=True)

    all_slice_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "unmitigated": {},
        "postproc_alone": {},
        "clahe_alone": {},
        "median_alone": {},
        "combined_contrast": {},
        "combined_noise": {},
    }
    rep_samples_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "unmitigated": {},
        "postproc_alone": {},
        "clahe_alone": {},
        "combined_contrast": {},
        "combined_noise": {},
    }

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
        results_dict["experiment_id"] = args.experiment_id
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(sanitize(results_dict), f, indent=2)

    # 1. Unmitigated Baseline across all 10 conditions
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
        slice_avg, per_patient, patient_sum, slice_recs, rep_samps = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="none", apply_postproc=False
        )
        dt = time.time() - t0
        unmitigated_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        all_slice_records["unmitigated"][cid] = slice_recs
        rep_samples_map["unmitigated"][cid] = rep_samps
        fg_slice = slice_avg["mean_dice_foreground"]
        fg_patient = patient_sum["mean_dice_foreground"]["mean"]
        print(f"  [{cid:<28}] Slice FG Dice: {fg_slice:.4f} | Patient 3D Dice: {fg_patient:.4f} [{dt:.1f}s]", flush=True)
        existing_results["unmitigated"] = unmitigated_results
        save_checkpoint(existing_results)

    # 2. Post-Processing Alone across all 10 conditions
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
        slice_avg, per_patient, patient_sum, slice_recs, rep_samps = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="none", apply_postproc=True
        )
        dt = time.time() - t0
        postproc_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        all_slice_records["postproc_alone"][cid] = slice_recs
        rep_samples_map["postproc_alone"][cid] = rep_samps
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["postproc_alone"] = postproc_results
        existing_results["unmitigated"] = unmitigated_results
        save_checkpoint(existing_results)

    # 3. CLAHE Alone across all 10 conditions
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
        slice_avg, per_patient, patient_sum, slice_recs, rep_samps = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="clahe", apply_postproc=False
        )
        dt = time.time() - t0
        clahe_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        all_slice_records["clahe_alone"][cid] = slice_recs
        rep_samples_map["clahe_alone"][cid] = rep_samps
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["clahe_alone"] = clahe_results
        save_checkpoint(existing_results)

    # 4. Median Denoising Alone on Gaussian Noise conditions and Clean
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
        slice_avg, per_patient, patient_sum, slice_recs, rep_samps = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="median", apply_postproc=False
        )
        dt = time.time() - t0
        median_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        all_slice_records["median_alone"][cid] = slice_recs
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["median_alone"] = median_results
        save_checkpoint(existing_results)

    # 5. Combined Pipeline (CLAHE + Morphological Post-Processing) on Contrast conditions
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
        slice_avg, per_patient, patient_sum, slice_recs, rep_samps = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="clahe", apply_postproc=True
        )
        dt = time.time() - t0
        combined_contrast_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        all_slice_records["combined_contrast"][cid] = slice_recs
        rep_samples_map["combined_contrast"][cid] = rep_samps
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["combined_contrast"] = combined_contrast_results
        save_checkpoint(existing_results)

    # 6. Combined Pipeline (Median Denoising + Morphological Post-Processing) on Noise conditions
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
        slice_avg, per_patient, patient_sum, slice_recs, rep_samps = evaluate_test_pipeline(
            model=model, loader=test_loader, device=device, condition=cond, preproc_mode="median", apply_postproc=True
        )
        dt = time.time() - t0
        combined_noise_results[cid] = {
            "slice_level": slice_avg,
            "per_patient": per_patient,
            "patient_summary": patient_sum,
        }
        all_slice_records["combined_noise"][cid] = slice_recs
        rep_samples_map["combined_noise"][cid] = rep_samps
        unmit_dice = unmitigated_results[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        new_dice = patient_sum["mean_dice_foreground"]["mean"]
        delta = new_dice - unmit_dice
        print(f"  [{cid:<28}] Patient 3D Dice: {new_dice:.4f} (Delta vs Unmitigated: {delta:+.4f}) [{dt:.1f}s]", flush=True)
        existing_results["combined_noise"] = combined_noise_results
        save_checkpoint(existing_results)

    # 4. Compile Master JSON Results
    all_ablation_summary: Dict[str, Any] = {
        "provenance": expected_provenance,
        "experiment_id": args.experiment_id,
        "unmitigated": unmitigated_results,
        "postproc_alone": postproc_results,
        "clahe_alone": clahe_results,
        "median_alone": median_results,
        "combined_contrast": combined_contrast_results,
        "combined_noise": combined_noise_results,
    }

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
    print(f"\nWritten complete JSON results to: {out_json_path}")

    # 5. Statistical Computations Across Pre-Specified Families
    print("\n" + "=" * 80)
    print("  COMPUTING PAIRED STATISTICAL COMPARISONS & FWER CONTROL")
    print("=" * 80, flush=True)

    p_ids = sorted(list(unmitigated_results["clean_baseline"]["per_patient"].keys()))
    all_cond_ids = [c["id"] for c in CONDITIONS]
    contrast_ids = ["contrast_reduction_mild", "contrast_reduction_moderate", "contrast_reduction_severe"]
    noise_ids = ["gaussian_noise_mild", "gaussian_noise_moderate", "gaussian_noise_severe"]

    # Helper function for full paired statistics
    def compute_full_stats(u_vals, t_vals):
        diffs = np.array(t_vals) - np.array(u_vals)
        n = len(diffs)
        mean_d = float(np.mean(diffs))
        std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        se_d = std_d / np.sqrt(n) if n > 0 else 0.0
        # 95% CI with t(14) critical value = 2.144787
        t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 1.96
        ci_lower = mean_d - t_crit * se_d
        ci_upper = mean_d + t_crit * se_d
        cohens_d = mean_d / std_d if std_d > 1e-12 else 0.0

        if np.allclose(diffs, 0.0):
            t_stat, t_pval = 0.0, 1.0
            w_stat, w_pval = 0.0, 1.0
        else:
            try:
                t_res = stats.ttest_rel(t_vals, u_vals)
                t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)
            except Exception:
                t_stat, t_pval = 0.0, 1.0
            try:
                w_res = stats.wilcoxon(t_vals, u_vals)
                w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, w_pval = 0.0, 1.0
        return {
            "n": n, "df": n - 1, "mean_u": float(np.mean(u_vals)), "mean_t": float(np.mean(t_vals)),
            "mean_d": mean_d, "std_d": std_d, "se_d": se_d,
            "ci_lower": ci_lower, "ci_upper": ci_upper, "cohens_d": cohens_d,
            "t_stat": t_stat, "t_pval": t_pval,
            "w_stat": w_stat, "w_pval": w_pval,
        }

    # Family 1: Post-processing alone across all conditions (m=10)
    post_stats = [
        compute_full_stats(
            [unmitigated_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
            [postproc_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
        )
        for cid in all_cond_ids
    ]
    post_t_holm, post_t_steps = step_down_holm_monotonic([s["t_pval"] for s in post_stats])
    post_w_holm, post_w_steps = step_down_holm_monotonic([s["w_pval"] for s in post_stats])

    # Family 2: CLAHE alone across contrast conditions (m=3)
    clahe_contrast_stats = [
        compute_full_stats(
            [unmitigated_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
            [clahe_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
        )
        for cid in contrast_ids
    ]
    clahe_t_holm, clahe_t_steps = step_down_holm_monotonic([s["t_pval"] for s in clahe_contrast_stats])
    clahe_w_holm, clahe_w_steps = step_down_holm_monotonic([s["w_pval"] for s in clahe_contrast_stats])

    # Family 3: Combined CLAHE + post-processing across contrast conditions (m=3)
    combc_contrast_stats = [
        compute_full_stats(
            [unmitigated_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
            [combined_contrast_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
        )
        for cid in contrast_ids
    ]
    combc_t_holm, combc_t_steps = step_down_holm_monotonic([s["t_pval"] for s in combc_contrast_stats])
    combc_w_holm, combc_w_steps = step_down_holm_monotonic([s["w_pval"] for s in combc_contrast_stats])

    # Family 4: Combined Median + post-processing across noise conditions (m=3)
    combn_noise_stats = [
        compute_full_stats(
            [unmitigated_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
            [combined_noise_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
        )
        for cid in noise_ids
    ]
    combn_t_holm, combn_t_steps = step_down_holm_monotonic([s["t_pval"] for s in combn_noise_stats])
    combn_w_holm, combn_w_steps = step_down_holm_monotonic([s["w_pval"] for s in combn_noise_stats])

    # 6. Build Master 38-Row Audited CSV
    master_csv_rows: List[Dict[str, Any]] = []

    # 1. Unmitigated rows (10 rows)
    for c in CONDITIONS:
        cid = c["id"]
        res = unmitigated_results[cid]
        psum = res["patient_summary"]
        master_csv_rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "Unmitigated Baseline",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": 0.0,
            "paired_t_pvalue_raw": "N/A",
            "paired_t_pvalue_holm": "N/A",
            "wilcoxon_pvalue_raw": "N/A",
            "wilcoxon_pvalue_holm": "N/A",
            "significance_fwer": "Reference Baseline",
        })

    # 2. Post-processing alone rows (10 rows, m=10 family)
    for i, c in enumerate(CONDITIONS):
        cid = c["id"]
        res = postproc_results[cid]
        psum = res["patient_summary"]
        st = post_stats[i]
        t_r, t_h = st["t_pval"], post_t_holm[i]
        w_r, w_h = st["w_pval"], post_w_holm[i]
        sig = classify_significance(
            paired_t_pvalue_raw=t_r,
            paired_t_pvalue_holm=t_h,
            wilcoxon_pvalue_raw=w_r,
            wilcoxon_pvalue_holm=w_h,
            alpha=0.05,
            policy="primary_t_test",
        )
        master_csv_rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "Post-Processing Alone (LCC+HoleFill)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(st["mean_d"], 4),
            "paired_t_pvalue_raw": f"{t_r:.4e}",
            "paired_t_pvalue_holm": f"{t_h:.4e}",
            "wilcoxon_pvalue_raw": f"{w_r:.4e}",
            "wilcoxon_pvalue_holm": f"{w_h:.4e}",
            "significance_fwer": sig,
        })

    # 3. CLAHE alone rows (10 rows)
    for c in CONDITIONS:
        cid = c["id"]
        res = clahe_results[cid]
        psum = res["patient_summary"]
        if cid in contrast_ids:
            idx = contrast_ids.index(cid)
            st = clahe_contrast_stats[idx]
            t_r, t_h = st["t_pval"], clahe_t_holm[idx]
            w_r, w_h = st["w_pval"], clahe_w_holm[idx]
            sig = classify_significance(
                paired_t_pvalue_raw=t_r,
                paired_t_pvalue_holm=t_h,
                wilcoxon_pvalue_raw=w_r,
                wilcoxon_pvalue_holm=w_h,
                alpha=0.05,
                policy="primary_t_test",
            )
            t_r_str, t_h_str = f"{t_r:.4e}", f"{t_h:.4e}"
            w_r_str, w_h_str = f"{w_r:.4e}", f"{w_h:.4e}"
            d_val = round(st["mean_d"], 4)
        else:
            st = compute_full_stats(
                [unmitigated_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
                [res["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
            )
            t_r_str, t_h_str = f"{st['t_pval']:.4e}", "N/A"
            w_r_str, w_h_str = f"{st['w_pval']:.4e}", "N/A"
            sig = classify_significance(
                paired_t_pvalue_raw=st["t_pval"],
                paired_t_pvalue_holm=None,
                wilcoxon_pvalue_raw=st["w_pval"],
                wilcoxon_pvalue_holm=None,
                alpha=0.05,
                policy="primary_t_test",
            )
            d_val = round(st["mean_d"], 4)

        master_csv_rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "CLAHE Alone (clip=0.01)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": d_val,
            "paired_t_pvalue_raw": t_r_str,
            "paired_t_pvalue_holm": t_h_str,
            "wilcoxon_pvalue_raw": w_r_str,
            "wilcoxon_pvalue_holm": w_h_str,
            "significance_fwer": sig,
        })

    # 4. Combined CLAHE + Post-processing on Contrast (4 rows: clean + 3 contrast)
    for cid in ["clean_baseline"] + contrast_ids:
        c = next(item for item in CONDITIONS if item["id"] == cid)
        res = combined_contrast_results[cid]
        psum = res["patient_summary"]
        if cid in contrast_ids:
            idx = contrast_ids.index(cid)
            st = combc_contrast_stats[idx]
            t_r, t_h = st["t_pval"], combc_t_holm[idx]
            w_r, w_h = st["w_pval"], combc_w_holm[idx]
            sig = classify_significance(
                paired_t_pvalue_raw=t_r,
                paired_t_pvalue_holm=t_h,
                wilcoxon_pvalue_raw=w_r,
                wilcoxon_pvalue_holm=w_h,
                alpha=0.05,
                policy="primary_t_test",
            )
            t_r_str, t_h_str = f"{t_r:.4e}", f"{t_h:.4e}"
            w_r_str, w_h_str = f"{w_r:.4e}", f"{w_h:.4e}"
            d_val = round(st["mean_d"], 4)
        else:
            st = compute_full_stats(
                [unmitigated_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
                [res["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
            )
            t_r_str, t_h_str = f"{st['t_pval']:.4e}", "N/A"
            w_r_str, w_h_str = f"{st['w_pval']:.4e}", "N/A"
            sig = classify_significance(
                paired_t_pvalue_raw=st["t_pval"],
                paired_t_pvalue_holm=None,
                wilcoxon_pvalue_raw=st["w_pval"],
                wilcoxon_pvalue_holm=None,
                alpha=0.05,
                policy="primary_t_test",
            )
            d_val = round(st["mean_d"], 4)

        master_csv_rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "Combined (CLAHE + Post-Processing)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": d_val,
            "paired_t_pvalue_raw": t_r_str,
            "paired_t_pvalue_holm": t_h_str,
            "wilcoxon_pvalue_raw": w_r_str,
            "wilcoxon_pvalue_holm": w_h_str,
            "significance_fwer": sig,
        })

    # 5. Combined Median + Post-processing on Noise (4 rows: clean + 3 noise)
    for cid in ["clean_baseline"] + noise_ids:
        c = next(item for item in CONDITIONS if item["id"] == cid)
        res = combined_noise_results[cid]
        psum = res["patient_summary"]
        if cid in noise_ids:
            idx = noise_ids.index(cid)
            st = combn_noise_stats[idx]
            t_r, t_h = st["t_pval"], combn_t_holm[idx]
            w_r, w_h = st["w_pval"], combn_w_holm[idx]
            sig = classify_significance(
                paired_t_pvalue_raw=t_r,
                paired_t_pvalue_holm=t_h,
                wilcoxon_pvalue_raw=w_r,
                wilcoxon_pvalue_holm=w_h,
                alpha=0.05,
                policy="primary_t_test",
            )
            t_r_str, t_h_str = f"{t_r:.4e}", f"{t_h:.4e}"
            w_r_str, w_h_str = f"{w_r:.4e}", f"{w_h:.4e}"
            d_val = round(st["mean_d"], 4)
        else:
            st = compute_full_stats(
                [unmitigated_results[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
                [res["per_patient"][p]["mean_dice_foreground"] for p in p_ids],
            )
            t_r_str, t_h_str = f"{st['t_pval']:.4e}", "N/A"
            w_r_str, w_h_str = f"{st['w_pval']:.4e}", "N/A"
            sig = classify_significance(
                paired_t_pvalue_raw=st["t_pval"],
                paired_t_pvalue_holm=None,
                wilcoxon_pvalue_raw=st["w_pval"],
                wilcoxon_pvalue_holm=None,
                alpha=0.05,
                policy="primary_t_test",
            )
            d_val = round(st["mean_d"], 4)

        master_csv_rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "Combined (Median + Post-Processing)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": d_val,
            "paired_t_pvalue_raw": t_r_str,
            "paired_t_pvalue_holm": t_h_str,
            "wilcoxon_pvalue_raw": w_r_str,
            "wilcoxon_pvalue_holm": w_h_str,
            "significance_fwer": sig,
        })

    # Write Master CSV (38 rows)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(master_csv_rows)
    print(f"Written master audited CSV ({len(master_csv_rows)} rows) to: {out_csv_path}")

    # 7. Write Specialized CSV Artifacts
    # Primary Hypotheses CSV (19 rows)
    primary_hypotheses_rows: List[Dict[str, Any]] = []
    for r in master_csv_rows:
        arm = r["intervention_arm"]
        cid = r["condition_id"]
        is_fam1 = (arm == "Post-Processing Alone (LCC+HoleFill)")
        is_fam2 = (arm == "CLAHE Alone (clip=0.01)" and cid in contrast_ids)
        is_fam3 = (arm == "Combined (CLAHE + Post-Processing)" and cid in contrast_ids)
        is_fam4 = (arm == "Combined (Median + Post-Processing)" and cid in noise_ids)
        if is_fam1 or is_fam2 or is_fam3 or is_fam4:
            fam_name = (
                "Family 1: Post-Processing Alone (m=10)" if is_fam1 else
                "Family 2: CLAHE Alone on Contrast (m=3)" if is_fam2 else
                "Family 3: Combined CLAHE + Post-Processing on Contrast (m=3)" if is_fam3 else
                "Family 4: Combined Median + Post-Processing on Noise (m=3)"
            )
            r_copy = dict(r)
            r_copy["hypothesis_family"] = fam_name
            primary_hypotheses_rows.append(r_copy)

    # Negative Control CSV
    negative_control_rows: List[Dict[str, Any]] = []
    for r in master_csv_rows:
        arm = r["intervention_arm"]
        cid = r["condition_id"]
        is_clahe_ctrl = (arm == "CLAHE Alone (clip=0.01)" and cid not in contrast_ids)
        is_combc_ctrl = (arm == "Combined (CLAHE + Post-Processing)" and cid == "clean_baseline")
        is_combn_ctrl = (arm == "Combined (Median + Post-Processing)" and cid == "clean_baseline")
        if is_clahe_ctrl or is_combc_ctrl or is_combn_ctrl:
            ctrl_type = (
                "Off-Target Preprocessing (Clean Baseline)" if cid == "clean_baseline" else
                "Off-Target Preprocessing (Noise)" if "noise" in cid else
                "Off-Target Preprocessing (Blur)"
            )
            r_copy = dict(r)
            r_copy["control_classification"] = ctrl_type
            negative_control_rows.append(r_copy)

    # Target directory for auxiliary files
    target_dir = output_dir or (PROJECT_ROOT / "reports")
    target_dir.mkdir(parents=True, exist_ok=True)

    prim_csv_path = target_dir / "primary_hypotheses_results.csv"
    with open(prim_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(primary_hypotheses_rows[0].keys()))
        writer.writeheader()
        writer.writerows(primary_hypotheses_rows)
    print(f"Written primary hypotheses CSV ({len(primary_hypotheses_rows)} rows) to: {prim_csv_path}")

    neg_csv_path = target_dir / "negative_control_results.csv"
    with open(neg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(negative_control_rows[0].keys()))
        writer.writeheader()
        writer.writerows(negative_control_rows)
    print(f"Written negative control CSV ({len(negative_control_rows)} rows) to: {neg_csv_path}")

    # Patient Level Results CSV
    patient_level_rows: List[Dict[str, Any]] = []
    for arm_name, arm_label in [
        ("unmitigated", "Unmitigated Baseline"),
        ("postproc_alone", "Post-Processing Alone (LCC+HoleFill)"),
        ("clahe_alone", "CLAHE Alone (clip=0.01)"),
        ("median_alone", "Median Alone (size=3)"),
        ("combined_contrast", "Combined (CLAHE + Post-Processing)"),
        ("combined_noise", "Combined (Median + Post-Processing)"),
    ]:
        arm_dict = all_ablation_summary.get(arm_name, {})
        for cid, c_data in arm_dict.items():
            per_pt = c_data.get("per_patient", {})
            for pid, p_metrics in per_pt.items():
                patient_level_rows.append({
                    "intervention_arm": arm_label,
                    "condition_id": cid,
                    "patient_id": pid,
                    "dice_foreground": round(p_metrics.get("mean_dice_foreground", 0.0), 4),
                    "dice_rv": round(p_metrics.get("dice_rv", 0.0), 4),
                    "dice_myo": round(p_metrics.get("dice_myo", 0.0), 4),
                    "dice_lv": round(p_metrics.get("dice_lv", 0.0), 4),
                    "iou_foreground": round(p_metrics.get("mean_iou_foreground", 0.0), 4),
                })
    pat_csv_path = target_dir / "patient_level_results.csv"
    if patient_level_rows:
        with open(pat_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(patient_level_rows[0].keys()))
            writer.writeheader()
            writer.writerows(patient_level_rows)
        print(f"Written patient-level CSV ({len(patient_level_rows)} rows) to: {pat_csv_path}")

    # Slice Level Results CSV
    slice_level_rows: List[Dict[str, Any]] = []
    for arm_name, arm_label in [
        ("unmitigated", "Unmitigated Baseline"),
        ("postproc_alone", "Post-Processing Alone (LCC+HoleFill)"),
        ("clahe_alone", "CLAHE Alone (clip=0.01)"),
        ("median_alone", "Median Alone (size=3)"),
        ("combined_contrast", "Combined (CLAHE + Post-Processing)"),
        ("combined_noise", "Combined (Median + Post-Processing)"),
    ]:
        for cid, sl_list in all_slice_records.get(arm_name, {}).items():
            for sl in sl_list:
                slice_level_rows.append({
                    "intervention_arm": arm_label,
                    "condition_id": cid,
                    "patient_id": sl["patient_id"],
                    "slice_idx": sl["slice_idx"],
                    "slice_id": sl["slice_id"],
                    "dice_foreground": round(sl["dice_foreground"], 4),
                    "dice_rv": round(sl["dice_rv"], 4),
                    "dice_myo": round(sl["dice_myo"], 4),
                    "dice_lv": round(sl["dice_lv"], 4),
                })
    sl_csv_path = target_dir / "slice_level_results.csv"
    if slice_level_rows:
        with open(sl_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(slice_level_rows[0].keys()))
            writer.writeheader()
            writer.writerows(slice_level_rows)
        print(f"Written slice-level CSV ({len(slice_level_rows)} rows) to: {sl_csv_path}")
    elif sl_csv_path.exists():
        print(f"Preserved existing slice-level CSV on cache hit: {sl_csv_path}")
    else:
        print("-> Cache hit active but slice_level_results.csv missing. Regenerating slice records...", flush=True)
        for arm_name, pmode, apost, arm_label in [
            ("unmitigated", "none", False, "Unmitigated Baseline"),
            ("postproc_alone", "none", True, "Post-Processing Alone (LCC+HoleFill)"),
            ("clahe_alone", "clahe", False, "CLAHE Alone (clip=0.01)"),
            ("median_alone", "median", False, "Median Alone (size=3)"),
            ("combined_contrast", "clahe", True, "Combined (CLAHE + Post-Processing)"),
            ("combined_noise", "median", True, "Combined (Median + Post-Processing)"),
        ]:
            eval_conds = CONDITIONS
            if arm_name == "combined_contrast":
                eval_conds = [c for c in CONDITIONS if c["family"] == "contrast_reduction"]
            elif arm_name == "combined_noise":
                eval_conds = [c for c in CONDITIONS if c["family"] == "gaussian_noise"]
            for cond in eval_conds:
                cid = cond["id"]
                _, _, _, s_recs, _ = evaluate_test_pipeline(
                    model=model, loader=test_loader, device=device, condition=cond, preproc_mode=pmode, apply_postproc=apost
                )
                for sl in s_recs:
                    slice_level_rows.append({
                        "intervention_arm": arm_label,
                        "condition_id": cid,
                        "patient_id": sl["patient_id"],
                        "slice_idx": sl["slice_idx"],
                        "slice_id": sl["slice_id"],
                        "dice_foreground": round(sl["dice_foreground"], 4),
                        "dice_rv": round(sl["dice_rv"], 4),
                        "dice_myo": round(sl["dice_myo"], 4),
                        "dice_lv": round(sl["dice_lv"], 4),
                    })
        if slice_level_rows:
            with open(sl_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(slice_level_rows[0].keys()))
                writer.writeheader()
                writer.writerows(slice_level_rows)
            print(f"Regenerated and wrote slice-level CSV ({len(slice_level_rows)} rows) to: {sl_csv_path}")

    # Detailed Statistical Tests CSV (19 rows)
    stat_test_rows: List[Dict[str, Any]] = []
    for fam_idx, (fam_name, c_list, st_list, t_h_list, w_h_list, arm_title) in enumerate([
        ("Family 1 (Post-Proc, m=10)", all_cond_ids, post_stats, post_t_holm, post_w_holm, "Post-Processing Alone (LCC+HoleFill)"),
        ("Family 2 (CLAHE, m=3)", contrast_ids, clahe_contrast_stats, clahe_t_holm, clahe_w_holm, "CLAHE Alone (clip=0.01)"),
        ("Family 3 (Combined Contrast, m=3)", contrast_ids, combc_contrast_stats, combc_t_holm, combc_w_holm, "Combined (CLAHE + Post-Processing)"),
        ("Family 4 (Combined Noise, m=3)", noise_ids, combn_noise_stats, combn_t_holm, combn_w_holm, "Combined (Median + Post-Processing)"),
    ]):
        for idx, cid in enumerate(c_list):
            st = st_list[idx]
            t_r, t_h = st["t_pval"], t_h_list[idx]
            w_r, w_h = st["w_pval"], w_h_list[idx]
            sig = classify_significance(
                paired_t_pvalue_raw=t_r,
                paired_t_pvalue_holm=t_h,
                wilcoxon_pvalue_raw=w_r,
                wilcoxon_pvalue_holm=w_h,
                alpha=0.05,
                policy="primary_t_test",
            )
            stat_test_rows.append({
                "family_name": fam_name,
                "condition_id": cid,
                "intervention_arm": arm_title,
                "n_patients": st["n"],
                "degrees_of_freedom": st["df"],
                "mean_baseline_dice": round(st["mean_u"], 4),
                "mean_mitigated_dice": round(st["mean_t"], 4),
                "mean_difference": round(st["mean_d"], 4),
                "std_difference": round(st["std_d"], 4),
                "se_difference": round(st["se_d"], 4),
                "ci_95_lower": round(st["ci_lower"], 4),
                "ci_95_upper": round(st["ci_upper"], 4),
                "cohens_d": round(st["cohens_d"], 4),
                "paired_t_statistic": round(st["t_stat"], 4),
                "paired_t_pvalue_raw": f"{t_r:.4e}",
                "paired_t_pvalue_holm": f"{t_h:.4e}",
                "wilcoxon_w_statistic": round(st["w_stat"], 1),
                "wilcoxon_pvalue_raw": f"{w_r:.4e}",
                "wilcoxon_pvalue_holm": f"{w_h:.4e}",
                "significance_decision": sig,
            })
    stat_csv_path = target_dir / "statistical_tests.csv"
    with open(stat_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stat_test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stat_test_rows)
    print(f"Written detailed statistical tests CSV ({len(stat_test_rows)} rows) to: {stat_csv_path}")

    # Detailed Holm Step-Down Audit CSV
    holm_detail_rows: List[Dict[str, Any]] = []
    for fam_name, c_list, t_steps, w_steps in [
        ("Family 1 (Post-Proc, m=10)", all_cond_ids, post_t_steps, post_w_steps),
        ("Family 2 (CLAHE, m=3)", contrast_ids, clahe_t_steps, clahe_w_steps),
        ("Family 3 (Combined Contrast, m=3)", contrast_ids, combc_t_steps, combc_w_steps),
        ("Family 4 (Combined Noise, m=3)", noise_ids, combn_t_steps, combn_w_steps),
    ]:
        for step in t_steps:
            cid = c_list[step["orig_idx"]]
            holm_detail_rows.append({
                "family_name": fam_name,
                "family_size_m": len(c_list),
                "test_type": "paired_student_t",
                "rank_i": step["rank"],
                "condition_id": cid,
                "multiplier": step["multiplier"],
                "pvalue_raw": f"{step['raw_p']:.4e}",
                "pvalue_raw_times_mult": f"{step['raw_times_mult']:.4e}",
                "running_max": f"{step['running_max']:.4e}",
                "pvalue_holm_adjusted": f"{step['capped_adj_p']:.4e}",
                "significant_at_alpha_0_05": step["capped_adj_p"] < 0.05,
            })
        for step in w_steps:
            cid = c_list[step["orig_idx"]]
            holm_detail_rows.append({
                "family_name": fam_name,
                "family_size_m": len(c_list),
                "test_type": "wilcoxon_signed_rank",
                "rank_i": step["rank"],
                "condition_id": cid,
                "multiplier": step["multiplier"],
                "pvalue_raw": f"{step['raw_p']:.4e}",
                "pvalue_raw_times_mult": f"{step['raw_times_mult']:.4e}",
                "running_max": f"{step['running_max']:.4e}",
                "pvalue_holm_adjusted": f"{step['capped_adj_p']:.4e}",
                "significant_at_alpha_0_05": step["capped_adj_p"] < 0.05,
            })
    holm_csv_path = target_dir / "holm_adjusted_results.csv"
    with open(holm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(holm_detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(holm_detail_rows)
    print(f"Written detailed Holm adjusted CSV ({len(holm_detail_rows)} rows) to: {holm_csv_path}")

    # 8. Manifests and Configuration Snapshots
    # Get git commit hash safely
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT)).decode().strip()
    except Exception:
        git_commit = "unknown"

    reproducibility_manifest = {
        "experiment_id": args.experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": expected_provenance["checkpoint_sha256"],
        },
        "config": {
            "path": str(config_path),
            "sha256": expected_provenance["config_sha256"],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": expected_provenance["manifest_sha256"],
        },
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "scipy_version": stats.__version__ if hasattr(stats, "__version__") else "1.10.x",
            "device": str(device),
            "base_seed": 42,
        },
        "test_cohort": {
            "num_patients": len(p_ids),
            "patient_ids": p_ids,
            "total_slices": len(test_dataset),
        },
        "locked_mitigation_hyperparameters": {
            "clahe_clip_limit": 0.01,
            "clahe_grid_size": [16, 16],
            "clahe_re_standardize": True,
            "median_filter_size": 3,
            "median_re_standardize": False,
            "morphology_min_area": {"RV": 30, "MYO": 50, "LV": 30},
            "myo_max_hole_size": 16,
        },
        "statistical_protocol": {
            "primary_test": "two-tailed paired Student's t-test",
            "sensitivity_test": "two-tailed Wilcoxon signed-rank test",
            "fwer_correction": "Holm-Bonferroni step-down with monotonicity enforcement",
            "alpha": 0.05,
            "comparison_families_m": {"Family 1": 10, "Family 2": 3, "Family 3": 3, "Family 4": 3},
        },
    }
    with open(target_dir / "reproducibility_manifest.json", "w", encoding="utf-8") as f:
        json.dump(reproducibility_manifest, f, indent=2)
    print(f"Written reproducibility manifest to: {target_dir / 'reproducibility_manifest.json'}")

    with open(target_dir / "checkpoint_sha256.txt", "w", encoding="utf-8") as f:
        f.write(expected_provenance["checkpoint_sha256"] + "\n")
    print(f"Written checkpoint SHA-256 to: {target_dir / 'checkpoint_sha256.txt'}")

    cohort_manifest = {
        "num_patients": len(p_ids),
        "patient_ids": p_ids,
        "total_slices": len(test_dataset),
        "target_spatial_size": cfg["dataset"]["target_spatial_size"],
    }
    with open(target_dir / "cohort_manifest.json", "w", encoding="utf-8") as f:
        json.dump(cohort_manifest, f, indent=2)
    print(f"Written cohort manifest to: {target_dir / 'cohort_manifest.json'}")

    config_snapshot = {
        "model_config": cfg,
        "mitigation_hyperparameters": reproducibility_manifest["locked_mitigation_hyperparameters"],
    }
    with open(target_dir / "experiment_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)
    print(f"Written config snapshot to: {target_dir / 'experiment_config_snapshot.json'}")

    # 9. Visualizations
    # Figure 1: Mitigation Comparison Plot (3-panel)
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
    ax1.set_ylim(0, 1.0)
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
    ax3.set_ylim(0.70, 1.0)
    for bar in bars_clean:
        h = bar.get_height()
        ax3.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved visual mitigation comparison plot to: {plot_path}")

    # Figure 2: Class-Wise Mitigation Effect (RV, MYO, LV)
    vis_dir = target_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig_cls, axes_cls = plt.subplots(1, 3, figsize=(18, 5))
    classes = [("dice_rv", "Right Ventricle (RV)", axes_cls[0]),
               ("dice_myo", "Myocardium (MYO)", axes_cls[1]),
               ("dice_lv", "Left Ventricle (LV)", axes_cls[2])]

    for metric_key, class_title, ax in classes:
        unmit_vals = [unmitigated_results[c]["patient_summary"].get(metric_key, {}).get("mean", 0.0) for c in contrast_ids]
        clahe_vals = [clahe_results[c]["patient_summary"].get(metric_key, {}).get("mean", 0.0) for c in contrast_ids]
        comb_vals = [combined_contrast_results[c]["patient_summary"].get(metric_key, {}).get("mean", 0.0) for c in contrast_ids]

        x = np.arange(len(contrast_ids))
        w = 0.25
        ax.bar(x - w, unmit_vals, width=w, label="Unmitigated", color="#e74c3c", edgecolor="black")
        ax.bar(x, clahe_vals, width=w, label="CLAHE Alone", color="#3498db", edgecolor="black")
        ax.bar(x + w, comb_vals, width=w, label="Combined", color="#2ecc71", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["Mild", "Moderate", "Severe"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Patient 3D Dice", fontsize=11, fontweight="bold")
        ax.set_title(class_title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.legend(loc="lower left")

    plt.suptitle("Class-Wise Mitigation Performance Across Contrast Reduction Severities", fontsize=14, y=1.02)
    plt.tight_layout()
    cls_plot_path = vis_dir / "class_wise_mitigation_effect.png"
    plt.savefig(cls_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved class-wise mitigation effect plot to: {cls_plot_path}")

    # Figure 3: Representative Overlays (saved to visualizations/overlays and representative_overlays/)
    overlays_dir1 = vis_dir / "overlays"
    overlays_dir2 = target_dir / "representative_overlays"
    overlays_dir1.mkdir(parents=True, exist_ok=True)
    overlays_dir2.mkdir(parents=True, exist_ok=True)

    # For each representative sample index
    num_samples = len(rep_samples_map.get("unmitigated", {}).get("clean_baseline", []))
    if num_samples == 0:
        print("-> Cache hit active: regenerating representative slice samples for overlays...", flush=True)
        cond_dict_by_id = {c["id"]: c for c in CONDITIONS}
        rep_cond_configs = [
            ("unmitigated", "clean_baseline", "none", False),
            ("unmitigated", "contrast_reduction_severe", "none", False),
            ("unmitigated", "gaussian_noise_severe", "none", False),
            ("unmitigated", "gaussian_blur_severe", "none", False),
            ("postproc_alone", "clean_baseline", "none", True),
            ("postproc_alone", "gaussian_blur_severe", "none", True),
            ("combined_contrast", "contrast_reduction_severe", "clahe", True),
            ("combined_noise", "gaussian_noise_severe", "median", True),
        ]
        for arm_name, cid, pmode, apost in rep_cond_configs:
            if cid in cond_dict_by_id:
                cond = cond_dict_by_id[cid]
                _, _, _, _, s_reps = evaluate_test_pipeline(
                    model=model, loader=test_loader, device=device, condition=cond, preproc_mode=pmode, apply_postproc=apost, rep_indices=REPRESENTATIVE_BATCH_INDICES
                )
                if arm_name not in rep_samples_map:
                    rep_samples_map[arm_name] = {}
                rep_samples_map[arm_name][cid] = s_reps
        num_samples = len(rep_samples_map.get("unmitigated", {}).get("clean_baseline", []))

    for s_idx in range(num_samples):
        s_clean = rep_samples_map["unmitigated"]["clean_baseline"][s_idx]
        p_id = s_clean["patient_id"]
        sl_idx = s_clean["slice_idx"]

        # Columns: Clean, Severe Contrast, Severe Noise, Severe Blur
        cols_info = [
            ("Clean Baseline", "unmitigated", "clean_baseline", "postproc_alone"),
            ("Severe Contrast (0.25)", "unmitigated", "contrast_reduction_severe", "combined_contrast"),
            ("Severe Noise (0.10)", "unmitigated", "gaussian_noise_severe", "combined_noise"),
            ("Severe Blur (2.50)", "unmitigated", "gaussian_blur_severe", "postproc_alone"),
        ]

        fig_ov, axes_ov = plt.subplots(4, 4, figsize=(16, 15))
        for col, (col_title, raw_arm, cond_key, mit_arm) in enumerate(cols_info):
            s_raw = rep_samples_map.get(raw_arm, {}).get(cond_key, [None])[s_idx] if len(rep_samples_map.get(raw_arm, {}).get(cond_key, [])) > s_idx else None
            s_mit = rep_samples_map.get(mit_arm, {}).get(cond_key, [None])[s_idx] if len(rep_samples_map.get(mit_arm, {}).get(cond_key, [])) > s_idx else None

            if s_raw is not None:
                img = s_raw["corrupt_img"]
                tgt = s_raw["target_mask"]
                pred_raw = s_raw["pred_mask"]
                pred_mit = s_mit["pred_mask"] if s_mit is not None else pred_raw

                # Row 0: MRI Slice
                axes_ov[0, col].imshow(img, cmap="gray")
                axes_ov[0, col].set_title(f"{col_title}\nInput Slice", fontsize=11, fontweight="bold")
                axes_ov[0, col].axis("off")

                # Row 1: Ground Truth
                axes_ov[1, col].imshow(img, cmap="gray")
                axes_ov[1, col].imshow(tgt, cmap="autumn", alpha=0.5, interpolation="none")
                axes_ov[1, col].set_title("Ground Truth Mask", fontsize=10)
                axes_ov[1, col].axis("off")

                # Row 2: Unmitigated Model Prediction
                axes_ov[2, col].imshow(img, cmap="gray")
                axes_ov[2, col].imshow(pred_raw, cmap="winter", alpha=0.5, interpolation="none")
                axes_ov[2, col].set_title("Unmitigated Prediction", fontsize=10)
                axes_ov[2, col].axis("off")

                # Row 3: Mitigated Model Prediction
                axes_ov[3, col].imshow(img, cmap="gray")
                axes_ov[3, col].imshow(pred_mit, cmap="winter", alpha=0.5, interpolation="none")
                axes_ov[3, col].set_title(f"Mitigated Prediction\n({mit_arm})", fontsize=10)
                axes_ov[3, col].axis("off")

        plt.suptitle(f"Cardiac MRI Segmentation: Mitigation Impact ({p_id}, slice {sl_idx})", fontsize=14, y=0.98)
        plt.tight_layout()
        fname = f"overlay_{p_id}_slice{sl_idx:02d}.png"
        fig_ov.savefig(overlays_dir1 / fname, dpi=200)
        fig_ov.savefig(overlays_dir2 / fname, dpi=200)
        plt.close(fig_ov)
        print(f"Saved overlay visual to: {overlays_dir1 / fname}")

    print("\nPhase 5 Mitigation Evaluation Completed Successfully!", flush=True)


if __name__ == "__main__":
    main()
